from typing import Callable, Iterable, Sequence, Tuple, Optional
from functools import partial
import warnings

import torch
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import default_collate, T_co

from timm.data.loader import fast_collate, _worker_init, PrefetchLoader

from PSZS.datasets.datasets import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PSZS.datasets import DatasetDescriptor
from PSZS.Utils import send_to_device

def build_dataloader(
        dataset,
        input_size: int|Tuple[int, int]|Tuple[int, int, int],
        batch_size: int,
        is_training: bool,
        re_prob: float = 0.,
        re_mode: str = 'const',
        re_count: int = 1,
        re_num_splits: int = 0,
        mean: Tuple[float] = IMAGENET_DEFAULT_MEAN,
        std: Tuple[float] = IMAGENET_DEFAULT_STD,
        num_workers: int = 2,
        collate_fn: Optional[Callable] = None,
        img_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cuda'),
        use_prefetcher: bool = True,
        shuffle: Optional[bool] = None,
        drop_last: bool = False,
        persistent_workers: bool = True,
        batch_sampler: Optional[Sampler | Iterable] = None,
):
    """

    Args:
        dataset: The image dataset to load.
        input_size: Target input size (channels, height, width) tuple or size scalar.
        batch_size: Number of samples in a batch.
        is_training: Return training (random) transforms.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        num_workers: Num worker processes per DataLoader.
        collate_fn: Override default collate_fn.
        img_dtype: Data type for input image.
        device: Device to transfer inputs and targets to.
        use_prefetcher: Use efficient pre-fetcher to load samples onto device.
        shuffle: Shuffle data in dataloader. If not set defaults to 'is_training'
        drop_last: Drop last batch if not of batch_size. If not set defaults to 'is_training'
        persistent_workers: Enable persistent worker processes.
        batch_sampler: Sampler to use for dataloader.
    Returns:
        DataLoader
    """
    assert not isinstance(dataset, IterableDataset), "IterableDataset are not supported"
    
    if collate_fn is None:
        collate_fn = fast_collate if use_prefetcher else default_collate

    loader_class = DataLoader
    
    # Shuffler, Drop_last and batch_size are mutually exclusive with Batch_Sampler
    # Batch_Sampler has higher priority everything else.
    # Otherwise shuffler and drop_last are set based on is_training (if not given otherwise)
    if batch_sampler is not None:
        if shuffle or is_training:
            # Warning vs just printing as info
            if shuffle is None:
                print("Batch Sampler is given, shuffle will be set to False.")
            else:
                warnings.warn("Batch Sampler is given, shuffle will be set to False.")
            shuffle = False
        if drop_last or is_training:
            # Warning vs just printing as info
            if drop_last is None:
                print("Batch Sampler is given, drop_last will be set to False.")
            else:
                warnings.warn("Batch Sampler is given, drop_last will be set to False.")
            drop_last = False
        batch_size = 1 # Default value for batch_size in DataLoader class

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        worker_init_fn=partial(_worker_init, worker_seeding='all'),
        persistent_workers=persistent_workers,
        batch_sampler=batch_sampler,
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        print("Couldn't instantiate dataloader with persistent_workers. Try without persistent_workers")
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    if isinstance(input_size, int):
        channels = 3
    elif len(input_size==3):
        channels = input_size[0]
    else:
        channels = 3
    if use_prefetcher:
        # In an addition check for no_aug param is given
        prefetch_re_prob = re_prob if is_training else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=channels,
            device=device,
            img_dtype=img_dtype,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""
    
    def _load_next_to_device(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data
    
    def _load_next_on_device(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __init__(self, data_loader: DataLoader | PrefetchLoader, device=None, on_device: bool = False):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device
        self.next_fn = self._load_next_on_device if on_device else self._load_next_to_device
        self.num_classes = self.get_num_classes()
        self.dataset_descriptor : DatasetDescriptor | None = getattr(self.dataset, 'descriptor', None)

    def __next__(self):
        return self.next_fn()

    def __len__(self):
        return len(self.data_loader)
    
    @property
    def dataset(self) -> Dataset[T_co]:
        return self.data_loader.dataset
    
    @property
    def batch_size(self) -> int:
        # Prefetch Loader do not have a direct batch_size attribute
        if isinstance(self.data_loader, PrefetchLoader):
            bs = self.data_loader.loader.batch_size
            # When batch_sampler is used directly retrieve it from there
            if bs is None:
                bs = getattr(self.data_loader.loader.batch_sampler, 'batch_size', None)
        else:
            bs = self.data_loader.batch_size
            # When batch_sampler is used directly retrieve it from there
            if bs is None:
                bs = getattr(self.data_loader.batch_sampler, 'batch_size', None)
        return bs
    
    def get_num_classes(self) -> int:
        if getattr(self.dataset, 'num_all_classes', None) is not None:
            return self.dataset.num_all_classes
        if getattr(self.dataset, 'num_classes', None) is not None:
            num_classes = self.dataset.num_classes
            if isinstance(num_classes, Iterable):
                num_classes = sum(num_classes)
            assert isinstance(num_classes, int), f"num_classes should be an int (after optional sum), but got {type(num_classes)}"
            return num_classes
        if getattr(self.dataset, 'classes', None) is not None:
            classes = self.dataset.classes
            assert isinstance(classes, Sequence), f"classes should be a Sequence, but got {type(classes)}"
            assert isinstance(classes[0], Sequence)==False, f"classes should be a single Sequence, but got inner type {type(classes[0])}"
            return len(classes)
        raise AttributeError("Dataset does not have num_classes, num_all_classes or classes attribute. Can not determine num_classes directly.")
    