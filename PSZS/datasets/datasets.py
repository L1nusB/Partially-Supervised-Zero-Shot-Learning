from collections import defaultdict
from typing import Iterable, Sequence, Optional, List, Tuple
import warnings
import math
from os import PathLike
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import ConcatDataset as _ConcatDataset

from timm.data.transforms_factory import create_transform

import PSZS.datasets as datasets
from PSZS.datasets import CustomDataset, DatasetDescriptor

from PSZS.Utils.transformations import _get_resizing_transform

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

__all__ = ['ConcatDataset', 'get_dataset_names', 'get_dataset', 
           'transform_target', 'build_remapped_descriptors', 'build_transform',
           'build_descriptors', 'build_descriptor']

def flatten_nested_list(nested_list : Sequence, make_unique: bool = True) -> List:
        flattened_list = []
        for item in nested_list:
            # isinstance(Sequence) does not work here as str is also a sequence
            if isinstance(item, list) or isinstance(item, tuple) or isinstance(item, set):
                flattened_list.extend(flatten_nested_list(item))
            else:
                flattened_list.append(item)
        # Remove duplicates
        if make_unique:
            return [*dict.fromkeys(flattened_list)]
        else:
            return flattened_list

# Wrapper around ConcatDataset to allow uniform access to .classes attribute
class ConcatDataset(_ConcatDataset):
    def __init__(self, 
                 datasets: Iterable[CustomDataset],
                 eval_class_index: int = -1):
        super().__init__(datasets)
        assert all([ds.multi_label for ds in datasets]) or all([not ds.multi_label for ds in datasets]), \
            'Multi-label and single-label datasets can not be combined'
        assert all([ds.descriptor == self.datasets[0].descriptor for ds in datasets]), \
            'All datasets must have the same dataset descriptor'
        # Already sure that all datasets have same value for multi_label and descriptor
        self.datasets : List[CustomDataset]
        self.multi_label = self.datasets[0].multi_label
        self.classes = [dataset.classes for dataset in self.datasets]
        self.num_classes = [dataset.num_classes for dataset in self.datasets]
        self.eval_class_index = eval_class_index
        self.main_class_index = self._get_main_class_index()
        if self.multi_label:
            self.all_classes : List[List[str]] = [flatten_nested_list(label_level_classes) for label_level_classes in [
                                                    [cl[level] for cl in self.classes] for level in range(len(self.classes[0]))
                                                    ]]
            self.num_all_classes = [len(classes) for classes in self.all_classes]
        else:
            self.all_classes : List[str] = flatten_nested_list(self.classes)
            self.num_all_classes = len(self.all_classes)
        self.label_index = self.merge_label_index()
            
    @property
    def eval_classes(self) -> List[int]:
        return self.datasets[self.eval_class_index].eval_classes
    
    @property
    def descriptor(self) -> DatasetDescriptor:
        """Return the dataset descriptor of the first dataset in the list of datasets.
        During construction it is ensured that all datasets have the same descriptor."""
        return self.datasets[0].descriptor
    
    def _get_main_class_index(self) -> int:
        # Since we force the all datasets to have the same descriptor, 
        # we can also require that they have the same main_class_idx
        assert all([getattr(self.datasets[0], 'main_class_idx', -1) == getattr(ds, 'main_class_idx', -1) 
                    for ds in self.datasets]), 'All datasets must have the same main_class_idx'
        return getattr(self.datasets[0],'main_class_idx', -1)
    
    def merge_label_index(self) -> defaultdict:
        """Merges the label indexes of all datasets into a single defaultdict.
        Elements with the same key are appended to the same list.
        """
        index = defaultdict(list)
        for dataset in self.datasets:
            for key, value in dataset.label_index.items():
                index[key].extend(value)
        return index
            
def get_dataset_names():
    return sorted(
        name for name in datasets.dataset_ls
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )
    

def get_dataset(dataset_name: str, 
                tasks: str | Sequence[str], 
                descriptor: Optional[DatasetDescriptor] = None,
                **dataset_kwargs) -> ConcatDataset:
    assert dataset_name in datasets.__dict__, f'Dataset {dataset_name} not supported'
    
    # load datasets from PSZS.datasets
    dataset : CustomDataset = datasets.__dict__[dataset_name]
    if isinstance(tasks, str):
        tasks = [tasks]
        
    filtered_kwargs = dataset.dataset_kwargs(**dataset_kwargs)
        
    # Always return ConcatDataset to allow uniform access to attributes
    return ConcatDataset([dataset(task=task, descriptor=descriptor, **filtered_kwargs) for task in tasks])

def build_descriptor(fileRoot: str | PathLike[str],
                     fName: str,
                     ds_split: Optional[str|int]=None,
                     level_names: Optional[Sequence[str]] = None,
                     ) -> DatasetDescriptor:
    """Constructs a dataset descriptors for the given dataset split.

    Args:
        fileRoot (str | PathLike[str]): 
            Pathlike object to the root directory of the dataset split.
        ds_split (Optional[str | int], optional): 
            Specified for dataset split. If not given no specified gets added. Defaults to None.
        level_names (Optional[Sequence[str]], optional):
            Names of the hierarchy levels. Will be stored in the Descriptors. Defaults to None.

    Returns:
        DatasetDescriptor: 
            Dataset descriptors.
    """
    root = Path(fileRoot)
    split_subpath = 'annfiles'
    if ds_split is not None:
        split_subpath += f'_Split{ds_split}'
    if fName[-3:] != 'txt':
        fName += '.txt'
        
    descriptor = DatasetDescriptor(filePath=root / split_subpath / fName, 
                                         level_names=level_names)
    return descriptor

def build_descriptors(fileRoot: str | PathLike[str],
                      ds_split: Optional[str|int]=None,
                      level_names: Optional[Sequence[str]] = None,
                      ) -> Tuple[DatasetDescriptor, DatasetDescriptor]:
    """Constructs the total and novel dataset descriptors for the given dataset split.
    No need for remapping or similar as there is no shared descriptors.

    Args:
        fileRoot (str | PathLike[str]): 
            Pathlike object to the root directory of the dataset split.
        ds_split (Optional[str | int], optional): 
            Specified for dataset split. If not given no specified gets added. Defaults to None.
        level_names (Optional[Sequence[str]], optional):
            Names of the hierarchy levels. Will be stored in the Descriptors. Defaults to None.

    Returns:
        Tuple[DatasetDescriptor, DatasetDescriptor]: 
            Tuple containing the total and novel dataset descriptors.
    """
    total_descriptor = build_descriptor(fileRoot, "descriptor_total.txt", ds_split, level_names)
    novel_descriptor = build_descriptor(fileRoot, "descriptor_novel.txt", ds_split, level_names)
    return total_descriptor, novel_descriptor    
    

def build_remapped_descriptors(fileRoot: str | PathLike[str],
                               ds_split: Optional[str|int]=None,
                               level_names: Optional[Sequence[str]] = None,
                               ) -> Tuple[DatasetDescriptor, DatasetDescriptor, DatasetDescriptor]:
    """Builds the remapped dataset descriptors for the given dataset split consisting of total, shared and novel descriptors.
    The shared descriptor is not modified while the novel descriptor is remapped by applying an offset to the models
    making their indexes start at the number of models from the shared descriptor as well reindexing the makes
    in case truly novel makes are present. The total descriptor is updated to reflect the changes in the shared and novel.
    The DatasetDescriptor fields targetId_to_predIndex and predIndex_to_targetId are updated accordingly.
    The modified descriptors are returned as a tuple (total_descriptor, shared_descriptor, novel_descriptor).
    Dataset Descriptors are loaded from the given fileRoot and the specified dataset split in the following format:
        - `fileRoot`/annfiles[_Split<`ds_split`>]/descriptor_total.txt
        - `fileRoot`/annfiles[_Split<`ds_split`>]/descriptor_shared.txt
        - `fileRoot`/annfiles[_Split<`ds_split`>]/descriptor_novel.txt
    where _Split<`ds_split`> is only added if `ds_split` is not None.

    Args:
        fileRoot (str | PathLike[str]): 
            Pathlike object to the root directory of the dataset split.
        ds_split (Optional[str | int], optional): 
            Specified for dataset split. If not given no specified gets added. Defaults to None.
        level_names (Optional[Sequence[str]], optional):
            Names of the hierarchy levels. Will be stored in the Descriptors. Defaults to None.

    Returns:
        Tuple[DatasetDescriptor, DatasetDescriptor, DatasetDescriptor]: 
            Tuple containing the total, shared and novel dataset descriptors.
    """
    total_descriptor = build_descriptor(fileRoot, "descriptor_total.txt", ds_split, level_names)
    shared_descriptor = build_descriptor(fileRoot, "descriptor_shared.txt", ds_split, level_names)
    novel_descriptor = build_descriptor(fileRoot, "descriptor_novel.txt", ds_split, level_names)
    
    # For types no offset needs to be applied
    # For makes a uniform offset can not be applied as they are partially shared 
    # between shared and novel
    # For models use num_classes of shared as offset as these are exclusive 
    # between shared and novel
    ### Only apply offset to the last level as the others are non exclusive ###
    novel_descriptor.offset = [0]*(novel_descriptor.hierarchy_levels-1) + [shared_descriptor.num_classes[-1]]
    
    # Reindex the levels such that the shared classes are first
    # i.e. truly novel classes start at the end of the shared classes
    for lvl in range(novel_descriptor.hierarchy_levels-1):
        new_idx = shared_descriptor.num_classes[lvl]
        shared_classes = shared_descriptor.targetId_to_predIndex[lvl].keys()
        novel_classes = novel_descriptor.targetId_to_predIndex[lvl].keys()
        true_novel_classes = set(novel_classes).difference(shared_classes)
        for c in true_novel_classes:
            prevIndex = novel_descriptor.targetId_to_predIndex[lvl][c]
            novel_descriptor.targetId_to_predIndex[lvl][c] = new_idx
            novel_descriptor.predIndex_to_targetId[lvl][new_idx] = c
            # Remove old index
            novel_descriptor.predIndex_to_targetId[lvl].pop(prevIndex)
            new_idx += 1
            
    # new_make_idx = shared_descriptor.num_classes[0]
    # shared_makes = shared_descriptor.targetId_to_predIndex[0].keys()
    # novel_makes = novel_descriptor.targetId_to_predIndex[0].keys()
    # true_novel_makes = set(novel_makes).difference(shared_makes)
    # for make in true_novel_makes:
    #     prevIndex = novel_descriptor.targetId_to_predIndex[0][make]
    #     novel_descriptor.targetId_to_predIndex[0][make] = new_make_idx
    #     novel_descriptor.predIndex_to_targetId[0][new_make_idx] = make
    #     # Remove old index
    #     novel_descriptor.predIndex_to_targetId[0].pop(prevIndex)
    #     new_make_idx += 1
        
    # Create mapping after applying offset and reindexing other levels
    # and update total_descriptor
    for i in range(novel_descriptor.hierarchy_levels):
        mapping = shared_descriptor.targetId_to_predIndex[i].copy()
        novel_entries = {k:v for k,v in novel_descriptor.targetId_to_predIndex[i].items() 
                         if k not in mapping}
        mapping.update(novel_entries)
        for id in total_descriptor.targetId_to_predIndex[i].keys():
            total_descriptor.targetId_to_predIndex[i][id] = mapping[id]
            total_descriptor.predIndex_to_targetId[i][mapping[id]] = id
    # Update classes lists and coarse_fine_pred_map in total descriptor to reflect changes
    total_descriptor.update_classes() # Needs to be called first as fine_coarse_name_map depends on it
    total_descriptor.update_coarse_fine_pred_map()
    return total_descriptor, shared_descriptor, novel_descriptor

def transform_target(target: torch.Tensor,
                     mode: str,
                     descriptor: Optional[DatasetDescriptor],
                     hierarchy_level: Optional[int] = None) -> torch.Tensor | np.ndarray:
    """Applied the specified transformation to the target labels based on the dataset `descriptor` and `mode`.
    This can be done either for a specified hierarchy level or for all levels if `hierarchy_level` is `None`.
    Available values for `mode` are:
        - 'target': Map the internal prediction index to the target class as in the annotation file
        - 'pred': Map the (original) target class to the internal prediction index
        - 'name': Map the target class to the class name. Needs to be the original target class, not the internal prediction index.
        - 'targetIndex': Map the class id to the index of that class in the target (independent of internal prediction)

    Args:
        target (torch.Tensor): 
            Target labels to transform.
        mode (str): 
            Transformation mode. Available values: 'target', 'pred', 'name', 'targetIndex'.
        descriptor (Optional[DatasetDescriptor], optional): 
            DatasetDescriptor. If not given nothing is done.
        hierarchy_level (Optional[int], optional): 
            Which level the transformation should be applied to. If not specified all levels are mapped. Defaults to None.

    Returns:
        torch.Tensor: Transformed target labels
    """
    if descriptor is None:
        warnings.warn("No descriptor given. Returning target as is")
        return target
    if mode == "target":
        # Map the interal prediction index to the target class
        mapping = descriptor.predIndex_to_targetId
        return_np = False
    elif mode == "pred":
        # Map the target class to the internal prediction index
        mapping = descriptor.targetId_to_predIndex
        return_np = False
    elif mode == "name":
        # Map the target class to the class name
        mapping = descriptor.id_to_name
        # As strings can not be used in tensors return as numpy array instead 
        # Better than raw list as np.arrays allow intelligent indexing etc.
        return_np = True
    elif mode == "targetIndex":
        # Map the class id to the index of that class in the target (independent of internal prediction)
        mapping = descriptor.id_to_index
        return_np = False
    else:
        raise ValueError(f"Mode {mode} not supported")
    
    if target.ndim == 2:
        # Happens when mixup/augmix is used
        assert mode == "pred", f"Only 'pred' mode supported for 2D target tensors but got {mode}"
        return target
    
    if hierarchy_level is not None:
        mapping = mapping[hierarchy_level]
        map_results = [mapping[t.item()] for t in target]
        if return_np:
            return np.array(map_results)
        return torch.tensor(map_results, device=target.device)
    else:
        map_results = [[mapping[lvl][t.item()] for lvl, t in enumerate(label)] for label in target]
        if return_np:
            return np.array(map_results)
        return torch.tensor(map_results, device=target.device)
    

def build_transform(
    input_size: int|Tuple[int, int]|Tuple[int, int, int],
    is_training: bool,
    re_prob: float = 0.,
    re_mode: str = 'const',
    re_count: int = 1,
    re_num_splits: int = 0,
    resizing: Optional[str] = 'rkrr',
    scale: Optional[Tuple[float, float]] = None,
    ratio: Optional[Tuple[float, float]] = None,
    hflip: float = 0.5,
    vflip: float = 0.,
    color_jitter: float = 0.4,
    color_jitter_prob: Optional[float] = None,
    auto_augment: Optional[str] = None,
    interpolation: str = 'bilinear',
    mean: Tuple[float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float] = IMAGENET_DEFAULT_STD,
    use_prefetcher: bool = True,
    crop_pct: float = 0.875
    ):
    """
    Builds a transformation based on the given arguments based on timm
    
    Args:
        input_size: Target input size (channels, height, width) tuple or size scalar.
        is_training: Return training (random) transforms.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        resizing: Defines the resizing and cropping mode applied during training
        scale: Random resize scale range (crop area, < 1.0 => zoom in).
        ratio: Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug)
        auto_augment: Auto augment configuration string (see timm auto_augment.py).
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        use_prefetcher: Use efficient pre-fetcher to load samples onto device.
        crop_pct: Crop percentage. Defaults to 0.875.
        
    Returns:
        Data Transformation object
        T.Compose | tuple(T.Compose, T.Compose, T.Compose)
    """
    # Set Resizing mode. If none given defaults to 'rrc' mode in timm
    # 'rrc' --> RandomResizedCrop
    # 'rkrr' --> ResizeKeepRatioRandomCrop --> Equivalent to 'ran.crop' from tllib
    # 'rkrc' --> ResizeKeepRatioCenterCrop --> Equivalent to 'cen.crop' from tllib
    if resizing:
        if resizing in ('rrc', 'rkrc', 'rkrr'):
            train_crop_mode = resizing
            replace_crop = False
        elif resizing in ('default', 'cen.crop', 'ran.crop', 'res.'):
            train_crop_mode = 'rrc'
            # Store to replace timm generated resize + crop operations
            # Only necessary for training (for validation it is the same in tllib and timm)
            replace_crop = is_training
        else:
            warnings.warn(
                f"Given resizing mode {resizing} not supported. Use default rrc mode"
            )
            train_crop_mode = 'rrc'

    transform = create_transform(
            input_size,
            is_training=is_training,
            train_crop_mode=train_crop_mode,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            color_jitter_prob=color_jitter_prob,
            auto_augment=auto_augment,
            interpolation=interpolation,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            use_prefetcher=use_prefetcher,
        )
    if replace_crop:
        # DEPRECATED #
        # Construct next larger power of 2 based on input size
        # 224 --> 256
        # pre_crop_size = 2**math.ceil(math.log2(input_size+1))
        
        # Use more intelligent way to compute resize 
        # Taken from timm
        if isinstance(input_size, (tuple, list)):
            if len(input_size) == 3:
                # First dimm represents channels if given
                input_size = input_size[1:]
            assert len(input_size) == 2
            scale_size = tuple([math.floor(x / crop_pct) for x in input_size])
        else:
            scale_size = math.floor(input_size / crop_pct)
        
        custom_resize, _ = _get_resizing_transform(mode=resizing,
                                                   pre_crop_size=scale_size,
                                                   resize_size=input_size,
                                                   scale=scale,
                                                   ratio=ratio)
        # rrc Mode creates single RandomResizedCropAndInterpolation transformation
        # --> Replace that
        transform.transforms[0] = custom_resize
    return transform