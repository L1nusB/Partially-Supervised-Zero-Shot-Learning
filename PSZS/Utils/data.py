from typing import Callable, Sequence, Tuple, Optional
from argparse import Namespace
import warnings

import torch
from torch.utils.data import DataLoader

from PSZS.datasets.datasets import get_dataset, build_transform
from PSZS.datasets import DatasetDescriptor
from PSZS.Utils.dataloader import build_dataloader, ForeverDataIterator

def create_data_objects(args: Namespace,
                        batch_size: int | Sequence[int],
                        phase: str,
                        device: torch.device = torch.device('cuda'),
                        collate_fn: Optional[Callable] = None,
                        img_dtype: torch.dtype = torch.float32,
                        descriptor: Optional[DatasetDescriptor|Sequence[DatasetDescriptor]] = None,
                        include_source_val_test: bool = False,
                        task_key: str = 'tasks'
                        ) -> Tuple[ForeverDataIterator, Optional[ForeverDataIterator]] | DataLoader | Tuple[DataLoader, ...]:
    if phase == 'train':
        return prepare_data_train(args=args,
                                  batch_size=batch_size,
                                  device=device,
                                  collate_fn=collate_fn,
                                  img_dtype=img_dtype,
                                  descriptor=descriptor)
    elif phase == 'val' or phase == 'test' or phase == 'val_test':
        return prepare_data_val_test(args=args, 
                                     batch_size=batch_size, 
                                     device=device, 
                                     collate_fn=collate_fn, 
                                     img_dtype=img_dtype,
                                     descriptor=descriptor, 
                                     include_source=include_source_val_test,
                                     phase=phase,)
    elif phase == 'custom_test':
        return prepare_data_test_custom(args=args,
                                        batch_size=batch_size,
                                        device=device,
                                        collate_fn=collate_fn,
                                        img_dtype=img_dtype,
                                        descriptor=descriptor,
                                        task_key=task_key)
    else:
        raise ValueError(f'Phase {phase} not supported. '
                         'Use "train", "val", "test","val_test" or "custom_test".')
    

def prepare_data_train(
        args: Namespace,
        batch_size: Sequence[int],
        device: torch.device = torch.device('cuda'),
        collate_fn: Optional[Callable] = None,
        img_dtype: torch.dtype = torch.float32,
        descriptor: Optional[Sequence[DatasetDescriptor]] = None,
    ) -> Tuple[ForeverDataIterator, Optional[ForeverDataIterator]]:
    # Prepare training data
    re_num_splits = 0
    if args.resplit:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        # in timm there is a differentiation if num_aug_splits is given (no param here)
        re_num_splits = 2
    if descriptor is not None:
        assert len(descriptor) == 2, 'Two descriptors required for source and shared target datasets.'
    else:
        descriptor = [None, None]
        
    dataloader_args = dict({
        "input_size":args.resize_size,
        "is_training":True,
        "re_prob":args.reprob,
        "re_mode":args.remode,
        "re_count":args.recount,
        "re_num_splits":re_num_splits,
        "mean":args.norm_mean,
        "std":args.norm_std,
        "num_workers":args.workers,
        "collate_fn":collate_fn,
        "img_dtype":img_dtype,
        "device":device,
        "use_prefetcher":not args.no_prefetch,
    })
    
    transform_args = dict({
        "input_size":args.resize_size,
        "is_training":True,
        "resizing":args.train_resizing,
        "scale":args.scale,
        "ratio":args.ratio,
        "hflip":args.h_flip,
        "vflip":args.v_flip,
        "color_jitter":args.color_jitter,
        "color_jitter_prob":args.color_jitter_prob,
        "auto_augment":args.aa,
        "interpolation":'bilinear',
        "mean":args.norm_mean,
        "std":args.norm_std,
        "re_prob":args.reprob,
        "re_mode":args.remode,
        "re_count":args.recount,
        "re_num_splits":re_num_splits,
        "use_prefetcher":not args.no_prefetch,
    })
    
    
    transform_train = build_transform(**transform_args)
    
    dataset_args = dict({
        "root":args.root,
        "split":args.ds_split,
        "transform":transform_train,
        "phase":'train',
        "infer_all_classes":args.infer_all_class_levels,
    })
    
    # Is set to None if not given as parameter
    dataset_args["descriptor"] = descriptor[0]
    dataset_args["descriptor_file"] = "descriptor_total.txt"
    dataset_source = get_dataset(dataset_name=args.data,
                                 tasks=args.source,
                                 **dataset_args)
    
    loader_source = build_dataloader(dataset=dataset_source,
                                      batch_size=batch_size[0],
                                      **dataloader_args)
    iter_source = ForeverDataIterator(loader_source, device, on_device=not args.no_prefetch)
    
    if args.target_shared:
        # Is set to None if not given as parameter
        dataset_args["descriptor"] = descriptor[1]
        dataset_args["descriptor_file"] = "descriptor_shared.txt"
        dataset_target = get_dataset(dataset_name=args.data,
                                     tasks=args.target_shared,
                                     **dataset_args)
        
        loader_target = build_dataloader(dataset=dataset_target,
                                          batch_size=batch_size[1],
                                          **dataloader_args)
        iter_target = ForeverDataIterator(loader_target, device, on_device=not args.no_prefetch)
    else:
        iter_target = None
    
    return iter_source, iter_target

def prepare_data_val_test(
        args: Namespace,
        batch_size: int,
        device: torch.device = torch.device('cuda'),
        collate_fn: Optional[Callable] = None,
        img_dtype: torch.dtype = torch.float32,
        descriptor: Optional[DatasetDescriptor] = None,
        include_source: bool = False,
        phase: str = 'val_test',
    ) -> Tuple[DataLoader, ...]:
    # Prepare validation data
    transform_val_test = build_transform(input_size=args.resize_size,
                                         is_training=False,
                                         resizing=args.val_resizing,
                                         scale=args.scale,
                                         ratio=args.ratio,
                                         interpolation='bilinear',
                                         mean=args.norm_mean,
                                         std=args.norm_std,
                                         use_prefetcher=not args.no_prefetch)
    
    target_tasks = args.target_shared + args.target_novel if args.target_shared else args.target_novel
    all_tasks = args.source + target_tasks if include_source else target_tasks
    
    loader_args = dict({
        "input_size":args.resize_size,
        "is_training":False,
        "mean":args.norm_mean,
        "std":args.norm_std,
        "num_workers":args.workers,
        "collate_fn":collate_fn,
        "img_dtype":img_dtype,
        "device":device,
        "use_prefetcher":not args.no_prefetch,
        "batch_size":batch_size,
    })
    
    dataset_args = dict({
        "root":args.root,
        "split":args.ds_split,
        "descriptor":descriptor,
        "descriptor_file":"descriptor_total.txt",
        "transform":transform_val_test,
        "infer_all_classes":args.infer_all_class_levels,
    })
    
    results = []
    
    if 'val' in phase:
        dataset_val = get_dataset(dataset_name=args.data,
                                  tasks=all_tasks,
                                  phase='val', 
                                  **dataset_args)
        loader_val = build_dataloader(dataset=dataset_val, **loader_args)
        results.append(loader_val)
    
    if 'test' in phase:
        dataset_test = get_dataset(dataset_name=args.data,
                                   tasks=all_tasks,
                                   phase='test', 
                                   **dataset_args)
        loader_test = build_dataloader(dataset=dataset_test, **loader_args)
        results.append(loader_test)
        
    if len(results) == 0:
        warnings.warn(f'No datasets loaded for phase {phase}. Include "val" or "test" in phase.')
    
    return results

def prepare_data_test_custom(
        args: Namespace,
        batch_size: int,
        device: torch.device = torch.device('cuda'),
        collate_fn: Optional[Callable] = None,
        img_dtype: torch.dtype = torch.float32,
        descriptor: Optional[DatasetDescriptor] = None,
        task_key: str = 'tasks'
    ) -> DataLoader:
    # Prepare validation data
    transform_val_test = build_transform(input_size=args.resize_size,
                                         is_training=False,
                                         resizing=args.val_resizing,
                                         scale=args.scale,
                                         ratio=args.ratio,
                                         interpolation='bilinear',
                                         mean=args.norm_mean,
                                         std=args.norm_std,
                                         use_prefetcher=not args.no_prefetch)
    loader_args = dict({
        "input_size":args.resize_size,
        "is_training":False,
        "mean":args.norm_mean,
        "std":args.norm_std,
        "num_workers":args.workers,
        "collate_fn":collate_fn,
        "img_dtype":img_dtype,
        "device":device,
        "use_prefetcher":not args.no_prefetch,
        "batch_size":batch_size,
    })
    
    dataset_args = dict({
        "root":args.root,
        "split":args.ds_split,
        "descriptor":descriptor,
        "descriptor_file":"descriptor_total.txt",
        "transform":transform_val_test,
        "phase": 'test',
        "infer_all_classes":args.infer_all_class_levels,
    })
    
    dataset = get_dataset(dataset_name=args.data,
                          tasks=getattr(args, task_key),
                          **dataset_args)
    loader = build_dataloader(dataset=dataset, **loader_args)
    
    return loader