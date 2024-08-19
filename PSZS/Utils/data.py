from typing import Callable, Sequence, Tuple, Optional
from argparse import Namespace
import warnings

import torch
from torch.utils.data import DataLoader

from PSZS.datasets.datasets import get_dataset, build_transform
from PSZS.datasets import DatasetDescriptor
from PSZS.Utils.dataloader import build_dataloader, ForeverDataIterator
from PSZS.Utils.sampler import RandomIdentitySampler

def create_data_objects(args: Namespace,
                        batch_size: int | Sequence[int],
                        phase: str,
                        device: torch.device = torch.device('cuda'),
                        collate_fn: Optional[Callable | Sequence[Callable]] = None,
                        img_dtype: torch.dtype = torch.float32,
                        descriptor: Optional[DatasetDescriptor|Sequence[DatasetDescriptor]] = None,
                        include_source_val_test: bool = False,
                        task_key: str = 'tasks',
                        shared_key: str = 'target_shared',
                        novel_key: str = 'target_novel',
                        split: bool = True,
                        ensure_balance: bool = False,
                        apply_aug: bool | Tuple[bool, bool] = True,
                        ) -> Tuple[ForeverDataIterator, Optional[ForeverDataIterator]] | DataLoader | Tuple[DataLoader, ...]:
    if phase == 'train':
        return prepare_data_train(args=args,
                                batch_size=batch_size,
                                device=device,
                                collate_fn=collate_fn,
                                img_dtype=img_dtype,
                                descriptor=descriptor,
                                shared=True if getattr(args, shared_key, False) else False,
                                shared_key=shared_key,
                                use_phase=split,
                                ensure_balance=ensure_balance,
                                apply_aug=apply_aug)
    elif phase == 'val' or phase == 'test' or phase == 'val_test':
        return prepare_data_val_test(args=args, 
                                    batch_size=batch_size, 
                                    device=device, 
                                    collate_fn=collate_fn, 
                                    img_dtype=img_dtype,
                                    descriptor=descriptor, 
                                    include_source=include_source_val_test,
                                    phase=phase,
                                    shared_key=shared_key,
                                    novel_key=novel_key,
                                    use_phase=split)
    elif phase == 'custom_test':
        return prepare_data_test_custom(args=args,
                                        batch_size=batch_size,
                                        device=device,
                                        collate_fn=collate_fn,
                                        img_dtype=img_dtype,
                                        descriptor=descriptor,
                                        task_key=task_key,
                                        use_phase=split)
    else:
        raise ValueError(f'Phase {phase} not supported. '
                        'Use "train", "val", "test","val_test" or "custom_test".')
    

def prepare_data_train(
        args: Namespace,
        batch_size: int | Sequence[int],
        device: torch.device = torch.device('cuda'),
        collate_fn: Optional[Callable | Sequence[Callable]] = None,
        img_dtype: torch.dtype = torch.float32,
        descriptor: Optional[DatasetDescriptor | Sequence[DatasetDescriptor]] = None,
        shared: bool = True,
        shared_key: str = 'target_shared',
        use_phase: bool = True,
        ensure_balance: bool = False,
        apply_aug: bool | Tuple[bool, bool] = True,
    ) -> Tuple[ForeverDataIterator, Optional[ForeverDataIterator]]:
    # Prepare training data
    re_num_splits = 0
    if args.resplit:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        # in timm there is a differentiation if num_aug_splits is given (no param here)
        re_num_splits = 2
    if descriptor is not None:
        if shared:
            assert isinstance(descriptor, DatasetDescriptor)==False and len(descriptor) == 2, 'Two descriptors required for source and shared target datasets.'
        else:
            # Ensure that descriptor is a list/sequence
            if isinstance(descriptor, DatasetDescriptor):
                descriptor = [descriptor]
    else:
        # Just universally set to two None values (irrespective of shared)
        # as we just index the first element anyways so the other does not matter
        descriptor = [None, None]
    if isinstance(batch_size, int):
        batch_size = [batch_size, batch_size]
    # Apply augmentation (or not) to both loaders if only one boolean is given
    if isinstance(apply_aug, bool):
        apply_aug = (apply_aug, apply_aug)
    if collate_fn is None or isinstance(collate_fn, Callable):
        collate_fn = [collate_fn, collate_fn]
        
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
        "collate_fn":collate_fn[0],
        "img_dtype":img_dtype,
        "device":device,
        "use_prefetcher":not args.no_prefetch,
        # Shuffle is mutually exclusive with batch_sampler
        # RandomSampler will produce a type of shuffling.
        "shuffle": False if getattr(args, 'identity_sampler', False) else True, 
    })
    
    transform_args = dict({
        "input_size":args.resize_size,
        "is_training":True,
        "resizing":args.train_resizing,
        "scale":args.scale,
        "ratio":args.ratio,
        "interpolation":'bilinear',
        "mean":args.norm_mean,
        "std":args.norm_std,
        "use_prefetcher":not args.no_prefetch,
        "hflip":args.h_flip, # Horizontal flipping is always applied
    })
    transform_args_aug = transform_args.copy()
    
    if apply_aug:
        transform_args_aug.update({
            "vflip":args.v_flip,
            "color_jitter":args.color_jitter,
            "color_jitter_prob":args.color_jitter_prob,
            "auto_augment":args.aa,
            "re_prob":args.reprob,
            "re_mode":args.remode,
            "re_count":args.recount,
            "re_num_splits":re_num_splits,
        })
    
    transform_train = build_transform(**transform_args)
    transform_train_aug = build_transform(**transform_args_aug)
    
    dataset_args = dict({
        "root":args.root,
        "split":args.ds_split,
        "transform":transform_train,
        "phase":'train' if use_phase else None,
        "infer_all_classes":getattr(args, "infer_all_class_levels", False),
        "label_index": getattr(args, 'identity_sampler', False)
    })
    
    # Is set to None if not given as parameter
    dataset_args["descriptor"] = descriptor[0]
    dataset_args["descriptor_file"] = "descriptor_total.txt"
    dataset_args["transform"] = transform_train_aug if apply_aug[0] else transform_train
    dataset_source = get_dataset(dataset_name=args.data,
                                 tasks=args.source,
                                 **dataset_args)
    
    if getattr(args, 'identity_sampler', False):
        # num_instances will use the internal default of 4
        dataloader_args["batch_sampler"] = RandomIdentitySampler(dataset_source, batch_size=batch_size[0])
    dataloader_args['collate_fn'] = collate_fn[0]
    loader_source = build_dataloader(dataset=dataset_source,
                                      batch_size=batch_size[0],
                                      **dataloader_args)
    iter_source = ForeverDataIterator(loader_source, device, on_device=not args.no_prefetch)
    
    if shared:
        # Is set to None if not given as parameter
        dataset_args["descriptor"] = descriptor[1]
        dataset_args["descriptor_file"] = "descriptor_shared.txt"
        dataset_args["transform"] = transform_train_aug if apply_aug[1] else transform_train
        dataset_target = get_dataset(dataset_name=args.data,
                                     tasks=getattr(args, shared_key),
                                     **dataset_args)
        if getattr(args, 'identity_sampler', False):
            # num_instances will use the internal default of 4
            dataloader_args["batch_sampler"] = RandomIdentitySampler(dataset_target, batch_size=batch_size[1])
        dataloader_args['collate_fn'] = collate_fn[1]
        loader_target = build_dataloader(dataset=dataset_target,
                                          batch_size=batch_size[1],
                                          **dataloader_args)
        iter_target = ForeverDataIterator(loader_target, device, on_device=not args.no_prefetch)
        
        # Only makes sense if there is another loader/iterator to compare to
        rebalance, bs_source, bs_target = check_balanced_dataloaders(iter_source, iter_target)
        if rebalance: 
            if ensure_balance:
                print(f"Rebalancing batch sizes to {bs_source} and {bs_target} for source and target shared domain.")
                if getattr(args, 'identity_sampler', False):
                    # This can again change the batch size (only increase) by maximally 4 (default for num_instances)
                    dataloader_args["batch_sampler"] = RandomIdentitySampler(dataset_source, batch_size=bs_source)
                loader_source = build_dataloader(dataset=dataset_source,
                                        batch_size=bs_source,
                                        **dataloader_args)
                iter_source = ForeverDataIterator(loader_source, device, on_device=not args.no_prefetch)
                
                if getattr(args, 'identity_sampler', False):
                    # This can again change the batch size (only increase) by maximally 4 (default for num_instances)
                    dataloader_args["batch_sampler"] = RandomIdentitySampler(dataset_source, batch_size=bs_target)
                loader_target = build_dataloader(dataset=dataset_target,
                                            batch_size=bs_target,
                                            **dataloader_args)
                iter_target = ForeverDataIterator(loader_target, device, on_device=not args.no_prefetch)
            else:
                print("Dataloaders for source and target shared are unbalanced. "
                      "To ensure balanced training set ensure_balance=True (--ensure-domain-balance) "
                      "or specify batch sizes manually (e.g. via --batch-split-ratio).")
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
        shared_key: str = 'target_shared',
        novel_key: str = 'target_novel',
        use_phase: bool = True,
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
    
    target_tasks = getattr(args, shared_key) + getattr(args, novel_key) if getattr(args, shared_key, False) else getattr(args, novel_key)
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
                                  phase='val' if use_phase else None, 
                                  **dataset_args)
        loader_val = build_dataloader(dataset=dataset_val, **loader_args)
        results.append(loader_val)
    
    if 'test' in phase:
        dataset_test = get_dataset(dataset_name=args.data,
                                   tasks=all_tasks,
                                   phase='test' if use_phase else None, 
                                   **dataset_args)
        loader_test = build_dataloader(dataset=dataset_test, **loader_args)
        results.append(loader_test)
        
    if len(results) == 0:
        warnings.warn(f'No datasets loaded for phase {phase}. Include "val" or "test" in phase.')
    elif len(results) == 2 and use_phase == False:
        # Could be possible that this is still desired so just inform the user but nothing more
        print(f'Loaded datasets for validation and test without phase information. '
              'This results in the same data being loaded for both validation and test.')
    
    return results

def prepare_data_test_custom(
        args: Namespace,
        batch_size: int,
        device: torch.device = torch.device('cuda'),
        collate_fn: Optional[Callable] = None,
        img_dtype: torch.dtype = torch.float32,
        descriptor: Optional[DatasetDescriptor] = None,
        task_key: str = 'tasks',
        use_phase: bool = True,
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
        "phase": 'test' if use_phase else None,
        "infer_all_classes":args.infer_all_class_levels,
    })
    
    dataset = get_dataset(dataset_name=args.data,
                          tasks=getattr(args, task_key),
                          **dataset_args)
    loader = build_dataloader(dataset=dataset, **loader_args)
    
    return loader

def check_balanced_dataloaders(loader1: DataLoader | ForeverDataIterator, 
                               loader2: DataLoader | ForeverDataIterator, 
                               threshold: float = 0.2
                               ) -> Tuple[bool, int, int]:
    len1 = len(loader1)
    len2 = len(loader2)
    data1 = len(loader1.dataset)
    data2 = len(loader2.dataset)
    bs1 = loader1.batch_size
    if bs1 is None:
        # ForeverDataIterator property batch_size already infers the batch size when using batch_sampler
        assert isinstance(loader1, DataLoader)
        bs1 = getattr(loader1.batch_sampler, 'batch_size', None)
    bs2 = loader2.batch_size
    if bs2 is None:
        # ForeverDataIterator property batch_size already infers the batch size when using batch_sampler
        assert isinstance(loader2, DataLoader)
        bs2 = getattr(loader1.batch_sampler, 'batch_size', None)
    
    total_batch_size = loader1.batch_size + loader2.batch_size
    dispary_ratio = abs(len1-len2) / (len1+len2)
    rebalance = False
    if dispary_ratio >= threshold:
        warnings.warn(f"Disparity between source and target shared domain is greater than 20% ({dispary_ratio*100:3.2f}%) "
                      f"when using batch sizes ({loader1.batch_size}, {loader2.batch_size}). "
                      "This can cause imbalanced training. ")
        rebalance = True
    balanced_batch_size1 = int(total_batch_size / ((data2/data1)+1))
    balanced_batch_size2 = total_batch_size-balanced_batch_size1
    # Ensure that all batches are at least 1
    if balanced_batch_size2 == 0:
        balanced_batch_size2 = 1
        balanced_batch_size1 = total_batch_size-1
    elif balanced_batch_size1 == 0:
        balanced_batch_size1 = 1
        balanced_batch_size2 = total_batch_size-1
    return rebalance, balanced_batch_size1, balanced_batch_size2