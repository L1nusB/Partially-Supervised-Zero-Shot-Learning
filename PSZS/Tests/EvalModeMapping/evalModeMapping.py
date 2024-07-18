import argparse
from functools import partial
from typing import Sequence

import torch

from PSZS.datasets import build_remapped_descriptors
from PSZS.Tests.utils import setup_seed, map_labels, generate_random_tensor, HIERARCHY_LEVELS, DEFAULT_NUM_CLASSES

def sample_check_ignore(*args, **kwargs) -> bool:
    return True

def sample_check_any(indicator: Sequence[bool], val: bool = True) -> bool:
    if isinstance(indicator, torch.Tensor):
        indicator = indicator.tolist()
    return any([i==val for i in indicator])

def sample_check_all(indicator: Sequence[bool], val: bool = True) -> bool:
    if isinstance(indicator, torch.Tensor):
        indicator = indicator.tolist()
    return all([i==val for i in indicator])
    

def main(args):
    setup_seed(args)
    # Construct and remap dataset descriptors
    total_descriptor, shared_descriptor, novel_descriptor = build_remapped_descriptors(fileRoot=args.root, ds_split=args.ds_split)
    if isinstance(args.hierarchy_level, str):
        args.hierarchy_level = HIERARCHY_LEVELS.get(args.hierarchy_level.lower(), None)
        if args.hierarchy_level is None:
            raise ValueError(f"Unknown hierarchy level {args.hierarchy_level}. "
                             f"Available values are: {','.join(HIERARCHY_LEVELS.keys())}.")
    novel_classes = set(novel_descriptor.targetIDs[args.hierarchy_level])
    num_objects = len(args.num_classes)
    if args.descriptor == 'total':
        descriptor = total_descriptor
        # For total descriptor have at least one novel samples
        sample_checker = partial(sample_check_any, val=True)
    elif args.descriptor == 'shared':
        descriptor = shared_descriptor
        # For shared descriptor no novel samples are allowed
        sample_checker = partial(sample_check_all, val=False)
    elif args.descriptor == 'novel':
        descriptor = novel_descriptor
        # For novel descriptor no non novel samples are allowed
        sample_checker = partial(sample_check_all, val=True)
    else:
        raise ValueError(f"Unknown descriptor mode {args.descriptor}. "
                         f"Available values are: total, shared, novel.")
        
    if args.no_sample_checks:
        sample_checker = sample_check_ignore
        
    # Empty tensor for initial while condition
    if args.multiple:
        test_obj = generate_random_tensor(size=4, max=args.num_classes, separate_tensors=True)
        hierarchy_levels = [args.hierarchy_level] * num_objects
        novel_indicator = tuple([torch.tensor([i.item() in novel_classes for i in el]) for el in test_obj])
        # Expand descriptor based on descriptor mode
        if args.descriptor_mode == 'repeat':
            descriptor = [descriptor] * num_objects
        elif args.descriptor_mode == 'trainLike':
            assert num_objects == 2, "Only two objects are supported for descriptor mode 'trainLike'."
            descriptor = [total_descriptor, shared_descriptor]
        else:
            raise ValueError(f"Unknown descriptor mode {args.descriptor_mode}.")
    else:
        hierarchy_levels = args.hierarchy_level
        count = 0
        test_obj = generate_random_tensor(size=args.test_size, max=args.num_classes[0], separate_tensors=False)
        novel_indicator = torch.tensor([i.item() in novel_classes for i in test_obj])
        while sample_checker(indicator=novel_indicator)==False and count < args.max_gen:
            test_obj = generate_random_tensor(size=args.test_size, max=args.num_classes[0], separate_tensors=False)
            novel_indicator = torch.tensor([i.item() in novel_classes for i in test_obj])
            count += 1
        if count == args.max_gen:
            raise RuntimeError("Could not generate valid test object. "
                               "Try to increase max-gen, decrease size or use another seed")
    if 'all' in args.map_modes:
        args.map_modes = ['target', 'pred', 'name', 'targetIndex']
    print(f"Test objects: {test_obj}")
    print(f"Is novel: {novel_indicator}")
    for mode in args.map_modes:
        # print(f"Testing mode {mode}")
        mapped_objects = map_labels(labels=test_obj, 
                                    mode=mode, 
                                    descriptor=descriptor, 
                                    multiple=args.multiple,
                                    hierarchy_level=hierarchy_levels)
        print(f"Mapped objects ({mode}): {mapped_objects}")
    print("-"*30)
    print("Test finished.")
    
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for different eval mode mapping modes')
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument('ds_split', type=str, default=None,
                       help='Which split of the dataset should be used. Gets appended to annfile dir. Default None.')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--max-gen', default=5000, type=int,
                        help='Maximal number of objects to be generated until condition is fulfilled. ')
    parser.add_argument('--test-size', default=4, type=int,
                        help='Size of the random test object to be generated. ')
    parser.add_argument('--hierarchy-level', default=-1, 
                        help='Hierarchy level that the generated test objects should represent. Default -1.'
                        'See HIERARCHY_LEVELS for available values. Can also be given as string. '
                        'If --multiple is set, the value is broadcasted to len(--num-classes).')
    parser.add_argument('--multiple', action='store_true',
                        help='Test objects consist of multiple tensors. Default False.')
    parser.add_argument('--no-sample-checks', action='store_true',
                        help='Do not perform any checks on what values are sampled. '
                        'This can cause errors due to missing dictionary keys. Default False.')
    parser.add_argument('--num-classes', type=int, nargs='+', default=DEFAULT_NUM_CLASSES['model'],
                        help='Maximal value for elements of test objects. '
                        'Default DEFAULT_NUM_CLASSES["model"] ([281, 225]).')
    parser.add_argument('--map-modes', type=str, nargs='+', default=['all'],
                        help='Which mapping modes should be tested. Default all.'
                        'Available values are: target, pred, name, targetIndex.',
                        choices=['all', 'target', 'pred', 'name', 'targetIndex'])
    parser.add_argument('--descriptor-mode', type=str, default='repeat',
                        help='How to handle the descriptor if --multiple is specified. Default repeat.',
                        choices=['repeat', 'trainLike'],)
    parser.add_argument('--descriptor', type=str, default='total',
                        help='Which descriptor should be used for mapping. Default total. '
                        'Available values are: total, shared, novel.',
                        choices=['total', 'shared', 'novel'])
    
    args = parser.parse_args()
    
    main(args)