import argparse
import random
import warnings
import time
import json
import numpy as np
import os
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from timm.data import resolve_data_config

from PSZS.Utils.io.logger import Logger
from PSZS.datasets.datasets import build_remapped_descriptors, get_dataset_names
from PSZS.Utils.data import create_data_objects
from PSZS.Utils.io import filewriter
from PSZS.Models import *
from PSZS.Utils.utils import setup_amp, ParseKwargs
from PSZS.Validator.Validator import Validator

def main(args):
    # Setup logger and create output directory
    if args.keep_out_dir is not None:
        exp_name = "val"
        if args.keep_out_dir == "in_dir":
            args.log = args.in_dir 
        elif args.keep_out_dir == 'checkpoint' and args.checkpoint is not None:
            args.log = osp.dirname(args.checkpoint)
        elif args.keep_out_dir == 'config' and args.config_file is not None:
            args.log = osp.dirname(args.config_file)
        else:
            print(f"keep_out_dir ({args.keep_out_dir}) given but missing other parameters. "
                  f"Using --in-dir value ({args.in_dir}).")    
            args.log = args.in_dir
    else:
        exp_name = "val_" + filewriter.get_experiment_name(model='Eval', backbone=args.model, seed=args.seed)
    logger = Logger(root=args.log, exp_name=exp_name, log_filename='eval.txt', create_checkpoint=False)
    
    # Ensure and load checkpoint
    if args.checkpoint is None:
        raise ValueError("Checkpoint is not specified. Please provide a checkpoint path to evaluate.")
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        warnings.warn('CUDA is not available, using CPU.')
        args.device = 'cpu'
    device = torch.device(args.device)

    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        
    args.amp = not args.no_amp
        
    # Setup amp for mixed precision training
    amp_autocast, loss_scaler = setup_amp(args=args, device=device)

    # Create backbone
    # Classifier + Heads are created after constructing the dataset.
    print(f"Using model {args.model} as backbone")
    # Use args to load backbone (instead of specifying all arguments)
    backbone = load_backbone(model_name=args.model, args=args)
    
    # Is still needed as old checkpoints did not save the changed resize_size
    data_config = resolve_data_config(model=backbone, verbose=True)
    if args.resize_size is None:
        # Check if we can not keep the tuple --> Only needs change in tllib part
        # --> Maybe once we can remove that part we can just keep it as the full entry
        args.resize_size = data_config['input_size'][-1]
    
    # Construct and remap dataset descriptors
    total_descriptor, _, novel_descriptor = build_remapped_descriptors(fileRoot=args.root, 
                                                                       ds_split=args.ds_split,
                                                                       level_names=['make', 'model'])
    
    eval_classes = list(novel_descriptor.targetIDs[-1])    
        
    # Create test data loader object
    print(f"Batch size: {args.batch_size}")
    # If no explicit task from user, use target_novel as tasks
    if args.tasks is None:
        args.tasks = args.target_novel
    test_loader : DataLoader = create_data_objects(args=args, 
                                                    batch_size=args.batch_size,
                                                    phase='custom_test',
                                                    device=device,
                                                    descriptor=total_descriptor,) 
    
    
    
    if args.num_classes is None:
        warnings.warn("Number of classes not specified. Using number of classes in dataset. "
                      "Inferring number of classes from dataset.")
        num_classes = np.array(test_loader.dataset.num_classes)
        # num_classes = np.array(total_descriptor.num_classes)
        print(f"Number of classes: {num_classes}")
    else:
        num_classes = np.array(args.num_classes)
    num_inputs = 1
        
    # Set iters_per_epoch based on dataset
    args.iters_per_epoch = len(test_loader)
    print(f"Iters per Epoch: {args.iters_per_epoch}")
    
    # Get classes for target_novel domain for evaluation
    # Dataset in the val_loader consists of (source), (target_shared) and target_novel (i.e. is a ConcatDataset)
    # The last dataset in the val_loader is the target_novel dataset 
    # Use property in wrapper around ConcatDataset (datasets.py) to get number of classes
    # if isinstance(test_loader.dataset, ConcatDataset):
    #     eval_classes = list(test_loader.dataset.datasets[-1].class_idx_name_map.keys())
    # else:
    #     eval_classes = list(test_loader.dataset.class_idx_name_map.keys())
        
    # Create model (classifier head)
    # For validation set method to val which is currently transformed to erm
    args.method = 'val'
    model = build_model(backbone=backbone,
                        device=device,
                        num_classes=num_classes,
                        num_inputs=num_inputs,
                        args=args,
                        **getattr(args, 'method_kwargs', {}),
                        **getattr(args, 'classifier_kwargs', {}))
    
    # Load checkpoint
    print(f"Loading checkpoint {args.checkpoint}")
    # For validation strict is required to ensure all keys are present
    load_checkpoint(model, 
                    checkpoint_path=args.checkpoint,
                    checkpoint_dir=args.checkpoint_dir,)
    runner = Validator(model=model, 
                        device=device, 
                        batch_size=args.batch_size, 
                        eval_classes=eval_classes,
                        logger=logger,
                        metrics=['acc@1', 'f1'] + args.metrics,
                        dataloader=test_loader,
                        send_to_device=args.no_prefetch,
                        print_freq=args.print_freq,
                        loss_scaler=loss_scaler,
                        amp_autocast=amp_autocast,)
    
    print("Start validation")
    start_time = time.time()
    # First test_loader object is constructed on only novel set
    if 'novel' in args.eval_groups or 'all' in args.eval_groups:
        print("Validation on only novel set")
        runner.result_suffix = 'Novel'
        metrics = runner.run()
        csv_summary = filewriter.update_summary(epoch='Test Novel', 
                                                metrics=metrics, 
                                                root=logger.out_dir,
                                                write_header=True)
        runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                             dir=logger.out_dir,)
        runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                             dir=logger.out_dir,
                                             start_class=-len(eval_classes),)
    if 'shared' in args.eval_groups or 'all' in args.eval_groups:
        print("Validation on only shared set")
        args.tasks = args.target_shared
        test_loader : DataLoader = create_data_objects(args=args, 
                                                        batch_size=args.batch_size,
                                                        phase='custom_test',
                                                        device=device,
                                                        descriptor=total_descriptor,) 
        runner.dataloader = test_loader
        runner.result_suffix = 'Shared'
        metrics = runner.run()
        csv_summary = filewriter.update_summary(epoch='Test Novel+Shared', 
                                                metrics=metrics, 
                                                root=logger.out_dir,
                                                write_header=True)
        runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                             dir=logger.out_dir,)
        runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                             dir=logger.out_dir,
                                             start_class=-len(eval_classes),)
    if 'target' in args.eval_groups or 'all' in args.eval_groups:
        print("Validation on target (novel and shared) set")
        args.tasks = args.target_shared + args.target_novel
        test_loader : DataLoader = create_data_objects(args=args, 
                                                        batch_size=args.batch_size,
                                                        phase='custom_test',
                                                        device=device,
                                                        descriptor=total_descriptor,) 
        runner.dataloader = test_loader
        runner.result_suffix = 'Target'
        metrics = runner.run()
        csv_summary = filewriter.update_summary(epoch='Test Novel+Shared', 
                                                metrics=metrics, 
                                                root=logger.out_dir,
                                                write_header=True)
        runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                             dir=logger.out_dir,)
        runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                             dir=logger.out_dir,
                                             start_class=-len(eval_classes),)
    if 'source' in args.eval_groups or 'all' in args.eval_groups:
        print("Validation on source set")
        args.tasks = args.source
        test_loader : DataLoader = create_data_objects(args=args, 
                                                        batch_size=args.batch_size,
                                                        phase='custom_test',
                                                        device=device,
                                                        descriptor=total_descriptor,) 
        runner.dataloader = test_loader
        runner.result_suffix = 'Source'
        metrics = runner.run()
        csv_summary = filewriter.update_summary(epoch='Test Novel+Shared', 
                                                metrics=metrics, 
                                                root=logger.out_dir,
                                                write_header=True)
        runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                             dir=logger.out_dir,)
        runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                             dir=logger.out_dir,
                                             start_class=-len(eval_classes),)
    if 'sourceNovel' in args.eval_groups or 'all' in args.eval_groups:
        print("Validation on only novel and source set")
        args.tasks = args.source + args.target_novel
        test_loader : DataLoader = create_data_objects(args=args, 
                                                        batch_size=args.batch_size,
                                                        phase='custom_test',
                                                        device=device,
                                                        descriptor=total_descriptor,) 
        runner.dataloader = test_loader
        runner.result_suffix = 'NovelSource'
        metrics = runner.run()
        csv_summary = filewriter.update_summary(epoch='Test Novel+Source', 
                                                metrics=metrics, 
                                                root=logger.out_dir,
                                                write_header=True)
        runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                             dir=logger.out_dir,)
        runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                             dir=logger.out_dir,
                                             start_class=-len(eval_classes),)
    if 'sourceShared' in args.eval_groups or 'all' in args.eval_groups:
        print("Validation on only Shared and source set")
        args.tasks = args.source + args.target_novel
        test_loader : DataLoader = create_data_objects(args=args, 
                                                        batch_size=args.batch_size,
                                                        phase='custom_test',
                                                        device=device,
                                                        descriptor=total_descriptor,) 
        runner.dataloader = test_loader
        runner.result_suffix = 'SharedSource'
        metrics = runner.run()
        csv_summary = filewriter.update_summary(epoch='Test Novel+Source', 
                                                metrics=metrics, 
                                                root=logger.out_dir,
                                                write_header=True)
        runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                             dir=logger.out_dir,)
        runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                             dir=logger.out_dir,
                                             start_class=-len(eval_classes),)
    if 'full' in args.eval_groups or 'all' in args.eval_groups:
        print("Validation on only novel, shared, and source set")
        args.tasks = args.source + args.target_shared + args.target_novel
        test_loader : DataLoader = create_data_objects(args=args, 
                                                        batch_size=args.batch_size,
                                                        phase='custom_test',
                                                        device=device,
                                                        descriptor=total_descriptor,) 
        runner.dataloader = test_loader
        runner.result_suffix = 'All'
        metrics = runner.run()
        csv_summary = filewriter.update_summary(epoch='Test Novel+Shared+Source', 
                                                metrics=metrics, 
                                                root=logger.out_dir,
                                                write_header=True)
        runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                             dir=logger.out_dir,)
        runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                             dir=logger.out_dir,
                                             start_class=-len(eval_classes),)
    if args.create_excel:
        filewriter.convert_csv_to_excel(csv_summary)
    print(f"Validation completed in {time.time()-start_time:.2f} seconds")
    logger.close()
        

if __name__ == '__main__':
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a json file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-cf', '--config', default='', type=str, metavar='FILE',
                        help='json config file specifying default arguments')
    parser.add_argument('--in-dir', type=str, default="",
                        help='Directory where input files (checkpoint and config) are stored. Default: None')
    
    
    parser = argparse.ArgumentParser(description='Baseline for Partially Supervised Zero Shot Domain Adaptation')
    # Outside group as it is positional
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    ### Dataset parameters ###
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('-d', '--data', metavar='DATA', choices=get_dataset_names(),
                        help='dataset: ' + ' | '.join(get_dataset_names()))
    group.add_argument('-t', '--tasks', help='Name of tasks to evaluate on', nargs='+')
    group.add_argument('-cp', '--checkpoint', type=str, 
                       help='Checkpoint to evaluate. Either absolute path or relative to --in-dir.')
    group.add_argument('--checkpoint-dir', type=str, default="checkpoints",
                       help='Directory where checkpoints are stored relative to --in-dir. Default: checkpoints')
    group.add_argument('--ds-split', type=str, default=None,
                       help='Which split of the dataset should be used. Gets appended to annfile dir. Default None.')
    group.add_argument('--num-classes', type=int, nargs='+')
    # Data Loader
    group = parser.add_argument_group('Dataloader')
    group.add_argument("--no-prefetch", action='store_true',
                       help="Do not use prefetcher for dataloading.")
    
    # Data Transformations
    group = parser.add_argument_group('Data Transformations')
    group.add_argument('--resize-size', type=int,
                        help="the image size after resizing. Set based on model if not specified")

    # model parameters
    group = parser.add_argument_group('Model Parameters')
    group.add_argument('-m', '--model', metavar='MODEL',
                       default='resnet50',
                       choices=get_model_names(),
                       help='backbone model architecture: '+
                            ' | '.join(get_model_names()) +
                            'Default: resnet50')
    group.add_argument("--use-local", action='store_true',
                        help='Use local models instead of loading from timm.')
    group.add_argument("--timm-chk", type=str,
                        help='Specifier for which checkpoint/weights should be used for timm models.' 
                        'Only effective is --use-timm is given.')
    group.add_argument("--classification-type", type=str, default="DefaultClassifier",
                       help="Type of classifier used for classification using Custom Model. If not specified model backbone is used.")
    group.add_argument("--head-type", type=str, default='SimpleHead',
                       help="Type of head used for classification using Custom Model. Default SimpleHead")
    group.add_argument('--classifier-kwargs', nargs='*', default={}, action=ParseKwargs)
    
    # Training parameters
    group = parser.add_argument_group('Training Parameters')
    group.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64)')
    group.add_argument("--iters-auto-scale", type=int, default=1,
                        help="Factor of times each image is seen at least during each \
                            epoch if number of iterations is set automatically.")
    group.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    group.add_argument('-w', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    
    # Device & distributed
    group = parser.add_argument_group('Device parameters')
    group.add_argument('--device', default='cuda', type=str,
                       help="Device (accelerator) to use.")
    group.add_argument('--no-amp', action='store_true',
                       help='Do not use Native PyTorch AMP for mixed precision training')
    group.add_argument('--amp-dtype', default='bfloat16', type=str,
                       choices=['bfloat16', 'float16'],
                       help='lower precision AMP dtype (default: bfloat16)')
    
    # Eval
    group = parser.add_argument_group('Evaluation Parameters')
    group.add_argument("--metrics", nargs='+', type=str,
                        help="Metrics to evaluate in addition to accuracy@1 and f1."
                        "Available options: acc@5, precision, recall, f1@5")
    group.add_argument("--eval-groups", nargs='+', type=str, default=['all'],
                       help="Groups to evaluate on. Available options: "
                       "novel, shared, target, source, sourceNovel, sourceShared, full, all",
                       choices=['novel', 'shared', 'target', 'source', 'sourceNovel', 'sourceShared', 
                                'full', 'all'])
    
    # Logging and checkpoints
    group = parser.add_argument_group('Logging and Checkpoints')
    group.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    group.add_argument("--log", type=str, default='eval',
                        help="Where to save logs, checkpoints and results csv.")
    group.add_argument('-kod', '--keep-out-dir', type=str, default=None, 
                       const='in_dir', nargs='?', choices=['checkpoint', 'config', 'in_dir'],
                       help='Keep the output directory as the same as either in_dir, checkpoint or config file.')
    group.add_argument("--create-excel", action='store_true',
                        help='Create an excel in addition to the csv file with the results (only at end).')
    
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        if args_config.config[-5:] != '.json':
            args_config.config = args_config.config + '.json'
        if os.path.exists(args_config.config)==False:
            args_config.config = os.path.join(args_config.in_dir, args_config.config)
        if (os.path.exists(args_config.config) and os.path.isfile(args_config.config))==False:
            warnings.warn(f"Config file {args_config.config} does not exist.")
            exit()
        assert os.path.exists(args_config.config) and os.path.isfile(args_config.config), f"Config file {args_config.config} does not exist."
        with open(args_config.config, 'r') as f:
            config = json.load(f)
            parser.set_defaults(**config)
    
    args = parser.parse_args(remaining)
    # Copy over config file and in_dir to args
    args.config_file = args_config.config
    args.in_dir = args_config.in_dir
    if '--checkpoint-dir' in remaining:
        # Option to use only in_dir as checkpoint dir without any subdirectory
        if args.checkpoint_dir.lower() == 'none':
            args.checkpoint_dir = args.in_dir
        else:
            args.checkpoint_dir = os.path.join(args.in_dir, args.checkpoint_dir)
    else:
        # If checkpoint dir is not given, set it to in_dir/<checkpoint_dir> which is 
        # either the default (checkpoints) or the one given in the config file
        args.checkpoint_dir = os.path.join(args.in_dir, args.checkpoint_dir)
    # If config file is given default values are already set
    # otherwise set default values for the config
    if args_config.config is None:
        args.val_resizing = 'default'
        args.scale = [0.08, 1.0]
        args.ratio = [3. / 4., 4. / 3.]
        args.norm_mean = (0.485, 0.456, 0.406)
        args.norm_std = (0.229, 0.224, 0.225)
        args.target_shared = None
        assert args.tasks is not None, "No tasks or config file specified."
        args.target_novel = args.tasks
        args.eval_train = False
        args.mixup_off_epoch = 0
        args.no_scale_acum = False
    # Overwrite the target_novel from config with tasks if tasks are specified
    if args.tasks is not None:
        args.target_novel = args.tasks
        # args.target_shared = None
    
    main(args)