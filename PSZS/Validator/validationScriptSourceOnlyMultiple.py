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
from PSZS.datasets.datasets import build_descriptors, get_dataset_names, build_descriptor
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
    args.eval_base = not args.no_base_eval
    args.split_data = not args.no_split_data
        
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
    
    # Construct and remap dataset descriptors (No Shared dataset)
    total_descriptor, novel_descriptor = build_descriptors(fileRoot=args.root, 
                                                           ds_split=args.ds_split,)
    eval_classes = list(novel_descriptor.targetIDs[-1])
    base_classes = [i for i in list(total_descriptor.targetIDs[-1]) if i not in eval_classes]
    
    if len(args.test_desc) > 0:
       eval_classes = [(desc_name, list(build_descriptor(fileRoot=args.root, fName=desc_name, ds_split=args.ds_split).targetIDs[-1])) 
                       for desc_name in args.test_desc]
    else:
        eval_classes = [('novel_vanilla',eval_classes)]
        
    # Create test data loader object
    print(f"Batch size: {args.batch_size}")
    # If no explicit task from user, use target as tasks
    if args.tasks is None:
        args.tasks = args.target
        assert args.tasks is not None, "No tasks specified for evaluation."
        
    test_loader : DataLoader = create_data_objects(args=args, 
                                                    batch_size=args.batch_size,
                                                    phase='custom_test',
                                                    device=device,
                                                    descriptor=total_descriptor,
                                                    split=args.split_data) 
    
    
    
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
    
    for desc_name, e_classes in eval_classes:
        print(f'Starting evaluation for descriptor: {desc_name}')
        # Load checkpoint
        # For validation strict is required to ensure all keys are present
        print(f"Loading checkpoint {args.checkpoint}")
        load_checkpoint(model, 
                        checkpoint_path=args.checkpoint,
                        checkpoint_dir=args.checkpoint_dir,)
        
        runner = Validator(model=model, 
                            device=device, 
                            batch_size=args.batch_size, 
                            eval_classes=e_classes,
                            additional_eval_group_classes=base_classes if args.eval_base else None,
                            eval_groups_names=['novel', 'base'], # Names can be provided anytime as they are ignored if not needed
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
            args.tasks = getattr(args, 'tasks_novel', None)
            if args.tasks is None:
                warnings.warn("No novel tasks specified. Using target tasks for evaluation in 'novel'.")
                args.tasks = args.target
            test_loader : DataLoader = create_data_objects(args=args, 
                                                            batch_size=args.batch_size,
                                                            phase='custom_test',
                                                            device=device,
                                                            descriptor=total_descriptor,
                                                            split=args.split_data) 
            runner.result_suffix = 'Novel'
            # Updates the num_iters and the progressbar
            runner.dataloader = test_loader
            metrics = runner.run()
            csv_summary = filewriter.update_summary(epoch='Test Novel', 
                                                    metrics=metrics, 
                                                    root=logger.out_dir,
                                                    write_header=True)
            runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                                dir=logger.out_dir,)
            runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                                dir=logger.out_dir,
                                                start_class=-len(e_classes),)
        if 'sourceNovel' in args.eval_groups or 'all' in args.eval_groups:
            print("Validation on only novel and source set")
            args.tasks = getattr(args, 'tasks_novel', None)
            if args.tasks is None:
                warnings.warn("No novel tasks specified. Using target tasks for evaluation in 'sourceNovel'.")
                args.tasks = args.target
            # Prepend source tasks to the tasks
            args.tasks = args.source + args.tasks
            test_loader : DataLoader = create_data_objects(args=args, 
                                                            batch_size=args.batch_size,
                                                            phase='custom_test',
                                                            device=device,
                                                            descriptor=total_descriptor,
                                                            split=args.split_data) 
            runner.dataloader = test_loader
            runner.result_suffix = 'NovelSource'
            # Updates the num_iters and the progressbar
            runner.dataloader = test_loader
            metrics = runner.run()
            csv_summary = filewriter.update_summary(epoch='Test Novel+Source', 
                                                    metrics=metrics, 
                                                    root=logger.out_dir,
                                                    write_header=True)
            runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                                dir=logger.out_dir,)
            runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                                dir=logger.out_dir,
                                                start_class=-len(e_classes),)
        if 'target' in args.eval_groups or 'all' in args.eval_groups:
            print("Validation on target set")
            args.tasks = getattr(args, 'tasks_full', None)
            if args.tasks is None:
                warnings.warn("No full tasks specified. Using target tasks for evaluation in 'target'.")
                args.tasks = args.target
            test_loader : DataLoader = create_data_objects(args=args, 
                                                            batch_size=args.batch_size,
                                                            phase='custom_test',
                                                            device=device,
                                                            descriptor=total_descriptor,
                                                            split=args.split_data) 
            runner.dataloader = test_loader
            runner.result_suffix = 'Target'
            # Updates the num_iters and the progressbar
            runner.dataloader = test_loader
            metrics = runner.run()
            csv_summary = filewriter.update_summary(epoch='Test Target', 
                                                    metrics=metrics, 
                                                    root=logger.out_dir,
                                                    write_header=True)
            runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                                dir=logger.out_dir,)
            runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                                dir=logger.out_dir,
                                                start_class=-len(e_classes),)
        if 'source' in args.eval_groups or 'all' in args.eval_groups:
            print("Validation on source set")
            args.tasks = args.source
            test_loader : DataLoader = create_data_objects(args=args, 
                                                            batch_size=args.batch_size,
                                                            phase='custom_test',
                                                            device=device,
                                                            descriptor=total_descriptor,
                                                            split=args.split_data) 
            runner.dataloader = test_loader
            runner.result_suffix = 'Source'
            # Updates the num_iters and the progressbar
            runner.dataloader = test_loader
            metrics = runner.run()
            csv_summary = filewriter.update_summary(epoch='Test Source', 
                                                    metrics=metrics, 
                                                    root=logger.out_dir,
                                                    write_header=True)
            runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                                dir=logger.out_dir,)
            runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                                dir=logger.out_dir,
                                                start_class=-len(e_classes),)
        if 'full' in args.eval_groups or 'all' in args.eval_groups:
            print("Validation source + target set")
            args.tasks = getattr(args, 'tasks_full', None)
            if args.tasks is None:
                warnings.warn("No full tasks specified. Using target tasks for evaluation in 'full'.")
                args.tasks = args.target
            # Prepend source tasks to the tasks
            args.tasks = args.source + args.tasks
            test_loader : DataLoader = create_data_objects(args=args, 
                                                            batch_size=args.batch_size,
                                                            phase='custom_test',
                                                            device=device,
                                                            descriptor=total_descriptor,
                                                            split=args.split_data) 
            runner.dataloader = test_loader
            runner.result_suffix = 'All'
            # Updates the num_iters and the progressbar
            runner.dataloader = test_loader
            metrics = runner.run()
            csv_summary = filewriter.update_summary(epoch='Test Full', 
                                                    metrics=metrics, 
                                                    root=logger.out_dir,
                                                    write_header=True)
            runner.confusion_matrix.update_class_summary(file='classMetricsFull',
                                                dir=logger.out_dir,)
            runner.confusion_matrix.update_class_summary(file='classMetricsEval',
                                                dir=logger.out_dir,
                                                start_class=-len(e_classes),)
    if args.create_excel:
        filewriter.convert_csv_to_excel(csv_summary)
    print(f"Validation completed in {time.time()-start_time:.2f} seconds")
    # Do not close Logger as this will cause the logger to close the sys.stdout and sys.stderr
    # leading to I/O on closed terminal
    logger.close(keepalive=True)
        

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
    group.add_argument('-tNovel', '--tasks-novel', help='Name of novel tasks to evaluate on. Defaults to --tasks if not specified', nargs='+')
    group.add_argument('-tFull', '--tasks-full', help='Name of full tasks to evaluate on. Defaults to --tasks if not specified', nargs='+')
    group.add_argument('-cp', '--checkpoint', type=str, 
                       help='Checkpoint to evaluate. Either absolute path or relative to --in-dir.')
    group.add_argument('--checkpoint-dir', type=str, default="checkpoints",
                       help='Directory where checkpoints are stored relative to --in-dir. Default: checkpoints')
    group.add_argument('--ds-split', type=str, default=None,
                       help='Which split of the dataset should be used. Gets appended to annfile dir. Default None.')
    group.add_argument('--num-classes', type=int, nargs='+')
    group.add_argument('--no-split-data', action='store_true',
                       help='Split data (source and target domain) into train/val/test. Default: False')
    
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
    group.add_argument('--test-desc', type=str, nargs='*',
                       help='Descriptor files to evaluate on. If not specified uses base novel descriptor.')
    group.add_argument('--no-base-eval', action='store_true',
                       help='Do not evaluate base classes during evaluation. I.e. only evaluation on novel classes.')
    
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
    base = args_config.in_dir
    config_name = args_config.config
    for dir in os.listdir(base):
        print(f'Checking {dir}')
        if os.path.isdir(os.path.join(base, dir)):
            print(f'Running {dir}')
            args_config.in_dir = os.path.join(base, dir)
        else:
            print(f'Skipping {dir}')
            continue
    
        if config_name:
            if config_name[-5:] != '.json':
                config_name = config_name + '.json'
            if os.path.exists(config_name)==False:
                args_config.config = os.path.join(args_config.in_dir, config_name)
            if (os.path.exists(args_config.config) and os.path.isfile(args_config.config))==False:
                print(f"Config file {args_config.config} does not exist. Skipping")
                continue
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
            assert args.tasks is not None, "No tasks or config file specified."
            args.target = args.tasks
            args.eval_train = False
            args.mixup_off_epoch = 0
            args.no_scale_acum = False
        # Overwrite the target from config with tasks if tasks are specified
        if args.tasks is not None:
            setattr(args, 'target', args.tasks)
        
        main(args)