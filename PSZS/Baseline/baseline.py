import argparse
import random
import warnings
import time
import json
import math
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from timm.data import resolve_data_config

from PSZS.Utils.io.logger import Logger
from PSZS.datasets.datasets import get_dataset_names
from PSZS.Optimizer import get_optim
from PSZS.Utils.data import create_data_objects
from PSZS.Utils.io import filewriter
from PSZS.Models import *
from PSZS.Validator.Validator import Validator
from PSZS.datasets import build_remapped_descriptors, build_descriptor
import PSZS.Utils.utils as utils
from PSZS.AWS import setup_aws, handle_aws_postprocessing

def main(args):
    setup_success, ec2_client, s3_client = setup_aws(args)
    if setup_success==False:
        warnings.warn("AWS setup failed. Exiting.")
        return
    exp_name = filewriter.get_experiment_name_v2(args=args)
    logger = Logger(root=args.log, exp_name=exp_name, sep_chk=args.separate_checkpoint)
    args.uuid = logger.out_dir.split("_")[-1]
    
    device = utils.setup_device(args)
    # Optimizer gets created in model
    print(f"Using optimizer: {args.opt}")   
    # LR Scheduler gets created in model
    print(f"Using lr_scheduler: {args.sched}")
    
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    args.lr = utils.calculate_learning_rate(args)
    # Check if the dataset is hierarchical but the head is not (and the other way around)
    # switch to the correct dataset
    if args.data == 'CompCarsHierarchy' and args.head_type != 'HierarchicalHead':
        warnings.warn(f"CompCarsHierarchy dataset is used but head_type ({args.head_type}) is not hierarchical. "
                      "Automatically switching to (CompCarsModel) dataset")
        args.data = 'CompCarsModel'
    elif args.data == 'CompCarsModel' and args.head_type == 'HierarchicalHead':
        warnings.warn(f"CompCarsModel dataset is used but head_type ({args.head_type}) is hierarchical. "
                      "Automatically switching to (CompCarsHierarchy) dataset")
        args.data = 'CompCarsHierarchy'
    
    args.amp = not args.no_amp
    args.eval_base = not args.no_base_eval

    print(args)
    
    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Create backbone
    # needed as some transform configs for the dataset are based on the backbone
    # Classifier + Heads are created after constructing the dataset.
    print(f"Using model {args.model} as backbone")
    # Use args to load backbone (instead of specifying all arguments)
    backbone = load_backbone(model_name=args.model, args=args)
    
    data_config = resolve_data_config(model=backbone, verbose=True)
    if args.resize_size is None:
        # Check if we can not keep the tuple --> Only needs change in tllib part
        # --> Maybe once we can remove that part we can just keep it as the full entry
        args.resize_size = data_config['input_size'][-1]
    
    # Currently mixup/cutmix does not work with the new mapping functions
    collate_fn, mixup_fn = utils.setup_mixup_cutmix(args, num_classes=281)
    
    bs_source = math.floor(args.batch_size * args.batch_split_ratio)
    bs_target = args.batch_size - bs_source
    # Bit manupulation to check if power of 2
    if ((bs_source & (bs_source-1) == 0) and bs_source != 0)==False:
        print(f"Batch size for source domain {bs_source} is not a power of 2.")
    if ((bs_target & (bs_target-1) == 0) and bs_target != 0)==False:
        print(f"Batch size for target shared domain {bs_target} is not a power of 2.")
    print(f'Batch size source domain: ({bs_source}), target shared domain: ({bs_target})')
    bs = (bs_source, bs_target)
    
    # Construct and remap dataset descriptors
    total_descriptor, shared_descriptor, novel_descriptor = build_remapped_descriptors(fileRoot=args.root, 
                                                                                       ds_split=args.ds_split,)
    
    eval_classes = list(novel_descriptor.targetIDs[-1])
    base_classes = list(shared_descriptor.targetIDs[-1])
    
    # Create data iterators for train (Batch size is split between source and target shared domain)
    train_source_iter, train_target_iter = create_data_objects(args=args, 
                                                               batch_size=bs,
                                                               phase='train',
                                                               device=device,
                                                               collate_fn=collate_fn,
                                                               descriptor=[total_descriptor, shared_descriptor],
                                                               ensure_balance=args.ensure_domain_balance,)
    # Create data loader for validation with full batch size.
    # create_data_objects returns Tuples
    # Use dataset descriptor from train_source_iter (which is the total dataset) for validation
    # Instead of directly using total_descriptor this ensures that even if descriptor is constructed later it is correct
    val_loader : DataLoader = create_data_objects(args=args, 
                                                  batch_size=args.batch_size,
                                                  phase='val',
                                                  device=device,
                                                  collate_fn=collate_fn,
                                                  descriptor=train_source_iter.dataset_descriptor,
                                                  include_source_val_test=args.include_source_eval,
                                                  )[0]
    # Use batch sizes of the iter objects instead of bs 
    # because if --ensure-domain-balance is set the batch size might be different
    if train_source_iter.batch_size != train_target_iter.batch_size:
        args.classifier_kwargs['auto_split_indices'] = train_source_iter.batch_size
    # Generate images for from dataloader
    # utils.get_images_dataset(train_source_iter, 5, osp.join(logger.out_dir, 'data_images'))
    
    num_classes = np.array([train_source_iter.num_classes, train_target_iter.num_classes])
    num_inputs = len(args.source) + len(args.target_shared)
    if num_inputs > 2:
        warnings.warn(f"Number of inputs ({num_inputs}) is greater than 2. Clipping to 2.")
        num_inputs = 2
        
    # Set iters_per_it dynamically if not set via argument
    # See every sample at least iters-auto-scale times per epoch
    if args.iters_per_epoch is None:
        args.iters_per_epoch = args.iters_auto_scale*max(len(train_source_iter), len(train_target_iter))
    print(f"Iters per Epoch: {args.iters_per_epoch}")
        
    # Set model_type based on args.method
    # No reasonable scenario where model_type is not set based on method
    # Use args to load classifier (instead of specifying all arguments)
    model = build_model(backbone=backbone,
                        device=device,
                        num_classes=num_classes,
                        num_inputs=num_inputs,
                        args=args,
                        **args.model_kwargs,
                        **args.classifier_kwargs)
    
    # Load checkpoint if specified
    if args.checkpoint is not None:
        print(f"Loading checkpoint {args.checkpoint}")
        load_checkpoint(model, 
                        checkpoint_path=args.checkpoint,
                        checkpoint_dir=args.checkpoint_dir,
                        strict=not model.allow_non_strict_checkpoint)
    
    # Setup amp for mixed precision training
    amp_autocast, loss_scaler = utils.setup_amp(args=args, device=device)
    
    # utils.preview_lr_schedule(lr_scheduler)
    
    # Get classes for target_novel domain for evaluation
    # Dataset in the val_loader consists of (source), (target_shared) and target_novel (i.e. is a ConcatDataset)
    # The last dataset in the val_loader is the target_novel dataset 
    # Use property in wrapper around ConcatDataset (datasets.py) to get number of classes
    # In case we don't have a ConcatDataset, the CustomDataset still has the property
    # eval_classes = val_loader.dataset.eval_classes
    
    # Create optimization object for training and validation
    optim = get_optim(method=args.method,
                      train_source_iter=train_source_iter,
                      train_target_iter=train_target_iter,
                      val_loader=val_loader,
                      model=model,
                      device=device,
                      args=args,
                      logger=logger,
                      eval_classes=eval_classes,
                      grad_accum_steps=args.grad_accum_steps,
                      mixup_fn=mixup_fn,
                      loss_scaler=loss_scaler,
                      amp_autocast=amp_autocast,
                      iter_names=None,
                      eval_groups_names=['novel', 'base'], # Names can be provided anytime as they are ignored if not needed
                      additional_eval_group_classes=base_classes if args.eval_base else None,
                      **args.optim_kwargs)
    
    # Only save args here because it can be modified until now
    filewriter.save_args(args, logger.out_dir)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    optim.train()
    print(f"Training finished in {time.time() - start_time} seconds")
    print("Starting test evaluation")
    # Evaluate on test set
    load_checkpoint(model, checkpoint_path=logger.get_checkpoint_path('best'), strict=True)
    # Create data loader for test with full batch size.
    # Only create when needed to avoid keeping an unused data_loader permanent from the start
    test_loader = create_data_objects(args=args, 
                                      batch_size=args.batch_size,
                                      phase='test',
                                      device=device,
                                      collate_fn=collate_fn,
                                      descriptor=train_source_iter.dataset_descriptor,
                                      include_source_val_test=False)[0]
    runner = Validator(model=model, 
                        device=device, 
                        batch_size=args.batch_size, 
                        eval_classes=eval_classes,
                        additional_eval_group_classes=base_classes if args.eval_base else None,
                        eval_groups_names=['novel', 'base'], # Names can be provided anytime as they are ignored if not needed
                        logger=logger,
                        metrics=['acc@1', 'f1'] + args.metrics,
                        dataloader=test_loader,
                        send_to_device=args.no_prefetch,
                        print_freq=args.print_freq,
                        loss_scaler=loss_scaler,
                        amp_autocast=amp_autocast,
                        create_report=args.create_report,
                        **args.confmat_kwargs)
    runner.result_suffix = 'Target'
    test_metrics = runner.run("Target")
    csv_summary = filewriter.update_summary(epoch='Test Target', 
                                            metrics=test_metrics, 
                                            root=logger.out_dir,
                                            write_header=True)
    print("Test evaluation with source data")
    test_loader = create_data_objects(args=args, 
                                      batch_size=args.batch_size,
                                      phase='test',
                                      device=device,
                                      collate_fn=collate_fn,
                                      descriptor=train_source_iter.dataset_descriptor,
                                      include_source_val_test=True)[0]
    runner.dataloader = test_loader
    runner.result_suffix = 'TargetSource'
    test_metrics = runner.run("Target+Source")
    csv_summary = filewriter.update_summary(epoch='Test Target+Source', 
                                            metrics=test_metrics, 
                                            root=logger.out_dir,
                                            write_header=True)
    # Default value from args is None (which is non Iterable)
    add_test_desc = getattr(args, 'additional_test_desc', [])
    add_test_desc = [] if add_test_desc is None else add_test_desc
    for add_test_desc_name in add_test_desc:
        print(f'Additional Test Evaluation using {add_test_desc_name}')
        additional_test_desc = build_descriptor(fileRoot=args.root, fName=add_test_desc_name, ds_split=args.ds_split)
        # eval_classes = [i for i in eval_classes if i in additional_test_desc.targetIDs[-1]]
        eval_classes = list(additional_test_desc.targetIDs[-1])
        test_loader = create_data_objects(args=args, 
                                      batch_size=args.batch_size,
                                      phase='test',
                                      device=device,
                                      collate_fn=collate_fn,
                                      descriptor=train_source_iter.dataset_descriptor,
                                      include_source_val_test=False)[0]
        runner = Validator(model=model, 
                        device=device, 
                        batch_size=args.batch_size, 
                        eval_classes=eval_classes,
                        additional_eval_group_classes=base_classes if args.eval_base else None,
                        eval_groups_names=['novel', 'base'], # Names can be provided anytime as they are ignored if not needed
                        logger=logger,
                        metrics=['acc@1', 'f1'] + args.metrics,
                        dataloader=test_loader,
                        send_to_device=args.no_prefetch,
                        print_freq=args.print_freq,
                        loss_scaler=loss_scaler,
                        amp_autocast=amp_autocast,
                        create_report=args.create_report,
                        **args.confmat_kwargs)
        runner.result_suffix = f'Target_{add_test_desc_name}'
        test_metrics = runner.run("Target")
        csv_summary = filewriter.update_summary(epoch='Test Target', 
                                                metrics=test_metrics, 
                                                root=logger.out_dir,
                                                write_header=True)
        print("Test evaluation with source data")
        test_loader = create_data_objects(args=args, 
                                        batch_size=args.batch_size,
                                        phase='test',
                                        device=device,
                                        collate_fn=collate_fn,
                                        descriptor=train_source_iter.dataset_descriptor,
                                        include_source_val_test=True)[0]
        runner.dataloader = test_loader
        runner.result_suffix = f'TargetSource_{add_test_desc_name}'
        test_metrics = runner.run("Target+Source")
        csv_summary = filewriter.update_summary(epoch='Test Target+Source', 
                                                metrics=test_metrics, 
                                                root=logger.out_dir,
                                                write_header=True)
    
    if args.create_excel:
        filewriter.convert_csv_to_excel(csv_summary)
    acc1 = runner.eval_acc_1
    f1 = runner.eval_f1
    print(f"Best Top 1 Accuracy on Test: {acc1:3.2f}, F1 on Test: {f1:3.2f}")
    print(f'Total time: {time.time() - start_time}')
    handle_aws_postprocessing(args, s3_client=s3_client, ec2_client=ec2_client)
    logger.close()
        

if __name__ == '__main__':
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a json file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-cf', '--config', default='', type=str, metavar='FILE',
                        help='json config file specifying default arguments')
    
    parser = argparse.ArgumentParser(description='Baseline for Partially Supervised Zero Shot Domain Adaptation')
    # Outside group as it is positional
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    ### Dataset parameters ###
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('-d', '--data', metavar='DATA', choices=get_dataset_names(),
                        help='dataset: ' + ' | '.join(get_dataset_names()))
    group.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    group.add_argument('-ts', '--target-shared', help='shared target domain(s) with labeled data', nargs='+')
    group.add_argument('-tn', '--target-novel', help='novel data for target domain(s)', nargs='+')
    group.add_argument('--ds-split', type=str, default=None,
                       help='Which split of the dataset should be used. Gets appended to annfile dir. Default None.')
    group.add_argument('-cp', '--checkpoint', type=str, 
                       help='Checkpoint to evaluate. Either absolute path or relative to --checkpoint-dir.')
    group.add_argument('--checkpoint-dir', type=str, default="checkpoints",
                       help='Directory where checkpoints are stored. Default: checkpoints')
    group.add_argument('--infer-all-class-levels', action='store_true',
                       help='Infer classes of all hierarchy levels from the dataset. Default: False')
    
    # Data Loader
    group = parser.add_argument_group('Dataloader')
    group.add_argument("--no-prefetch", action='store_true',
                       help="Do not use prefetcher for dataloading.")
    
    # Data Transformations
    group = parser.add_argument_group('Data Transformations')
    group.add_argument('--train-resizing', type=str, default='default',
                        help='Resizing mode applied to train pipeline')
    group.add_argument('--val-resizing', type=str, default='default',
                        help='Resizing mode applied to val pipeline')
    group.add_argument('--resize-size', type=int,
                        help="the image size after resizing. Set based on model if not specified")
    group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    group.add_argument('--h-flip', type=float, default='0.5',
                        help='Probability to do random horizontal flipping during training (default 0.5)')
    group.add_argument('--v-flip', type=float, default='0.',
                        help='Probability to do random vertical flipping during training (default 0)')
    group.add_argument('--color-jitter-prob', type=float, default='0.',
                        help='Probability to apply random color jitter during training (default 0)')
    group.add_argument('--color-jitter', type=float, default='0.4',
                        help='Scale of random color applied if --color-jitter-prob is positive (default 0.4)')
    group.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    group.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    group.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    group.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    group.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    group.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    group.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup-mode', type=str, default='batch',
                         help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                          help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    group.add_argument('--smoothing', type=float, default=0.1,
                          help='Label smoothing (default: 0.1)')
    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (aka Stochastic Depth) (default: None)')
    group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

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
    group.add_argument('--classifier-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
    group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
    group.add_argument('--normalize-weights', action='store_true', default=None,
                       help='Normalize the weights for the hierarchical loss i.e. sum to 1.')
    group.add_argument('--hierarchy-weights', type=float, nargs='+',
                       help='Fixed weights for each level in the hierarchy if weighted loss is used.')
    group.add_argument('--initial-weights', type=float, nargs='+',
                       help='Initial weights for each level in the hierarchy if dynamic weighted loss is used. '
                       'If a single value is provided other levels are inferred to add up to 1.')
    group.add_argument('--final-weights', type=float, nargs='+',
                       help='Final weights for each level in the hierarchy if dynamic weighted loss is used. '
                       'If a single value is provided other levels are inferred to add up to 1.')
    group.add_argument('--max-mixing-epochs', type=float, nargs='+',
                       help='Number of epochs until final mixing weight is reached for each level in the hierarchy if dynamic weighted loss is used. '
                       'If a single value is provided it is used for all levels.')
    group.add_argument('--max-mixing-steps', type=float, nargs='+',
                       help='Number of iterations until final mixing weight is reached for each level in the hierarchy if dynamic weighted loss is used. '
                       'If a single value is provided it is used for all levels.')
    group.add_argument('--freeze-backbone', type=int, default=None, const=0, nargs='?', 
                        help='Freeze backbone for first N layers. '
                        'If no number (or 0) specified freeze all layers. '
                        'If not specified do not freeze any layers.')
    group.add_argument('--method', type=str, default='base',
                        help='Method for domain adaptation. Default: base '
                        'Available: ' + ','.join(METHOD_MODEL_MAP.keys()))
    group.add_argument('--optim-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
    group.add_argument('--bottleneck-dim', type=int,
                        help='Dimension of bottleneck in model. Default None')
    
    # Training parameters
    group = parser.add_argument_group('Training Parameters')
    group.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64)')
    group.add_argument("--batch-split-ratio", type=float, default=0.5,
                       help="Ratio of batch size of source and target domain. "
                       "Value represents source ratio. Remainder is for target shared. Default 0.5")
    group.add_argument('--ensure-domain-balance', action='store_true',
                       help='Ensure that during train the number of samples from each domain is balanced per batch.')
    group.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    group.add_argument('-i', '--iters-per-epoch', default=None, type=int,
                        help='Number of iterations per epoch. If None set automatically based on dataset. Default None.')
    group.add_argument("--iters-auto-scale", type=int, default=1,
                        help="Factor of times each image is seen at least during each \
                            epoch if number of iterations is set automatically.")
    group.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='The number of steps to accumulate gradients (default: 1)')
    group.add_argument('-w', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    group.add_argument("--no-scale-acum", action='store_true')
    
    # Device & distributed
    group = parser.add_argument_group('Device parameters')
    group.add_argument('--device', default='cuda', type=str,
                       help="Device (accelerator) to use.")
    group.add_argument('--no-amp', action='store_true',
                       help='Do not use Native PyTorch AMP for mixed precision training')
    group.add_argument('--amp-dtype', default='float16', type=str,
                       choices=['bfloat16', 'float16'],
                       help='lower precision AMP dtype (default: float16)')
    
    # AWS 
    group = parser.add_argument_group('AWS Parameters')
    group.add_argument('--aws', action='store_true',
                        help='Use AWS to run the model.')
    group.add_argument('--stop-instance', action='store_true',
                        help='Try to stop the instance after training if no other job is running.')
    group.add_argument('--force_stop', action='store_true',
                        help='Force the stopping of the instance regardless of other jobs.')
    group.add_argument('--aws-config', type=str, default='aws_config.json',
                        help='AWS configuration file for default S3, Dataset EBS volume, '
                        'and other aws config information.')
    group.add_argument('--s3-bucket', type=str, default=None,
                        help='S3 bucket to store results and checkpoints. '
                        'If not specified use value from aws_config file.')
    group.add_argument('--dataset-volume-id', type=str, default=None,
                        help='Volume ID of the dataset EBS volume. '
                        'If not specified use home directory of ec2 instance.')
    group.add_argument('--store-local', action='store_true',
                        help='Store results and checkpoints locally instead of on S3 bucket.')
    
    
    # Optimizer parameters
    group = parser.add_argument_group('Training Parameters')
    group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    group.add_argument('--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
    group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')
    
    # Learning rate schedule parameters
    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
    group.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate overrides lr-base if set (default: None)')
    group.add_argument('--lr-base', type=float, default=0.001, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
    group.add_argument('--lr-base-size', type=int, default=64, metavar='DIV',
                    help='base learning rate batch size (divisor, default: 64).')
    group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
    group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
    group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
    group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
    group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
    group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
    group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
    group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
    group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                    help='warmup learning rate (default: 1e-5)')
    group.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for schedulers (default: 1e-6)')
    group.add_argument('--decay-milestones', default=[30,60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
    group.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
    group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--warmup-prefix', action='store_true', default=False,
                    help='Exclude warmup period from decay schedule.'),
    group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10)')
    group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
    
    # Eval
    group = parser.add_argument_group('Evaluation Parameters')
    group.add_argument("--eval-train", action='store_true',
                        help='Compute eval metrics during training.')
    group.add_argument("--metrics", nargs='+', type=str, default=["acc@5"],
                        help="Metrics to evaluate in addition to accuracy@1 and f1."
                        "Available options: acc@5, precision, recall")
    group.add_argument("--include-source-eval", action='store_true',
                        help='Include source domain in evaluation (validation + test).')
    group.add_argument('--confmat-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
    group.add_argument('--additional-test-desc', type=str, nargs='*',
                       help='Additional descriptor files to evaluate in additional test runs.')
    group.add_argument('--no-base-eval', action='store_true',
                       help='Do not evaluate base classes during evaluation. I.e. only evaluation on novel classes.')
    
    # Logging and checkpoints
    group = parser.add_argument_group('Logging and Checkpoints')
    group.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    group.add_argument("--log", type=str, default='baseline',
                        help="Where to save logs, checkpoints and results csv.")
    group.add_argument("-sp", "--separate-checkpoint", action='store_true',
                        help='Create separate directory for checkpoints.')
    group.add_argument("--create-excel", action='store_true',
                        help='Create an excel in addition to the csv file with the results (only at end).')
    group.add_argument("--create-class-summary", action='store_true',
                        help='Save class summary tracking class wise metrics for each epoch.')
    group.add_argument("--create-report", action='store_true',
                        help='Save final report as a .html file.')
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            config = json.load(f)
            parser.set_defaults(**config)
    
    args = parser.parse_args(remaining)
    
    main(args)