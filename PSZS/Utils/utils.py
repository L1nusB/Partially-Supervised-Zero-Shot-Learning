"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
from contextlib import suppress
from functools import partial
import os.path as osp
import os
import psutil
from typing import Optional, Sequence, Tuple, overload
import argparse
import ast
import warnings

import torch

from timm.data import Mixup, FastCollateMixup
from timm.scheduler.scheduler import Scheduler
from timm.utils import NativeScaler
from timm.utils import dispatch_clip_grad

class NativeScalerMultiple(NativeScaler):
    def __call__(
            self,
            loss,
            optimizers,
            clip_grad=None,
            clip_mode='norm',
            parameters=None,
            create_graph=False,
            need_update=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if need_update:
            for optimizer in optimizers:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
                self._scaler.step(optimizer)
            self._scaler.update()

# Do not show source line for warnings
# warnings.formatwarning = lambda message, *args, **kwargs: f'{message}\n'

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            # replace hyphens with underscores for argparse namespace
            # for uniform representations in code
            key = key.replace('-', '_')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)
        
def setup_device(args: Optional[argparse.Namespace]=None, 
                 device_key: str = "device", 
                 device: Optional[str]=None) -> torch.device:
    if device is None:
        assert args is not None, "Either args or device must be provided."
        device = getattr(args, device_key, "cpu")
    
    if torch.cuda.is_available() == False:
        warnings.warn("CUDA is not available, using CPU.")
        device = torch.device('cpu')
    elif device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def send_to_device(tensor: torch.Tensor, device) -> torch.Tensor:
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

def calculate_learning_rate(args:argparse.Namespace) -> float:
    # If lr is set directly do not touch
    if args.lr:
        print(f'Learning rate from argument is taken: {args.lr}')
        return args.lr

    accum_batch_size = args.batch_size * args.grad_accum_steps
    batch_ratio = accum_batch_size / args.lr_base_size
    if not args.lr_base_scale:
        used_optimizer = args.opt.lower()
        args.lr_base_scale = 'sqrt' if any([o in used_optimizer for o in ('ada', 'lamb')]) else 'linear'
    if args.lr_base_scale == 'sqrt':
        batch_ratio = batch_ratio ** 0.5
    lr = args.lr_base * batch_ratio
    print(f'Learning rate ({lr}) calculated from base learning rate ({args.lr_base}), '
          f'base learning rate batch size ({args.lr_base_size}), batch ratio ({batch_ratio}) '
          f'and effective global batch size ({accum_batch_size}) with {args.lr_base_scale} scaling.')
    return lr

def preview_lr_schedule(scheduler: Scheduler, num_steps: int = None):
    if num_steps is None:
        num_steps = scheduler.t_initial
    for i in range(num_steps):
        print(f"{i}: {scheduler._get_lr(i)}")

@overload
def setup_mixup_cutmix(args: argparse.Namespace,
                       num_classes: int) -> Tuple[FastCollateMixup, Mixup]:...
@overload
def setup_mixup_cutmix(args: argparse.Namespace,
                       num_classes: Sequence[int]) -> Tuple[Sequence[FastCollateMixup], Sequence[Mixup]]:...
def setup_mixup_cutmix(args: argparse.Namespace,
                       num_classes: int | Sequence[int]
                       ) -> Tuple[FastCollateMixup, Mixup] | Tuple[Sequence[FastCollateMixup], Sequence[Mixup]]:
    # setup mixup / cutmix
    single_return = isinstance(num_classes, int)
    if single_return:
        num_classes = [num_classes]
    collates = []
    mixups = []
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        # Due to the way the datasets are actually zero-indexed in ascending order this should work anyways
        warnings.warn("Mixup/Cutmix not tested/validated with internal mapping functions.")
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
        )
        for nc in num_classes:
            mixup_args['num_classes'] = nc
            if not args.no_prefetch:
                collate_fn = FastCollateMixup(**mixup_args)
            else:
                mixup_fn = Mixup(**mixup_args)
            collates.append(collate_fn)
            mixups.append(mixup_fn)
    else:
        return None, None
        
    if single_return:
        return collates[0], mixups[0]
    else:
        return collates, mixups

def setup_amp(args: argparse.Namespace,
              device: torch.device) -> Tuple[suppress | torch.autocast, None | NativeScalerMultiple]:
    # Setup amp for mixed precision training
    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        print(f"Using native torch AMP for mixed precision training with dtype {args.amp_dtype}.")
        if args.amp_dtype == 'float16':
            amp_dytpe = torch.float16
        elif args.amp_dtype == 'bfloat16':
            amp_dytpe = torch.bfloat16
        else:
            warnings.warn(f"AMP dtype {args.amp_dtype} not supported. Using bfloat16 instead.")
            amp_dytpe = torch.bfloat16
        try:    
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dytpe)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == 'cuda'
            amp_autocast = torch.cuda.amp.autocast
        
        if device.type == 'cuda' and amp_dytpe == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScalerMultiple()
    else:
        print("AMP not enabled. Training in float32.")
    return amp_autocast, loss_scaler

def visualize_image(img):
    import matplotlib.pyplot as plt
    if isinstance(img, Sequence):
        img = img[0]
    for i in range(len(img)):
        plt.imshow(img.cpu()[i].permute(1,2,0))

def get_images_dataset(data_iter, 
                       num_batches: Optional[int] = None, 
                       save_dir: Optional[str] = None):
    
    import logging
    if num_batches is None:
        num_batches = len(data_iter)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for i in range(len(data_iter)):
        dat, _ = next(data_iter)
        # Terminate if num_batches is reached
        if i > num_batches:
            break
        if save_dir:
            save_filepath = osp.join(save_dir, f'batch_{i}.png')
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(100)
        display_images(dat, save_filepath)
        logger.setLevel(old_level)

def display_images(images: Sequence[torch.Tensor], 
                   save_filepath: Optional[str] = None):
    import matplotlib.pyplot as plt
    # Turn interactive plotting off
    plt.ioff()
    num_images = len(images)
    num_cols = int(num_images ** 0.5)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i, img in enumerate(images):
        row = i // num_cols
        col = i % num_cols

        if isinstance(img, torch.Tensor):
            img = img.cpu()
            
        axes[row, col].imshow(img.permute(1, 2, 0))
        axes[row, col].axis('off')

    if save_filepath:
        plt.savefig(save_filepath)
    else:
        plt.show()

def is_other_sh_scripts_running():
    current_pid = os.getpid()
    current_script = os.path.basename(__file__)
    
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Process info
            pid = process.info['pid']
            cmdline = process.info['cmdline']
            
            # Check if the process is a shell script and not the current script
            if pid != current_pid and cmdline and (cmdline[0] == 'sh' or cmdline[-1].endswith('.sh')):
                script_name = os.path.basename(cmdline[-1])
                if script_name != current_script:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return False