# Needed to reference CustomModel in type hints before __init__ is initialized
# see https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
from __future__ import annotations 
from contextlib import suppress
from typing import TYPE_CHECKING, Callable, Optional, Union, Sequence

from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Utils.io.logger import Logger

import torch
from torch.utils.data import DataLoader

from timm.data.loader import PrefetchLoader

from argparse import Namespace
from PSZS.Optimizer import *

if TYPE_CHECKING:
    from PSZS.Models import CustomModel
    
def _get_optim_single(method:str,
                      train_iter: ForeverDataIterator,
                      val_loader: Union[DataLoader, PrefetchLoader],
                      model: CustomModel,
                      device: torch.device,
                      args: Namespace,
                      logger: Logger,
                      eval_classes: Sequence[int | Sequence[int]],
                      grad_accum_steps: int = 1,
                      mixup_fn: Optional[Callable] = None,
                      loss_scaler: Optional[torch.cuda.amp.GradScaler] = None,
                      amp_autocast = suppress,
                      eval_groups_names: Optional[Sequence[str]] = None, 
                      additional_eval_group_classes: Optional[Sequence[int] | Sequence[Sequence[int]]] = None,
                      **optim_kwargs) -> Base_Single:
    method = method.lower()
    assert method == 'erm', f"Only Method erm supported for single source domain but got {method}"
    optimClass = ERM_Single
    print(f"Creating optimizer for method: {method} (Multiple Domain)")
    optim_params = optimClass.get_optim_kwargs(**optim_kwargs)
    optim = optimClass(train_iter=train_iter, 
                       val_loader=val_loader, 
                       model=model, 
                       device=device,
                       iters_per_epoch=args.iters_per_epoch, 
                       print_freq=args.print_freq, 
                       batch_size=args.batch_size,
                       eval_classes=eval_classes,
                       send_to_device=args.no_prefetch,
                       eval_during_train=args.eval_train,
                       eval_metrics=args.metrics,
                       grad_accum_steps=grad_accum_steps,
                       mixup_off_epoch=args.mixup_off_epoch,
                       mixup_fn=mixup_fn,
                       loss_scaler=loss_scaler,
                       amp_autocast=amp_autocast,
                       scale_loss_accum=not args.no_scale_acum,
                       num_epochs=args.epochs,
                       logger=logger,
                       create_class_summary=args.create_class_summary,
                       eval_groups_names=eval_groups_names,
                       additional_eval_group_classes=additional_eval_group_classes,
                       **optim_params)
    return optim
    
def _get_optim_multiple(method:str,
                        train_source_iter: ForeverDataIterator,
                        train_target_iter: ForeverDataIterator,
                        val_loader: Union[DataLoader, PrefetchLoader],
                        model: CustomModel,
                        device: torch.device,
                        args: Namespace,
                        logger: Logger,
                        eval_classes: Sequence[int | Sequence[int]],
                        grad_accum_steps: int = 1,
                        mixup_fn: Optional[Callable] = None,
                        loss_scaler: Optional[torch.cuda.amp.GradScaler] = None,
                        amp_autocast = suppress,
                        iter_names: Optional[Sequence[str]]=None,
                        eval_groups_names: Optional[Sequence[str]] = None, 
                        additional_eval_group_classes: Optional[Sequence[int] | Sequence[Sequence[int]]] = None,
                        **optim_kwargs) -> Base_Multiple:
    # No need to check if method is supported since this is already checked when 
    # constructing `model` in build_model() in PSZS.Models.models.py
    method = method.lower()
    match method:
        case "erm":
            optimClass = ERM_Multiple
        case "adda":
            optimClass = ADDA_Multiple
        case "base":
            optimClass = Base_Multiple
        case "dann":
            optimClass = DANN_Multiple
        case "jan":
            optimClass = JAN_Multiple
        case "mcc":
            optimClass = MCC_Multiple
        case "mcd":
            optimClass = MCD_Multiple
        case "mdd":
            optimClass = MDD_Multiple
        case "ujda":
            optimClass = UJDA_Multiple
        case "pan":
            optimClass = PAN_Multiple
        case _:
            raise NotImplementedError(f'Method {method} not supported')
    print(f"Creating optimizer for method: {method} (Multiple Domain)")
    
    max_mixing_epochs_args = getattr(args, 'max_mixing_epochs', None)
    
    if max_mixing_epochs_args is not None:
        if optim_kwargs.get('max_mixing_epochs', None) is not None:
            print(f'Overwriting max_mixing_epochs from kwargs {optim_kwargs["max_mixing_epochs"]} '
                    f'with value from --max-mixing-epochs {max_mixing_epochs_args}.')
        optim_kwargs['max_mixing_epochs'] = max_mixing_epochs_args
    optim_params = optimClass.get_optim_kwargs(**optim_kwargs)
    optim = optimClass(train_iters=[train_source_iter, train_target_iter], 
                            val_loader=val_loader, 
                            model=model, 
                            device=device,
                            iters_per_epoch=args.iters_per_epoch, 
                            print_freq=args.print_freq, 
                            batch_size=args.batch_size,
                            eval_classes=eval_classes,
                            send_to_device=args.no_prefetch,
                            eval_during_train=args.eval_train,
                            eval_metrics=args.metrics,
                            grad_accum_steps=grad_accum_steps,
                            mixup_off_epoch=args.mixup_off_epoch,
                            mixup_fn=mixup_fn,
                            loss_scaler=loss_scaler,
                            amp_autocast=amp_autocast,
                            scale_loss_accum=not args.no_scale_acum,
                            iter_names=iter_names,
                            num_epochs=args.epochs,
                            logger=logger,
                            create_class_summary=args.create_class_summary,
                            eval_groups_names=eval_groups_names,
                            additional_eval_group_classes=additional_eval_group_classes,
                            **optim_params)
    return optim

def get_optim(method:str,
              train_source_iter: ForeverDataIterator,
              val_loader: Union[DataLoader, PrefetchLoader],
              model: CustomModel,
              device: torch.device,
              args: Namespace,
              logger: Logger,
              eval_classes: Sequence[int | Sequence[int]],
              train_target_iter: Optional[ForeverDataIterator]=None,
              grad_accum_steps: int = 1,
              mixup_fn: Optional[Callable] = None,
              loss_scaler: Optional[torch.cuda.amp.GradScaler] = None,
              amp_autocast = suppress,
              iter_names: Optional[Sequence[str]]=None,
              eval_groups_names: Optional[Sequence[str]] = None, 
              additional_eval_group_classes: Optional[Sequence[int] | Sequence[Sequence[int]]] = None,
              **optim_kwargs) -> Base_Single | Base_Multiple:
    if train_target_iter is None:
        return _get_optim_single(method=method,
                                 train_iter=train_source_iter,
                                 val_loader=val_loader,
                                 model=model,
                                 device=device,
                                 args=args,
                                 logger=logger,
                                 eval_classes=eval_classes,
                                 grad_accum_steps=grad_accum_steps,
                                 mixup_fn=mixup_fn,
                                 loss_scaler=loss_scaler,
                                 amp_autocast=amp_autocast,
                                 eval_groups_names=eval_groups_names,
                                 additional_eval_group_classes=additional_eval_group_classes,
                                 **optim_kwargs)
    else:
        return _get_optim_multiple(method=method,
                                   train_source_iter=train_source_iter,
                                   train_target_iter=train_target_iter,
                                   val_loader=val_loader,
                                   model=model,
                                   device=device,
                                   args=args,
                                   logger=logger,
                                   eval_classes=eval_classes,
                                   grad_accum_steps=grad_accum_steps,
                                   mixup_fn=mixup_fn,
                                   loss_scaler=loss_scaler,
                                   amp_autocast=amp_autocast,
                                   iter_names=iter_names,
                                   eval_groups_names=eval_groups_names,
                                   additional_eval_group_classes=additional_eval_group_classes,
                                   **optim_kwargs)