# Needed to reference Base_Optimizer and CustomModel in type hints
# see https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
from __future__ import annotations 
from contextlib import suppress
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union, Sequence

from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Utils.io.logger import Logger

import torch
from torch.utils.data import DataLoader

from timm.data.loader import PrefetchLoader

from argparse import Namespace
from PSZS.Optimizer import *

if TYPE_CHECKING:
    from PSZS.Models import CustomModel

def get_optim(method:str,
              train_source_iter: ForeverDataIterator,
              train_target_iter: ForeverDataIterator,
              val_loader: Union[DataLoader, PrefetchLoader],
              model: CustomModel,
              device: torch.device,
              args: Namespace,
              logger: Logger,
              eval_classes: Iterable[int],
              grad_accum_steps: int = 1,
              mixup_fn: Optional[Callable] = None,
              loss_scaler: Optional[torch.cuda.amp.GradScaler] = None,
              amp_autocast = suppress,
              iter_names: Optional[Sequence[str]]=None,
              **optim_kwargs) -> Base_Optimizer:
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
    print(f"Creating optimizer for method: {method}")
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
                            **optim_params)
    return optim