from typing import Iterable, Tuple, Sequence
import warnings

import torch

from torch.utils.data import DataLoader

from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple
from PSZS.Models import CustomModel
from PSZS.Utils.io.logger import Logger

class ERM_Multiple(Base_Multiple):
    """
    For ERM this is basically just a wrapper around the Base_Multiple class
    that does not allow domain adaptation, feature or logit loss.
    No modifications besides the forward pass not returning features.
    """
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: CustomModel, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 **optim_kwargs,
                 ) -> None:
        super().__init__(
            train_iters=train_iters,
            val_loader=val_loader,
            model=model,
            iters_per_epoch=iters_per_epoch,
            print_freq=print_freq,
            batch_size=batch_size,
            eval_classes=eval_classes,
            logger=logger,
            device=device,
            **optim_kwargs
        )
        # Disable feature and logit loss by setting weights to 0
        if self.has_feature_loss:
            warnings.warn("ERM does not support feature loss, feature loss will be disabled.")
            self.feature_loss_weight = 0
        if self.has_logit_loss:
            warnings.warn("ERM does not support logit loss, logit loss will be disabled.")
            self.logit_loss_weight = 0
    
    def _forward_train(self, data: Tuple[torch.Tensor]
                       ) -> Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]]:
        # For ERM we do not need features
        y, _ = super()._forward_train(data)
        return y, None