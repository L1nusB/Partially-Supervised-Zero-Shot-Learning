from collections import OrderedDict

from typing import Iterable, Tuple, Sequence

import torch
from torch.utils.data import DataLoader

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import CustomModel, METHOD_MODEL_MAP
from PSZS.Utils.io.logger import Logger
from PSZS.Alignment import MinimumClassConfusionLoss

class MCC_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: CustomModel, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 domain_loss_weight: float = 1.0,
                 temperature: float = 2.5,
                 source_mcc: bool = False,
                 **optim_kwargs,
                 ) -> None:    
        if temperature <= 0:
            raise ValueError("Temperature must be larger than 0")
        if domain_loss_weight == 0:
            raise ValueError("Specify a domain_loss_weight != 0.")
        # Need to be set before super as super calls _expand_progress_bars
        self.domain_loss_weight = domain_loss_weight
        self.temperature = temperature
        self.source_mcc = source_mcc
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
        assert isinstance(self.model, METHOD_MODEL_MAP['mcc']), \
            f"MCC_Multiple model must be of type {METHOD_MODEL_MAP['mcc'].__name__}"
        self.model : CustomModel
        self.mcc_loss = MinimumClassConfusionLoss(temperature=temperature).to(device)
            
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        self.meter_mcc_loss = StatsMeter.get_stats_meter_min_max('MCC Loss', fmt=":3.2f",)
        train_progress.add_meter(self.meter_mcc_loss, exclude_simple_reset=True)
        return train_progress, val_progress
    
    def _compute_loss_adaptation(self, pred: TRAIN_PRED_TYPE, **others) -> torch.Tensor:
        """Computes the domain adaptation loss for MCC.
        The features are assumed to be filtered based on `adaptation_filter_mode`.
        This is handeled in the `_compute_loss` method of Base_Optimizer.
        .. note::
            The `others` argument is required as Base_Optimizer will pass all arguments explicitly
            but they are not needed here.

        Args:
            pred (TRAIN_PRED_TYPE): Prediction logits to compute the MCC feature loss over.

        Returns:
            torch.Tensor: Domain adaptation loss scaled by `domain_loss_weight`.
        """
        # For hierarchical models, each pred has multiple components
        # Thus we need to get only the relevant component 
        p_s, p_t = pred
        if self.model.classifier.returns_multiple_outputs:
            p_t = p_t[self.model.classifier.test_head_pred_idx]
            p_s = p_s[self.model.classifier.test_head_pred_idx]
        if self.source_mcc:
            mcc_loss : torch.Tensor = self.mcc_loss(p_s) + self.mcc_loss(p_t)
        else:
            mcc_loss : torch.Tensor = self.mcc_loss(p_t)
        self.meter_mcc_loss.update(mcc_loss.item(), self.batch_size)
        return self.domain_loss_weight * mcc_loss
    
    def _get_train_results(self) -> OrderedDict:
        results = super()._get_train_results()
        results.update([(f'MCC_Loss', self.meter_mcc_loss.get_avg())])
        return results