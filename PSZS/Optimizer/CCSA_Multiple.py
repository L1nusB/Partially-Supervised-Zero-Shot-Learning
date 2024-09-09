from collections import OrderedDict

from typing import Iterable, Tuple, Sequence

import torch
from torch.utils.data import DataLoader

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple
from PSZS.Models import CustomModel, METHOD_MODEL_MAP
from PSZS.Models.Losses import ContrastiveLoss
from PSZS.Utils.io.logger import Logger

class CCSA_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: CustomModel, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 alignment_weight: float = 0.25,
                 margin: float = 0.5,
                 **optim_kwargs,
                 ) -> None:    
        if alignment_weight <= 0:
            raise ValueError("alignment_weight must be larger than 0")
        # Need to be set before super as super calls _expand_progress_bars
        self.alignment_weight = alignment_weight
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
        assert isinstance(self.model, METHOD_MODEL_MAP['ccsa']), \
            f"CCSA_Multiple model must be of type {METHOD_MODEL_MAP['ccsa'].__name__}"
        self.alignment_module = ContrastiveLoss(margin=margin, mean=True).to(device)
            
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        self.meter_alignment_loss = StatsMeter.get_stats_meter_min_max('Alignment Loss', fmt=":3.2f",)
        train_progress.add_meter(self.meter_alignment_loss, exclude_simple_reset=True)
        return train_progress, val_progress
    
    def _compute_loss_adaptation(self, features: Sequence[torch.Tensor], 
                                 target: Tuple[torch.Tensor, torch.Tensor], **others) -> torch.Tensor:
        """Computes the domain adaptation loss for CCSA.
        The CCSA Domain loss is the contrastive semantic alignment loss.
        This equates to the contrastive loss between the source and target domain features.
        The features are assumed to be filtered based on `adaptation_filter_mode`.
        This is handeled in the `_compute_loss` method of Base_Optimizer.
        
        .. note::
            The `others` argument is required as Base_Optimizer will pass all arguments explicitly
            but they are not needed here.

        Args:
            features (Sequence[torch.Tensor]): Features to compute the CCSA alignment loss.
            target (Tuple[torch.Tensor, torch.Tensor]): Target labels to compute the CCSA alignment loss over.

        Returns:
            torch.Tensor: Domain adaptation loss scaled by `alignment_weight`.
        """
        features : torch.Tensor = torch.cat(features, dim=0)
        # In case adaptation mode filters the filters only account for actual features
        f_count = features.size(0)
        
        target : torch.Tensor = torch.cat(target, dim=0)
        if self.model.classifier.returns_multiple_outputs:
            target = target[:,self.model.classifier.test_head_pred_idx]
        
        # Compute Contrastive Semantic Alignment Loss over both domains together
        # this allows potential pairs to be compared when batches contain samples 
        # from the same class.
        csa_loss : torch.Tensor = self.alignment_module(features, target)
        self.meter_alignment_loss.update(csa_loss.item(), f_count)
        return self.alignment_weight * csa_loss
    
    def _get_train_results(self) -> OrderedDict:
        results = super()._get_train_results()
        results.update([(f'Alignment_Loss', self.meter_alignment_loss.get_avg())])
        return results