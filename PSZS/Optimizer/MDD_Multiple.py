from collections import OrderedDict

from typing import Iterable, Optional, Tuple, Sequence, overload

import torch

from PSZS.Alignment.mdd import ClassificationMarginDisparityDiscrepancy

from torch.utils.data import DataLoader

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import MDD_Model, METHOD_MODEL_MAP
from PSZS.Utils.io.logger import Logger

class MDD_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: MDD_Model, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 domain_loss_weight: float = 1.0,
                 grl_epochs: Optional[int] = None,
                 margin: float = 4.,
                 **optim_kwargs,
                 ) -> None:    
        if domain_loss_weight == 0:
            print("Domain loss weight is 0, norm loss will not be computed")
        # Need to be set before super as super calls _expand_progress_bars
        self.domain_loss_weight = domain_loss_weight
        super().__init__(
            train_iters=train_iters,
            val_loader=val_loader,
            model=model,
            device=device,
            iters_per_epoch=iters_per_epoch,
            print_freq=print_freq,
            batch_size=batch_size,
            eval_classes=eval_classes,
            logger=logger,
            **optim_kwargs
        )
        assert isinstance(self.model, METHOD_MODEL_MAP['mdd']), \
            f"MDD_Multiple model must be of type {METHOD_MODEL_MAP['mdd'].__name__}"
        self.model : MDD_Model
        # Dynamically set the max_iters for the GRL based on epoch count
        # can not be done during model construction as epochs and iters are not known
        if grl_epochs is not None:
            self.model.grl.max_iters = grl_epochs * iters_per_epoch
        self.margin = margin
        self.mdd = ClassificationMarginDisparityDiscrepancy(margin).to(device)
        self.mdd.train() # Will always stay in train as irrelevant for validation
            
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        self.meter_mdd_loss = StatsMeter.get_stats_meter_min_max('MDD Loss', fmt=":3.2f",)
        train_progress.add_meter(self.meter_mdd_loss, exclude_simple_reset=True)
        return train_progress, val_progress
    
    def _compute_loss_adaptation(self, pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor], **others) -> torch.Tensor:
        """Computes the domain adaptation loss for MDD.
        The features are assumed to be filtered based on `adaptation_filter_mode`.
        This is handeled in the `_compute_loss` method of Base_Optimizer.
        .. note::
            The `others` argument is required as Base_Optimizer will pass all arguments explicitly
            but they are not needed here.

        Args:
            pred (Tuple[TRAIN_PRED_TYPE, torch.Tensor]): Prediction logits and adversarial prediction to compute the MDD feature loss over.

        Returns:
            torch.Tensor: Domain adaptation loss scaled by `domain_loss_weight`.
        """
        (p_s, p_t), p_adv = pred
        if self.model.classifier.returns_multiple_outputs:
            p_s = p_s[self.model.classifier.test_head_pred_idx]
            p_t = p_t[self.model.classifier.test_head_pred_idx]
        # Need to be split here (not in MDD_Model.forward) to 
        # potentially allow for different batch sizes and non multiple
        # Split the adversarial predictions into source and target matching the batch sizes
        # obtained from the main predictions
        p_s_adv, p_t_adv = p_adv.tensor_split((p_s.size(0),))
        mdd_loss : torch.Tensor = -self.mdd(y_s=p_s, y_s_adv=p_s_adv, 
                                            y_t=p_t, y_t_adv=p_t_adv)
        self.meter_mdd_loss.update(mdd_loss.item(), self.batch_size)
        return self.domain_loss_weight * mdd_loss
    
    def _compute_loss_cls(self, 
                          pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor], 
                          target: Tuple[torch.Tensor], 
                          features: Sequence[torch.Tensor],
                          og_labels: Optional[Tuple[torch.Tensor]]=None) -> torch.Tensor:
        # Pred contains the main predictions and the adversarial predictions
        return super()._compute_loss(pred=pred[0],
                                     target=target,
                                     features=features,
                                     og_labels=og_labels)
    
    def _get_train_results(self) -> OrderedDict:
        results = super()._get_train_results()
        results.update([(f'MDD_Loss', self.meter_mdd_loss.get_avg())])
        return results