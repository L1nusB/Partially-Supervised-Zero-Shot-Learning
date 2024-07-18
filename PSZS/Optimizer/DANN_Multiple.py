from collections import OrderedDict

from typing import Iterable, Optional, Tuple, Sequence


import torch

from torch.utils.data import DataLoader

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import Domain_Adv_Model, METHOD_MODEL_MAP
from PSZS.Utils.io.logger import Logger

class DANN_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: Domain_Adv_Model, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 domain_loss_weight: float = 1.0,
                 grl_epochs: Optional[int] = None,
                 **optim_kwargs,
                 ) -> None:
        # No point in using DANN if domain_loss_weight is 0 (use ERM instead)
        if domain_loss_weight == 0:
            raise ValueError("Specify a domain_loss_weight != 0 for DANN.")
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
        assert isinstance(self.model, METHOD_MODEL_MAP['dann']), \
            f"DANN_Multiple model must be of type {METHOD_MODEL_MAP['dann'].__name__}"
        self.model : Domain_Adv_Model
        # Dynamically set the max_iters for the GRL based on epoch count
        # can not be done during model construction as epochs and iters are not known
        if grl_epochs is not None:
            self.model.domain_adversarial_loss.grl.max_iters = grl_epochs * iters_per_epoch
            
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        self.meter_transfer_loss = StatsMeter.get_stats_meter_min_max('Transfer Loss', fmt=":3.2f",)
        train_progress.add_meter(self.meter_transfer_loss, exclude_simple_reset=True)
        # Only get display/filled with values if eval_during_train is True
        if self.eval_during_train:
            self.meter_domain_acc = StatsMeter.get_stats_meter_min_max('Domain Acc', fmt=":3.2f",)
            train_progress.add_meter(self.meter_domain_acc, exclude_simple_reset=True)
        return train_progress, val_progress
    
    def _compute_loss_adaptation(self, features: Sequence[torch.Tensor], **others) -> torch.Tensor:
        """Computes the domain adaptation loss for DANN.
        The features are assumed to be filtered based on `adaptation_filter_mode`.
        This is handeled in the `_compute_loss` method of Base_Optimizer.
        .. note::
            The `others` argument is required as Base_Optimizer will pass all arguments explicitly
            but they are not needed here.

        Args:
            features (Sequence[torch.Tensor]): Features to compute the DANN loss over.

        Returns:
            torch.Tensor: Domain adaptation loss scaled by `domain_loss_weight`.
        """
        f_s, f_t = features
        # In case adaptation mode filters the filters only account for actual features
        f_count = sum([f_i.size(0) for f_i in features])
        transfer_loss : torch.Tensor = self.model.domain_adversarial_loss(f_s=f_s, f_t=f_t)
        self.meter_transfer_loss.update(transfer_loss.item(), f_count)
        return self.domain_loss_weight * transfer_loss
            
    def _compute_eval_metrics_train(self, 
                                    pred: TRAIN_PRED_TYPE, 
                                    target: Tuple[torch.Tensor, ...], 
                                    og_labels: Tuple[torch.Tensor, ...], 
                                    features: Optional[Sequence[torch.Tensor]]=None,
                                    ) -> None:
        super()._compute_eval_metrics_train(pred=pred, 
                                            target=target, 
                                            og_labels=og_labels,
                                            features=features,)
        domain_acc = self.model.domain_adversarial_loss.domain_discriminator_accuracy
        self.meter_domain_acc.update(domain_acc, self.batch_size)
        
    def _get_train_results(self) -> OrderedDict:
        results = super()._get_train_results()
        results.update([(f'Transfer_Loss', self.meter_transfer_loss.get_avg())])
        # Values/Meters are only updated if eval_during_train is True
        if self.eval_during_train:
            results.update([(f'Domain_Acc', self.meter_domain_acc.get_avg())])
        return results