from collections import OrderedDict

from typing import Iterable, Tuple, Sequence
import warnings

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import JAN_Model, METHOD_MODEL_MAP
from PSZS.Utils.io.logger import Logger

class JAN_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: JAN_Model, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 adaptation_filter_mode: str = 'ignore',
                 domain_loss_weight: float = 1.0,
                 **optim_kwargs,
                 ) -> None:    
        # No point in using JAN if domain_loss_weight is 0 (use ERM instead)
        if domain_loss_weight == 0:
            raise ValueError("Specify a domain_loss_weight != 0 for JAN.")
        self.domain_loss_weight = domain_loss_weight
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
            adaptation_filter_mode=adaptation_filter_mode,
            **optim_kwargs
        )
        if adaptation_filter_mode != 'ignore':
            # For the jmmd loss we need valid correspondences of source and target
            # and thus we can not do filtering when using linear kernel as this would 
            # break the correspondences
            # For non-linear methods a non-linear kernel matrix can handle mismatching dimensions
            assert (model.linear == False), "When using adaptation filter mode other than 'ignore', JAN must be non-linear"
        assert isinstance(self.model, METHOD_MODEL_MAP['jan']), \
            f"JAN_Multiple model must be of type {METHOD_MODEL_MAP['jan'].__name__}"
        self.model : JAN_Model
        self.source_pred_mask = self._construct_source_pred_mask()
            
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        self.meter_transfer_loss = StatsMeter.get_stats_meter_min_max('JMMD Loss', fmt=":3.2f",)
        train_progress.add_meter(self.meter_transfer_loss, exclude_simple_reset=True)
            
        return train_progress, val_progress
    
    def _compute_loss_adaptation(self, pred: TRAIN_PRED_TYPE,
                                 features: Sequence[torch.Tensor], 
                                 **others) -> torch.Tensor:
        f_s, f_t = features
        # For hierarchical models, each pred has multiple components
        # Thus we need to get only the relevant component 
        p_s, p_t = pred
        if self.model.classifier.returns_multiple_outputs:
            p_s = p_s[self.model.classifier.test_head_pred_idx]
            p_t = p_t[self.model.classifier.test_head_pred_idx]
        # Filter out the predictions that are not shared with the target domain
        # No need to change target domain predictions p_t
        p_s = p_s[:, self.source_pred_mask]
        
        jmmd_loss : torch.Tensor = self.model.jmmd_loss(feature_source=f_s,
                                                        feature_target=f_t,
                                                        pred_norm_source=softmax(p_s, dim=1),
                                                        pred_norm_target=softmax(p_t, dim=1))
        self.meter_transfer_loss.update(jmmd_loss.item(), self.batch_size)
            
        return self.domain_loss_weight * jmmd_loss
    
    def _get_train_results(self) -> OrderedDict:
        results = super()._get_train_results()
        results.update([(f'JMMD_Loss', self.meter_transfer_loss.get_avg())])
        return results