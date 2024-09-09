from collections import OrderedDict

from typing import Iterable, List, Optional, Tuple, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from timm.scheduler.scheduler import Scheduler

from PSZS.Utils.meters import DynamicStatsMeter, StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple
from PSZS.Optimizer.Base_Optimizer import PRED_TYPE, TRAIN_PRED_TYPE
from PSZS.Models import MCD_Model, METHOD_MODEL_MAP, zero
from PSZS.Utils.io.logger import Logger

class MCD_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: MCD_Model, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 num_steps_phase_3: int = 4,
                 loss_entropy_weight: float = 0.01,
                 loss_discrepancy_weight: float = 1.0,
                 **optim_kwargs,
                 ) -> None:    
        assert loss_entropy_weight > 0, "Entropy loss weight must be greater than 0"
        assert loss_discrepancy_weight > 0, "Discrepancy loss weight must be greater than 0"
        # Need to be set before super as super calls _expand_progress_bars
        self.loss_entropy_weight = loss_entropy_weight
        self.loss_discrepancy_weight = loss_discrepancy_weight
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
        assert isinstance(self.model, METHOD_MODEL_MAP['mcd']), \
            f"MCD_Multiple model must be of type {METHOD_MODEL_MAP['mcd'].__name__}"
        self.model : MCD_Model
        self.num_steps_phase_3 = num_steps_phase_3
        self.phase = 1
        
    @property
    def optimizers(self) -> List[torch.optim.Optimizer]:
        """Return optimizer for backbone and/or classifier."""
        # If called before constructer is finished (i.e. during super().__init__) return all optimizers
        if hasattr(self, 'phase') == False:
            return super().optimizers
        if self.phase == 1:
            # Phase 1 uses all optimizers
            return super().optimizers
        elif self.phase == 2:
            # Phase 2 optimizes only the classifiers
            return [super().optimizers[1]]
        elif self.phase == 3:
            # Phase 3 optimizes only the backbone
            return [super().optimizers[0]]
        else:
            return super().optimizers
        
    @property
    def lr_schedulers(self) -> List[Scheduler]:
        """Return lr_schedulers for backbone and/or classifier."""
        if self.phase == 1:
            # Phase 1 uses all optimizers
            return super().lr_schedulers
        elif self.phase == 2:
            # Phase 2 optimizes only the classifiers
            return [super().lr_schedulers[1]]
        elif self.phase == 3:
            # Phase 3 optimizes only the backbone
            return [super().lr_schedulers[0]]
        else:
            return super().lr_schedulers
            
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        
        self.meter_cls_loss.name = "Cls Loss [1]"
        self.meter_cls_loss2 = DynamicStatsMeter.get_stats_meter_min_max("Cls Loss [2]", 
                                                                fields=self.watched_fields_loss, 
                                                                fmt=":6.2f")
        self.meter_entropy = StatsMeter.get_stats_meter_min_max('Entropy Loss', fmt=":3.2f",)
        self.meter_discrepancy = StatsMeter.get_stats_meter_min_max('Discrepancy Loss', fmt=":3.2f",)
        
        train_progress.add_meter(self.meter_cls_loss2, exclude_simple_reset=True)
        train_progress.add_meter(self.meter_entropy, exclude_simple_reset=True)
        train_progress.add_meter(self.meter_discrepancy, exclude_simple_reset=True)
        # Remove total loss as it becomes somewhat skewed as we update it for each phase 
        # but each phase has different losses. 
        # And adding separate total_loss for each phase would be too much
        train_progress.remove_meter(self.meter_total_loss)
        return train_progress, val_progress
    
    def _entropy(self, pred: torch.Tensor) -> torch.Tensor:
        r"""Entropy of N predictions :math:`(p_1, p_2, ..., p_N)`.

        .. note::
            This entropy function is specifically used in MCD and different from the usual entropy function.
        """ 
        return -torch.mean(torch.log(torch.mean(pred, dim=0) + 1e-6))
    
    def _classifier_discrepancy(self, predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:
        r"""The `Classifier Discrepancy` in
        `Maximum Classiﬁer Discrepancy for Unsupervised Domain Adaptation (CVPR 2018) <https://arxiv.org/abs/1712.02560>`_.

        The classfier discrepancy between predictions :math:`p_1` and :math:`p_2`

        Args:
            predictions1 (torch.Tensor): Classifier predictions :math:`p_1`. Expected to contain raw, normalized scores for each class
            predictions2 (torch.Tensor): Classifier predictions :math:`p_2`
        """
        return torch.mean(torch.abs(predictions1 - predictions2)) 
    
    def _update_params(self, loss:torch.Tensor, need_update:bool) -> None:
        """Updates the parameters of the model and optimizers.
        The gradients of all optimizers (independent of phase) are reset after the update."""
        super()._update_params(loss, need_update)
        # Reset all gradients if an update was performed
        # The super call only affects the currently active optimizers
        if need_update:
            self._zero_grad()
            
    def _filter_loss_components(self,
                                pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE],
                                target: Tuple[torch.Tensor,torch.Tensor],
                                features: Optional[Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]]=None,
                                og_labels: Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
                                mode: Optional[str]=None,
                                ) -> Tuple[Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE], Tuple[torch.Tensor,torch.Tensor], Optional[Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]], Optional[Tuple[torch.Tensor,torch.Tensor]]]:
        """For now no direct support of filter_loss_components for MCD model."""
        return pred, target, features, og_labels
    
    def __compute_loss_cls(self, 
                          pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE], 
                          target: Tuple[torch.Tensor,torch.Tensor],) -> torch.Tensor:
        """Same logic as _compute_loss_cls but works for both classifiers and updates respective meters.
        Returns the total summed loss of both classifiers."""
        pred_1, pred_2 = pred
        loss_1, loss_components_1 = self.model.compute_cls_loss(pred_1, target, mixup=self.uses_mixup)
        loss_2, loss_components_2 = self.model.compute_cls_loss(pred_2, target, mixup=self.uses_mixup)
        
        # In case of uneven split between source and target shared we have to scale the loss
        # and set the number of samples based on the actual counts
        # Shape of target is guaranteed as compared to pred which could be different
        self.meter_cls_loss.update(vals=[loss_1] + loss_components_1,
                           n=[self.batch_size] + [trg.size(0) for trg in target])
        self.meter_cls_loss2.update(vals=[loss_2] + loss_components_2,
                           n=[self.batch_size] + [trg.size(0) for trg in target])
        return loss_1 + loss_2
    
    def _compute_loss_features(self, features: Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]], 
                               target: Tuple[torch.Tensor,torch.Tensor]) -> torch.Tensor:
        # Need to specify scope when using super in list comprehension
        # https://stackoverflow.com/questions/59520967/super-keyword-doesnt-work-properly-in-list-comprehension
        return sum([super(MCD_Multiple, self)._compute_loss_features(f, target) for f in features])
    
    def _compute_loss_logits(self, pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE]) -> torch.Tensor:
        # Need to specify scope when using super in list comprehension
        # https://stackoverflow.com/questions/59520967/super-keyword-doesnt-work-properly-in-list-comprehension
        return sum([super(MCD_Multiple, self)._compute_loss_logits(p) for p in pred])
    
    def _compute_loss_adaptation_phase1(self, pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE], **others) -> torch.Tensor:
        if self.model.classifier.returns_multiple_outputs:
            pred_1, pred_2 = pred
            (_, pred_1_t) = pred_1[self.model.classifier.test_head_pred_idx]
            (_, pred_2_t) = pred_2[self.model.classifier.test_head_pred_idx]
        else:
            (_, pred_1_t), (_, pred_2_t) = pred
        
        entropy_loss = self._entropy(F.softmax(torch.cat((pred_1_t, pred_2_t), dim=0), dim=1))
        self.meter_entropy.update(entropy_loss.item(), self.batch_size)
        return self.loss_entropy_weight * entropy_loss
    
    def _compute_loss_adaptation_phase2(self, pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE], **others) -> torch.Tensor:
        if self.model.classifier.returns_multiple_outputs:
            pred_1, pred_2 = pred
            (_, pred_1_t) = pred_1[self.model.classifier.test_head_pred_idx]
            (_, pred_2_t) = pred_2[self.model.classifier.test_head_pred_idx]
        else:
            (_, pred_1_t), (_, pred_2_t) = pred
        
        target_pred_scores_1_t = F.softmax(pred_1_t, dim=1)
        target_pred_scores_2_t  = F.softmax(pred_2_t, dim=1)
        entropy_loss = self._entropy(target_pred_scores_1_t) + self._entropy(target_pred_scores_2_t)
        discrepancy_loss = self._classifier_discrepancy(target_pred_scores_1_t, target_pred_scores_2_t)
        self.meter_entropy.update(entropy_loss.item(), self.batch_size)
        self.meter_discrepancy.update(discrepancy_loss.item(), self.batch_size)
            
        return self.loss_entropy_weight * entropy_loss - self.loss_discrepancy_weight * discrepancy_loss
    
    def _compute_loss_adaptation_phase3(self, pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE], **others) -> torch.Tensor:
        if self.model.classifier.returns_multiple_outputs:
            pred_1, pred_2 = pred
            (_, pred_1_t) = pred_1[self.model.classifier.test_head_pred_idx]
            (_, pred_2_t) = pred_2[self.model.classifier.test_head_pred_idx]
        else:
            (_, pred_1_t), (_, pred_2_t) = pred
        discrepancy_loss = self._classifier_discrepancy(F.softmax(pred_1_t, dim=1),
                                                        F.softmax(pred_2_t, dim=1))
        self.meter_discrepancy.update(discrepancy_loss.item(), self.batch_size)
        
        return self.loss_discrepancy_weight * discrepancy_loss
    
    def _compute_eval_metrics_train(self, 
                                    pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE], 
                                    target: Tuple[torch.Tensor,torch.Tensor], 
                                    og_labels: Tuple[torch.Tensor,torch.Tensor], 
                                    features: Optional[Tuple[Sequence[torch.Tensor],Sequence[torch.Tensor]]]=None,
                                    ) -> None:
        # Use the first part of the prediction which corresponds to normal classification prediction
        return super()._compute_eval_metrics_train(pred=pred[0], 
                                                   target=target, 
                                                   og_labels=og_labels,
                                                   features=features)
    
    def _train_computation(self, 
                           data: Tuple[torch.Tensor], 
                           labels: Tuple[torch.Tensor], 
                           accum_steps: int,
                           need_update: bool) -> None:
        # Only evaluate once after all phases have run
        eval_train = self.eval_during_train
        self.eval_during_train = False
        ### Phase 1 ###
        self.phase = 1
        # Reset the loss computation to use both classifiers for phase 1 and 2
        self._compute_loss_cls = self.__compute_loss_cls
        self._compute_loss_adaptation = self._compute_loss_adaptation_phase1
        super()._train_computation(data, labels, accum_steps, need_update)

        ### Phase 2 ###
        self.phase = 2
        self._compute_loss_adaptation = self._compute_loss_adaptation_phase2
        super()._train_computation(data, labels, accum_steps, need_update)
        
        ### Phase 3 ###
        # For the phase 3 we need to do all the steps
        # for the specified number of times
        self.phase = 3
        self._compute_loss_cls = zero # Phase 3 does not use classification loss
        self._compute_loss_adaptation = self._compute_loss_adaptation_phase3
        # Separate the number of steps for phase 3 to allow for train eval in final iteration
        # and do not have an if condition in the loop
        for _ in range(self.num_steps_phase_3 - 1):
            super()._train_computation(data, labels, accum_steps, need_update)
        # Reset eval_during_train to its original (i.e. allow eval during training)
        self.eval_during_train = eval_train
        super()._train_computation(data, labels, accum_steps, need_update)
            
            
    def _get_train_results(self) -> OrderedDict:
        results = super()._get_train_results()
        for watched_field in self.watched_fields_loss:
            results.update([(f'Cls_Loss_2{watched_field}', self.meter_cls_loss2.get_avg(watched_field))])
        results.update([(f'Entropy_Loss', self.meter_entropy.get_avg()),
                        (f'Discrepancy_Loss', self.meter_discrepancy.get_avg())])
        return results