from collections import OrderedDict

import time
from typing import Iterable, List, Optional, Tuple, Sequence
import warnings

import torch

from torch.utils.data import DataLoader

from timm.scheduler.scheduler import Scheduler

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import METHOD_MODEL_MAP, UJDA_Model, save_checkpoint
from PSZS.Utils.io.logger import Logger
from PSZS.Utils.io import filewriter

EXTENDED_TRAIN_PRED_TYPE = Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor]

class UJDA_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: UJDA_Model, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 num_steps_adv: int = 4,
                 joint_loss_weight: float = 1,
                 vat_loss_weight: float = 1,
                 entropy_loss_weight: float = 0.1,
                 discrepancy_loss_weight: float = 1,
                 joint_cls_loss_labeled_weight: float = 1,
                 joint_cls_loss_unlabeled_weight: float = 1,
                 joint_adv_loss_weight: float = 0.1,
                 **optim_kwargs,
                 ) -> None:    
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
        assert isinstance(self.model, METHOD_MODEL_MAP['ujda']), \
            f"UJDA_Multiple model must be of type {METHOD_MODEL_MAP['ujda'].__name__}"
        self.model : UJDA_Model
        self.phase = 1
        # Subphases for phase 2
        self.subphase = "1"
        self.num_iters_phase_1 = iters_per_epoch
        self.num_iters_phase_2 = iters_per_epoch
        # The number of internal iteration steps for the "third phase" (i.e. the adversarial training)
        # which is part of the second phase
        self.num_steps_adv = num_steps_adv
        self.joint_loss_weight = joint_loss_weight
        self.vat_loss_weight = vat_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.discrepancy_loss_weight = discrepancy_loss_weight
        self.joint_cls_loss_labeled_weight = joint_cls_loss_labeled_weight
        self.joint_cls_loss_unlabeled_weight = joint_cls_loss_unlabeled_weight
        self.joint_adv_loss_weight = joint_adv_loss_weight
        
        # Set the functions for the first phase that are used in the base class
        # For phase 2 the individual functions are called in _train_computation_p2
        self._forward_train = self._forward_train_phase_1
        self._compute_loss = self._compute_loss_phase_1
    
    @property
    def iters_per_epoch(self) -> int:
        # If called before phase is set (i.e. during __init__) return dummy value of 1 by default. 
        # The correct num_batches is set in _set_progress_bar
        if hasattr(self, 'phase') == False:
            return 1
        elif self.phase == 1:
            return self.num_iters_phase_1
        elif self.phase == 2:
            return self.num_iters_phase_2
        else:
            raise ValueError(f"Phase {self.phase} not supported")
    @iters_per_epoch.setter
    def iters_per_epoch(self, value: int) -> None:
        # If called before phase is set (i.e. during __init__) do nothing
        if hasattr(self, 'phase') == False:
            return
        if self.phase == 1:
            self.num_iters_phase_1 = value
        elif self.phase == 2:
            self.num_iters_phase_2 = value
        else:
            raise ValueError(f"Phase {self.phase} not supported")
        
    @property
    def optimizers(self) -> List[torch.optim.Optimizer]:
        """Return only first or only second optimizer depending on the phase.
        If no matching phase return all optimizers."""
        # If called before constructer is finished (i.e. during super().__init__) return all optimizers
        if hasattr(self, 'phase') == False:
            return super().optimizers
        if self.phase == 1:
            # Phase 1 uses all optimizers
            return super().optimizers
        elif self.phase == 2 and self.subphase == '1':
            # Phase 2_1 only uses the second optimizer (classifier)
            # TODO Check why UJDA also steps the optimizer for backbone here... (paper says only classifier is updated)
            return [super().optimizers[1]]
        elif self.phase == 2 and self.subphase == '2':
            # Phase 2_2 uses the third and fourth optimizer (joint losses)
            return [super().optimizers[2], super().optimizers[3]]
        elif self.phase == 2 and self.subphase == '3':
            # Phase 2_3 only uses the first optimizer (backbone)
            return [super().optimizers[0]]
        else:
            return super().optimizers
        
    @property
    def lr_schedulers(self) -> List[Scheduler]:
        """Return only first or only second scheduler depending on the phase.
        If no matching phase return all schedulers."""
        if hasattr(self, 'phase') == False:
            return super().lr_schedulers
        if self.phase == 1:
            # Phase 1 uses all optimizers
            return super().lr_schedulers
        elif self.phase == 2 and self.subphase == '1':
            # Phase 2_1 only uses the second optimizer (classifier)
            # TODO Check why UJDA also steps the optimizer for backbone here... (paper says only classifier is updated)
            return [super().lr_schedulers[1]]
        elif self.phase == 2 and self.subphase == '2':
            # Phase 2_2 uses the third and fourth optimizer (joint losses)
            return [super().lr_schedulers[2], super().lr_schedulers[3]]
        elif self.phase == 2 and self.subphase == '3':
            # Phase 2_3 only uses the first optimizer (backbone)
            return [super().lr_schedulers[0]]
        else:
            return super().lr_schedulers
        
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        """Mainly create the meters for all the phases.
        Only the meters for the joint losses are added to the progress as they are used in every phase.
        Setting of the other meters is done in _set_progress_bar."""
        self.meter_loss_joint_1 = StatsMeter.get_stats_meter_min_max('Joint Loss 1', fmt=":3.2f",)
        self.meter_loss_joint_2 = StatsMeter.get_stats_meter_min_max('Joint Loss 2', fmt=":3.2f",)
        self.meter_entropy_loss = StatsMeter.get_stats_meter_min_max('Entropy Loss', fmt=":3.2f",)
        self.meter_vat_loss = StatsMeter.get_stats_meter_min_max('VAT Loss', fmt=":3.2f",)
        self.meter_discrepancy = StatsMeter.get_stats_meter_min_max('Discrepancy', fmt=":3.2f",)
        self.meter_adversarial = StatsMeter.get_stats_meter_min_max('Adversarial', fmt=":3.2f",)
        self.meter_total_loss_1 = StatsMeter.get_stats_meter_min_max('Total Loss (Phase 1)', fmt=":3.2f",)
        self.meter_total_loss_2_1 = StatsMeter.get_stats_meter_min_max('Total Loss (Phase 2[1])', fmt=":3.2f",)
        self.meter_total_loss_2_2 = StatsMeter.get_stats_meter_min_max('Total Loss (Phase 2[2])', fmt=":3.2f",)
        self.meter_total_loss_2_3 = StatsMeter.get_stats_meter_min_max('Total Loss (Phase 2[3])', fmt=":3.2f",)
        
        
        train_progress.add_meter(self.meter_loss_joint_1, exclude_simple_reset=True)
        train_progress.add_meter(self.meter_loss_joint_2, exclude_simple_reset=True)
        return train_progress, val_progress
    
    def _set_progress_bar(self, phase: Optional[str]=None) -> None:
        if phase is None:
            phase = self.phase
        print(f"Setting progress bar for phase: {phase}")
        if phase == 1:
            # Remove all meters from other phases
            self.progress_bar_train.remove_meter(self.meter_entropy_loss)
            self.progress_bar_train.remove_meter(self.meter_vat_loss)
            self.progress_bar_train.remove_meter(self.meter_discrepancy)
            self.progress_bar_train.remove_meter(self.meter_adversarial)
            self.progress_bar_train.remove_meter(self.meter_total_loss_2_1)
            self.progress_bar_train.remove_meter(self.meter_total_loss_2_2)
            self.progress_bar_train.remove_meter(self.meter_total_loss_2_3)
            # Add relevant meter for phase 1
            self.progress_bar_train.add_meter(self.meter_total_loss_1, exclude_simple_reset=True)
            # Set num_batches
            self.progress_bar_train.set_num_batches(self.num_iters_phase_1)
        elif phase == 2:
            # Remove all meters from other phases
            self.progress_bar_train.remove_meter(self.meter_total_loss_1)
            # Add relevant meters for phase 2 (2_1 and 2_2) and phase 3
            self.progress_bar_train.add_meter(self.meter_entropy_loss, exclude_simple_reset=True)
            self.progress_bar_train.add_meter(self.meter_vat_loss, exclude_simple_reset=True)
            self.progress_bar_train.add_meter(self.meter_discrepancy, exclude_simple_reset=True)
            self.progress_bar_train.add_meter(self.meter_adversarial, exclude_simple_reset=True)
            self.progress_bar_train.add_meter(self.meter_total_loss_2_1, exclude_simple_reset=True)
            self.progress_bar_train.add_meter(self.meter_total_loss_2_2, exclude_simple_reset=True)
            self.progress_bar_train.add_meter(self.meter_total_loss_2_3, exclude_simple_reset=True)  
            # Set num_batches
            self.progress_bar_train.set_num_batches(self.num_iters_phase_2)
        else:
            raise ValueError(f"Phase {phase} not supported")
        
    def _set_phase(self, phase: int) -> None:
        self._set_progress_bar(phase)
        if phase == 1:
            # The functions for phase 1 are already set in __init__
            self._train_computation = super()._train_computation
        elif phase == 2:
            # Only need to set the _train_computation pointer as 
            # the functions for the subphases are called in _train_computation_p2
            self._train_computation = self._train_computation_p2
        else:
            raise ValueError(f"Phase {phase} not supported")
        # Set phase only after successfully setting the functions
        self.phase = phase

    def _forward_train_phase_1(self, data: Tuple[torch.Tensor, torch.Tensor]
                               ) -> Tuple[EXTENDED_TRAIN_PRED_TYPE, None]:
        """For phase 1 only the classification loss is computed from a single forward pass.
        As the features are not needed we can discard them."""
        y, _ = super()._forward(data)
        return y, None
    
    def _forward_train_phase_2_1(self, data: Tuple[torch.Tensor, torch.Tensor]
                                 ) -> Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE]:
        """For phase 2 (first part) the Virtual Adversarial Training (VAT) components and resulting
        predictions are computed. Note that in model.vat() some gradients and backwards passes are computed already.
        Note that only a single other forward pass is computed instead of original UJDA where
        each loss computation will call a forward pass again on the same data.
        This is both computationally expensive and not necessary as the data is the same even potentially
        causing overfitting or error accumulation."""
        data = torch.cat(data, dim=0)
        # vat_s = self.model.vat(data_s, radius=self.vat_radius)
        # self._zero_grad() # Reset gradients as in .vat() gradients are computed
        # vat_t = self.model.vat(data_t, radius=self.vat_radius)
        data_vat = self.model.vat(data)
        self._zero_grad() # Reset gradients as in .vat() gradients are computed
        # Only do a single forward pass as compared to original UJDA where each loss makes its own forward pass
        (pred_cls, _, _), _ = self.model(data)
        # Instead of making two forward passes we can concatenate the data and make a single forward pass
        (pred_vat, _, _), _ = self.model(data_vat)
        return pred_cls, pred_vat
    
    def _forward_train_phase_2_2(self, 
                                 data: Tuple[torch.Tensor, torch.Tensor]
                                 ) -> EXTENDED_TRAIN_PRED_TYPE:
        """For phase 2 (second part) only do a simple forward pass.
        As for phase 2 (first part) this differs from UJDA where each loss computation makes its own forward pass."""
        # Only do a single forward pass as compared to original UJDA where each loss makes its own forward pass
        data = torch.cat(data, dim=0)
        (pred_cls, pred_joint_1, pred_joint_2), _ = self.model(data)
        return pred_cls, pred_joint_1, pred_joint_2
    
    def _forward_train_phase_2_3(self, 
                                 data: Tuple[torch.Tensor, torch.Tensor]
                                 ) -> Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor]:
        """For phase 2 (third part) only do a simple forward pass.
        As for the rest of phase 2 this differs from UJDA where each loss computation makes its own forward pass.
        This is the same as for phase 2 (second part) but for consistency and potential future changes
        they are kept separate."""
        # Only do a single forward pass as compared to original UJDA where each loss makes its own forward pass
        data = torch.cat(data, dim=0)
        (pred_cls, pred_joint_1, pred_joint_2), _ = self.model(data)
        return pred_cls, pred_joint_1, pred_joint_2
    
        
    def _compute_loss_phase_1(self, 
                              pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor], 
                              target: Tuple[torch.Tensor, torch.Tensor],
                              features: None=None,
                              og_labels: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                              ) -> torch.Tensor:
        pred_cls, pred_joint_1, pred_joint_2 = pred
        # Updating of losses meter is done in base class
        cls_loss = super()._compute_loss(pred=pred_cls, target=target, features=features, og_labels=og_labels)
        joint_loss_1, joint_loss_2 = self.model.compute_joint_loss_labeled(pred_joint_1, pred_joint_2, target)
        self.meter_loss_joint_1.update(joint_loss_1.item(), self.batch_size)
        self.meter_loss_joint_2.update(joint_loss_2.item(), self.batch_size)

        total_loss = cls_loss + self.joint_loss_weight*(joint_loss_1 + joint_loss_2)
        self.meter_total_loss_1.update(total_loss.item(), self.batch_size)
        return total_loss
    
    def _compute_loss_phase_2_1(self, 
                                pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE], 
                                target: Tuple[torch.Tensor, torch.Tensor],
                                features: None=None,
                                og_labels: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                                ) -> torch.Tensor:
        pred_cls, pred_vat = pred
        cls_loss = super()._compute_loss(pred=pred_cls, target=target, features=features, og_labels=og_labels)
        if self.model.head_type.returns_multiple_outputs:
            pred_cls = pred_cls[self.model.classifier.test_head_pred_idx]
            pred_vat = pred_vat[self.model.classifier.test_head_pred_idx]
        vat_loss = self.model.compute_loss_vat(pred_cls, pred_vat)
        self.meter_vat_loss.update(vat_loss.item(), self.batch_size)
        entropy_loss = self.model.compute_loss_entropy(pred_cls)
        self.meter_entropy_loss.update(entropy_loss.item(), self.batch_size)
        
        total_loss = cls_loss + self.vat_loss_weight * vat_loss + self.entropy_loss_weight * entropy_loss
        self.meter_total_loss_2_1.update(total_loss.item(), self.batch_size)
        return total_loss
    
    def _compute_loss_phase_2_2(self, 
                                pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor], 
                                target: Tuple[torch.Tensor, torch.Tensor],
                                features: None=None,
                                og_labels: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                                ) -> torch.Tensor:
        pred_cls, pred_joint_1, pred_joint_2 = pred
        if self.model.head_type.returns_multiple_outputs:
            pred_cls = pred_cls[self.model.classifier.test_head_pred_idx]
            target = [trg[..., self.model.classifier.test_head_pred_idx] for trg in target]
        joint_loss_1_l, joint_loss_2_l = self.model.compute_joint_loss_labeled(pred_joint_1, pred_joint_2, target)
        joint_loss_1_ul, joint_loss_2_ul = self.model.compute_joint_loss_unlabeled(pred_cls=pred_cls,
                                                                                   pred_joint_1=pred_joint_1,
                                                                                   pred_joint_2=pred_joint_2)
        loss_discrepancy = self.model.compute_loss_discrepancy(pred_joint_1=pred_joint_1,
                                                               pred_joint_2=pred_joint_2)
        self.meter_loss_joint_1.update(joint_loss_1_l.item() + joint_loss_1_ul.item(), self.batch_size)
        self.meter_loss_joint_2.update(joint_loss_2_l.item() + joint_loss_2_ul.item(), self.batch_size)
        self.meter_discrepancy.update(loss_discrepancy.item(), self.batch_size)
        
        total_loss = self.joint_cls_loss_labeled_weight*(joint_loss_1_l + joint_loss_2_l) + \
            self.joint_cls_loss_unlabeled_weight*(joint_loss_1_ul + joint_loss_2_ul) - \
                self.discrepancy_loss_weight * loss_discrepancy
        self.meter_total_loss_2_2.update(total_loss.item(), self.batch_size)
        return total_loss
    
    def _compute_loss_phase_2_3(self, 
                                pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor], 
                                target: Tuple[torch.Tensor, torch.Tensor],
                                features: None=None,
                                og_labels: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                                ) -> torch.Tensor:
        pred_cls, pred_joint_1, pred_joint_2 = pred
        if self.model.head_type.returns_multiple_outputs:
            pred_cls = pred_cls[self.model.classifier.test_head_pred_idx]
            target = [trg[..., self.model.classifier.test_head_pred_idx] for trg in target]
        adv_joint_loss = self.model.compute_loss_adv(pred_cls=pred_cls, 
                                                      pred_joint_1=pred_joint_1, 
                                                      pred_joint_2=pred_joint_2,
                                                      target=target)
        loss_discrepancy = self.model.compute_loss_discrepancy(pred_joint_1=pred_joint_1,
                                                               pred_joint_2=pred_joint_2)
        self.meter_discrepancy.update(loss_discrepancy.item(), self.batch_size)
        self.meter_adversarial.update(adv_joint_loss.item(), self.batch_size)
        
        total_loss = self.joint_adv_loss_weight * adv_joint_loss + \
            self.discrepancy_loss_weight * loss_discrepancy
        self.meter_total_loss_2_3.update(total_loss.item(), self.batch_size)
        return total_loss
        
    def _calculate_metrics_train_single_output(self, 
                                               pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE],
                                               target: Tuple[torch.Tensor],
                                               og_labels: Tuple[torch.Tensor],
                                               features: Optional[Sequence[torch.Tensor]]=None,
                                               ) -> Tuple[List[float], List[float], List[int],
                                                           List[float], List[float], List[float],
                                                           List[float], List[int]]:
        """Calculate the metrics for the training phase.
        Wrapper around the base class function to handle the multiple outputs of the model.
        Which returns (pred_cls, pred_joint_1, pred_joint_2) but only the classification prediction is used."""
        return super()._calculate_metrics_train_single_output(pred=pred[0], 
                                                              target=target, 
                                                              og_labels=og_labels,
                                                              features=features, )
    
    def _calculate_metrics_train_multiple_output(self, 
                                                 pred: Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE],
                                                 target: Tuple[torch.Tensor],
                                                 og_labels: Tuple[torch.Tensor],
                                                 features: Optional[Sequence[torch.Tensor]]=None,
                                                 ) -> Tuple[List[float], List[float], List[int],
                                                           List[float], List[float], List[float],
                                                           List[float], List[int]]:
        """Calculate the metrics for the training phase.
        Wrapper around the base class function to handle the multiple outputs of the model.
        Which returns (pred_cls, pred_joint_1, pred_joint_2) but only the classification prediction is used."""
        return super()._calculate_metrics_train_multiple_output(pred=pred[0], 
                                                              target=target, 
                                                              og_labels=og_labels,
                                                              features=features, )
    
    def _get_train_results(self) -> OrderedDict:
        results = super()._get_train_results()
        results.update([(f'Joint_Loss_1', self.meter_loss_joint_1.get_avg()),
                        (f'Joint_Loss_2', self.meter_loss_joint_2.get_avg()),
                        (f'Entropy_Loss', self.meter_entropy_loss.get_avg()),
                        (f'VAT_Loss', self.meter_vat_loss.get_avg()),
                        (f'Discrepancy_Loss', self.meter_discrepancy.get_avg()),
                        (f'Adversarial_Loss', self.meter_adversarial.get_avg()),])
        return results
    
    def _update_params(self, loss:torch.Tensor, need_update:bool) -> None:
        """Updates the parameters of the model and optimizers.
        The gradients of all optimizers (independent of phase) are reset after the update."""
        super()._update_params(loss, need_update)
        # Reset all gradients if an update was performed
        # The super call only affects the currently active optimizers
        if need_update:
            self._zero_grad()
            
    def _train_computation_p2(self, 
                              data: Tuple[torch.Tensor], 
                              labels: Tuple[torch.Tensor], 
                              accum_steps: int,
                              need_update: bool) -> None:
        # Map labels to internal index
        # _map_labels is defined in Base_Multiple
        interal_labels = self._map_labels(labels=labels, mode='pred')
        """ Phase 2_1 """
        self.subphase = "1"
        # Use autocast for mixed precision
        with self.amp_autocast():
            # Forward Pass/Compute Output
            pred = self._forward_train_phase_2_1(data)
            # Compute Loss and update meters
            loss = self._compute_loss_phase_2_1(pred=pred, target=labels)
            # Scale loss if accumulation to preserve same impact in backward
            if self.scale_loss_accum and accum_steps > 1:
                loss /= accum_steps
        # Compute Gradient and do SGD/Optimizer step if update required
        # outside of autocast scope (not sure if necessary but to be safe)
        self._update_params(loss=loss, need_update=need_update)
        
        """ Phase 2_2 """
        self.subphase = "2"
        # Reestablish autocast for mixed precision
        with self.amp_autocast():
            # Forward Pass/Compute Output
            pred = self._forward_train_phase_2_2(data)
            # Compute Loss and update meters
            loss = self._compute_loss_phase_2_2(pred=pred, target=labels)
            # Scale loss if accumulation to preserve same impact in backward
            if self.scale_loss_accum and accum_steps > 1:
                loss /= accum_steps
        # Compute Gradient and do SGD/Optimizer step if update required
        # outside of autocast scope (not sure if necessary but to be safe)
        self._update_params(loss=loss, need_update=need_update)

        """ Phase 2_3 """
        self.subphase = "3"
        # For the adversarial training we need to do all the steps
        # for the specified number of times
        for _ in range(self.num_steps_adv):
            # Reestablish autocast for mixed precision
            with self.amp_autocast():
                # Forward Pass/Compute Output
                pred = self._forward_train_phase_2_3(data)
                # Compute Loss and update meters
                loss = self._compute_loss_phase_2_3(pred=pred, target=labels)
                # Scale loss if accumulation to preserve same impact in backward
                if self.scale_loss_accum and accum_steps > 1:
                    loss /= accum_steps
            # Compute Gradient and do SGD/Optimizer step if update required
            # outside of autocast scope (not sure if necessary but to be safe)
            self._update_params(loss=loss, need_update=need_update)
        
        if self.eval_during_train:
            # Compute Eval Metrics (Accuracy) and Update Meters
            # Only done once in the phase as metrics are computed which are not updated in the subphases
            # Use the first part of the prediction from the last subphase
            # which corresponds to the classification prediction
            # This gets handled by _calculate_metrics_train_single_output/_multiple_output
            self._compute_eval_metrics(pred=pred, 
                                       target=interal_labels,
                                       og_labels=labels, 
                                       features=None, 
                                       train=True)
            
    def train(self, epochs: Optional[int]=None, start_epoch: Optional[int] = None) -> None:
        if epochs is None:
            epochs = self.num_epochs
            
        if epochs is None:
            warnings.warn('No number of epochs specified. No training.')
            return
        
        if start_epoch is not None:
            self.epoch = start_epoch
        
        start_time = time.time()     
        best_acc1 = 0.
        first_epoch = True
        metrics = OrderedDict()
        # start training
        best_acc1 = 0.
        print("Starting training for phase 1.")
        self._set_phase(1)
        for _ in range(epochs):
            # Optimizers are set properly in _set_phase to only contain relevant ones for the phase
            lrl = [param_group['lr'] for opt in self.optimizers for param_group in opt.param_groups]
            print(f'Learning rate: {",".join([f"{i:.4e}" for i in lrl])}')
            # No need to pass epoch as we set it appropriately in right before
            metrics = self.process_epoch()
            acc1 = self.cls_acc_1
            save_checkpoint(model=self.model, 
                            logger=self.logger, 
                            optimizer=self,
                            metric='acc1', 
                            current_best=best_acc1,
                            save_val_test=True)
            
            best_acc1 = max(acc1, best_acc1)
            filewriter.update_summary(epoch=self.epoch, 
                                      metrics=metrics, 
                                      root=self.logger.out_dir,
                                      write_header=first_epoch)
            first_epoch = False
        print("best_acc1 = {:3.1f}".format(best_acc1)) 
        p1_time = time.time()
        print(f'Total time for training Phase 1: {p1_time - start_time}')
        
        print("Starting training for phase 2.")
        # Set epoch to 1 to display progress correctly
        self.epoch = 1
        # Subphases are set in _train_computation_p2 i.e. in process_epoch
        self._set_phase(2)
        for _ in range(epochs):
            lrl = [param_group['lr'] for opt in self.optimizers for param_group in opt.param_groups]
            print(f'Learning rate: {",".join([f"{i:.4e}" for i in lrl])}')
            # No need to pass epoch as we set it appropriately in right before
            metrics = self.process_epoch()
            acc1 = self.cls_acc_1
            save_checkpoint(model=self.model, 
                            logger=self.logger, 
                            optimizer=self,
                            metric='acc1', 
                            current_best=best_acc1,
                            save_val_test=True)
            
            best_acc1 = max(acc1, best_acc1)
            filewriter.update_summary(epoch=self.epoch, 
                                      metrics=metrics, 
                                      root=self.logger.out_dir,
                                      write_header=first_epoch)
        print("best_acc1 = {:3.1f}".format(best_acc1)) 
        print(f'Total time for training Phase 2: {time.time() - p1_time}')
        print(f'Total time: {time.time() - start_time}')