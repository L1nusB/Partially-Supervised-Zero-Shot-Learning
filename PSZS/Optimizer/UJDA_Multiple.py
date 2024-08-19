from collections import OrderedDict

import time
from typing import Iterable, List, Optional, Tuple, Sequence
import warnings
import copy

import torch

from torch.utils.data import DataLoader

from timm.scheduler.scheduler import Scheduler

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import METHOD_MODEL_MAP, UJDA_Model, save_checkpoint, zero
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
        
    @property
    def progress_bar_train(self) -> ProgressMeter:
        if hasattr(self, 'phase') == False:
            return self._train_progress
        if self.phase == 1:
            return self.train_progress1
        elif self.phase == 2:
            return self.train_progress2
        else:
            return self._train_progress
    @progress_bar_train.setter
    def progress_bar_train(self, value: ProgressMeter) -> None:
        self._train_progress = value
        
    @property
    def meter_total_loss(self) -> StatsMeter:
        if hasattr(self, 'phase') == False:
            return self._meter_total_loss
        if self.phase == 1:
            return self.meter_total_loss_1
        elif self.phase == 2 and self.subphase == '1':
            return self.meter_total_loss_2_1
        elif self.phase == 2 and self.subphase == '2':
            return self.meter_total_loss_2_2
        elif self.phase == 2 and self.subphase == '3':
            return self.meter_total_loss_2_3
        else:
            return self._meter_total_loss
    @meter_total_loss.setter
    def meter_total_loss(self, value: StatsMeter) -> None:
        self._meter_total_loss = value
        
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        """Create the meters for all the phases. And construct the progress bars for the phases."""
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
        
        # Remove general total loss meter as we have separate ones for each phase
        train_progress.remove_meter(self.meter_total_loss)
        # These meters are used in all phases
        train_progress.add_meter(self.meter_loss_joint_1, exclude_simple_reset=True)
        train_progress.add_meter(self.meter_loss_joint_2, exclude_simple_reset=True)
        
        # Make shallow copies to keep the shared meters
        self.train_progress1: ProgressMeter = train_progress.copy()
        self.train_progress2: ProgressMeter = train_progress.copy()
        
        # Add relevant meter for phase 1 and set num_batches
        self.train_progress1.add_meter(self.meter_total_loss_1, exclude_simple_reset=True)
        self.train_progress1.set_num_batches(self.iters_per_epoch)
        # Add relevant meter for phase 2 and set num_batches
        self.train_progress2.add_meter(self.meter_entropy_loss, exclude_simple_reset=True)
        self.train_progress2.add_meter(self.meter_vat_loss, exclude_simple_reset=True)
        self.train_progress2.add_meter(self.meter_discrepancy, exclude_simple_reset=True)
        self.train_progress2.add_meter(self.meter_adversarial, exclude_simple_reset=True)
        self.train_progress2.add_meter(self.meter_total_loss_2_1, exclude_simple_reset=True)
        self.train_progress2.add_meter(self.meter_total_loss_2_2, exclude_simple_reset=True)
        self.train_progress2.add_meter(self.meter_total_loss_2_3, exclude_simple_reset=True)  
        self.train_progress2.set_num_batches(self.iters_per_epoch)
        
        return train_progress, val_progress
    
    @property
    def train_progress(self) -> ProgressMeter:
        if hasattr(self, 'phase') == False:
            return self._train_progress
        if self.phase == 1:
            return self.train_progress1
        elif self.phase == 2:
            return self.train_progress2
        else:
            return self._train_progress
    @train_progress.setter
    def train_progress(self, value: ProgressMeter) -> None:
        self._train_progress = value
    
    def _forward_train(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """Computes the forward pass based on the current phase and subphase."""
        if self.phase == 2 and self.subphase == '1':
            return self._forward_train_phase_2_1(data)
        else:
            # For phase 1 and all other subphases of phase 2 the forward pass is just a simple forward pass
            return super()._forward(data)
    
    def _forward_train_phase_2_1(self, data: Tuple[torch.Tensor, torch.Tensor]
                                 ) -> Tuple[Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE], torch.Tensor]:
        """For phase 2 (first part) the Virtual Adversarial Training (VAT) components and resulting
        predictions are computed. Note that in model.vat() some gradients and backwards passes are computed already.
        Note that only a single other forward pass is computed instead of original UJDA where
        each loss computation will call a forward pass again on the same data.
        This is both computationally expensive and not necessary as the data is the same even potentially
        causing overfitting or error accumulation."""
        data = torch.cat(data, dim=0)
        data_vat = self.model.vat(data)
        self._zero_grad() # Reset gradients as in .vat() gradients are computed
        # Can not simply concatenate the data as the auto_split between domains would wrongly
        # mix the domains. (I.e. all data becomes source and all data_vat becomes target)
        (pred_cls,_,_), f = self.model(data)
        (pred_vat,_,_), f = self.model(data_vat)
        return (pred_cls, pred_vat), f
    
    def _filter_loss_components(self,
                                pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor],
                                target: Tuple[torch.Tensor,torch.Tensor],
                                features: Optional[Sequence[torch.Tensor]]=None,
                                og_labels: Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
                                mode: Optional[str]=None,
                                ) -> Tuple[Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor], Tuple[torch.Tensor,torch.Tensor], Optional[Sequence[torch.Tensor]], Optional[Tuple[torch.Tensor,torch.Tensor]]]:
        """For now no direct support of filter_loss_components for UJDA model."""
        return pred, target, features, og_labels
        
    def _compute_loss_cls(self, 
                          pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor] | Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE], 
                          target: Tuple[torch.Tensor,torch.Tensor],) -> torch.Tensor:
        if self.phase == 1:
            pred_cls, _, _ = pred
            return super()._compute_loss_cls(pred=pred_cls, target=target)
        elif self.phase == 2 and self.subphase == '1':
            pred_cls, _ = pred
            return super()._compute_loss_cls(pred=pred_cls, target=target)
        else:
            return zero()
    
    def _compute_loss_logits(self, pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return super()._compute_loss_logits(pred[0])
    
    def _compute_loss_adaptation(self, **others) -> torch.Tensor:
        if self.phase == 1:
            return self._compute_loss_adaptation_phase1(**others)
        elif self.phase == 2 and self.subphase == '1':
            return self._compute_loss_adaptation_phase2_1(**others)
        elif self.phase == 2 and self.subphase == '2':
            return self._compute_loss_adaptation_phase2_2(**others)
        elif self.phase == 2 and self.subphase == '3':
            return self._compute_loss_adaptation_phase2_3(**others)
    
    def _compute_loss_adaptation_phase1(self, pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor], target: Tuple[torch.Tensor, torch.Tensor], **others) -> torch.Tensor:
        _, pred_joint_1, pred_joint_2 = pred
        joint_loss_1, joint_loss_2 = self.model.compute_joint_loss_labeled(pred_joint_1, pred_joint_2, target)
        self.meter_loss_joint_1.update(joint_loss_1.item(), self.batch_size)
        self.meter_loss_joint_2.update(joint_loss_2.item(), self.batch_size)
        return self.joint_loss_weight*(joint_loss_1 + joint_loss_2)
    
    def _compute_loss_adaptation_phase2_1(self, pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor], **others) -> torch.Tensor:
        pred_cls, pred_vat = pred
        if self.model.head_type.returns_multiple_outputs:
            pred_cls = pred_cls[self.model.classifier.test_head_pred_idx]
            pred_vat = pred_vat[self.model.classifier.test_head_pred_idx]
        vat_loss = self.model.compute_loss_vat(pred_cls, pred_vat)
        self.meter_vat_loss.update(vat_loss.item(), self.batch_size)
        entropy_loss = self.model.compute_loss_entropy(pred_cls)
        self.meter_entropy_loss.update(entropy_loss.item(), self.batch_size)
        return self.vat_loss_weight * vat_loss + self.entropy_loss_weight * entropy_loss
    
    def _compute_loss_adaptation_phase2_2(self, pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor], 
                                          target: Tuple[torch.Tensor, torch.Tensor], **others) -> torch.Tensor:
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
        return self.joint_cls_loss_labeled_weight*(joint_loss_1_l + joint_loss_2_l) + \
                self.joint_cls_loss_unlabeled_weight*(joint_loss_1_ul + joint_loss_2_ul) - \
                self.discrepancy_loss_weight * loss_discrepancy
    
    def _compute_loss_adaptation_phase2_3(self, pred: Tuple[TRAIN_PRED_TYPE, torch.Tensor, torch.Tensor], 
                                          target: Tuple[torch.Tensor, torch.Tensor], **others) -> torch.Tensor:
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
        
        return self.joint_adv_loss_weight * adv_joint_loss + self.discrepancy_loss_weight * loss_discrepancy
        
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
            
    def _train_computation(self, 
                           data: Tuple[torch.Tensor, torch.Tensor], 
                           labels: Tuple[torch.Tensor, torch.Tensor], 
                           accum_steps: int,
                           need_update: bool) -> None:
        if self.phase == 1:
            return super()._train_computation(data, labels, accum_steps, need_update)
        elif self.phase == 2:
            return self._train_computation_p2(data, labels, accum_steps, need_update)
        else:
            raise ValueError(f"Phase {self.phase} not supported")
            
    def _train_computation_p2(self, 
                              data: Tuple[torch.Tensor, torch.Tensor], 
                              labels: Tuple[torch.Tensor, torch.Tensor], 
                              accum_steps: int,
                              need_update: bool) -> None:
        """Runs the subphases for phase 2. The individual functions for the subphases are dynamically resolved
        based on `self.subphase`."""
        
        # Only evaluate once after all phases have run
        eval_train = self.eval_during_train
        self.eval_during_train = False
        ### Phase 2_1 ###
        self.subphase = "1"
        super()._train_computation(data, labels, accum_steps, need_update)
        
        ### Phase 2_2 ###
        self.subphase = "2"
        super()._train_computation(data, labels, accum_steps, need_update)
        
        ### Phase 2_3 ###
        # For the adversarial training we need to do all the steps for the specified number of times
        self.subphase = "3"
        # Separate the number of steps for phase 3 to allow for train eval in final iteration
        # and do not have an if condition in the loop
        for _ in range(self.num_steps_adv - 1):
            super()._train_computation(data, labels, accum_steps, need_update)
        # Reset eval_during_train to its original (i.e. allow eval during training)
        self.eval_during_train = eval_train
        super()._train_computation(data, labels, accum_steps, need_update)
            
    def train(self, epochs: Optional[int]=None, start_epoch: Optional[int] = None) -> None:
        if epochs is None:
            epochs = self.num_epochs
            
        if epochs is None:
            warnings.warn('No number of epochs specified. No training.')
            return
        
        if start_epoch is not None:
            self.epoch = start_epoch
        else:
            start_epoch = self.epoch
        
        start_time = time.time()     
        best_acc1 = 0.
        first_epoch = True
        metrics = OrderedDict()
        # start training
        best_acc1 = 0.
        print("Starting training for phase 1.")
        self.phase = 1
        for epoch in range(epochs):
            # Optimizers are set properly in _set_phase to only contain relevant ones for the phase
            lrl = [param_group['lr'] for opt in self.optimizers for param_group in opt.param_groups]
            print(f'Learning rate: {",".join([f"{i:.4e}" for i in lrl])}')
            # No need to pass epoch as we set it appropriately in right before
            metrics = self.process_epoch()
            acc1 = self.eval_acc_1
            save_checkpoint(model=self.model, 
                            logger=self.logger, 
                            optimizer=self,
                            metric='acc1', 
                            current_best=best_acc1,
                            save_val_test=True)
            
            best_acc1 = max(acc1, best_acc1)
            filewriter.update_summary(epoch=epoch+start_epoch, 
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
        self.phase = 2
        for epoch in range(epochs):
            lrl = [param_group['lr'] for opt in self.optimizers for param_group in opt.param_groups]
            print(f'Learning rate: {",".join([f"{i:.4e}" for i in lrl])}')
            # No need to pass epoch as we set it appropriately in right before
            metrics = self.process_epoch()
            acc1 = self.eval_acc_1
            save_checkpoint(model=self.model, 
                            logger=self.logger, 
                            optimizer=self,
                            metric='acc1', 
                            current_best=best_acc1,
                            save_val_test=True)
            
            best_acc1 = max(acc1, best_acc1)
            filewriter.update_summary(epoch=epoch, 
                                      metrics=metrics, 
                                      root=self.logger.out_dir,
                                      write_header=first_epoch)
        print("best_acc1 = {:3.1f}".format(best_acc1)) 
        print(f'Total time for training Phase 2: {time.time() - p1_time}')
        print(f'Total time: {time.time() - start_time}')