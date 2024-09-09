from collections import OrderedDict

import time
from typing import Iterable, List, Optional, Tuple, Sequence
import warnings


import torch

from torch.utils.data import DataLoader

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import DASA_Model, METHOD_MODEL_MAP
from PSZS.Utils.io.logger import Logger

class DASA_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: DASA_Model, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 domain_loss_weight: float = 1.0,
                 class_align_weight: float = 0.1,
                 grl_epochs: Optional[int] = None,
                 domain_update_freq: int = 1,
                 **optim_kwargs,
                 ) -> None:
        if domain_loss_weight == 0:
            warnings.warn("domain_loss_weight 0 given.")
        if class_align_weight == 0:
            warnings.warn("class_align_weight 0 given.")
        self.domain_loss_weight = domain_loss_weight
        self.class_align_weight = class_align_weight
        # In the original code they only update the discriminator every 9 steps
        # but these seems a bit odd (and no justification or mention in original paper)
        self.domain_update_freq = domain_update_freq
        self.update_domain_discriminator = False # Is set during training loop based on current iteration
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
        assert isinstance(self.model, METHOD_MODEL_MAP['dasa']), \
            f"DASA_Multiple model must be of type {METHOD_MODEL_MAP['dasa'].__name__}"
        self.model : DASA_Model
        # Dynamically set the max_iters for the GRL based on epoch count if max_iters not given explicitly otherwise.
        # If grl_epochs is not given it will default to half the number of epochs
        # Can not be done during model construction as epochs and iters are not known
        if self.model.grl_max_iters is None:
            if grl_epochs is None:
                grl_epochs = self.num_epochs // 2
            # Take into account the domain update frequency
            self.model.domain_adversarial_loss.grl.max_iters = grl_epochs * iters_per_epoch / self.domain_update_freq
            
    @property
    def optimizers(self) -> List[torch.optim.Optimizer]:
        """Return optimizer for backbone + classifier and 
        domain discriminator based on `update_domain_discriminator`."""
        if self.update_domain_discriminator:
            return super().optimizers
        else:
            return [super().optimizers[0]]
        
    # Note that we do not restrict the lr_schedulers intentionally
    # learning rate for the domain discriminator should update at the same rate as the backbone
        
    def _update_params(self, loss:torch.Tensor, need_update:bool) -> None:
        """Updates the parameters of the model and optimizers.
        The gradients of all optimizers (including domain discriminator) are reset after the update."""
        super()._update_params(loss, need_update)
        # Reset all gradients if an update was performed
        # The super call only affects the currently active optimizers
        if need_update:
            self._zero_grad()
            
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
    
    def _compute_loss_adaptation(self, features: Sequence[torch.Tensor], 
                                 target: Tuple[torch.Tensor, torch.Tensor], **others) -> torch.Tensor:
        """Computes the domain adaptation loss for DASA.
        The domain adaptation loss consists of the binary domain discriminator as well as the class alignment loss.
        The features are assumed to be filtered based on `adaptation_filter_mode`.
        This is handeled in the `_compute_loss` method of Base_Optimizer.
        
        .. note::
            The `others` argument is required as Base_Optimizer will pass all arguments explicitly
            but they are not needed here.

        Args:
            features (Sequence[torch.Tensor]): Features to compute the DASA loss over.
            target (Tuple[torch.Tensor, torch.Tensor]): Target labels to compute the DASA loss over.

        Returns:
            torch.Tensor: Domain adaptation loss scaled by `domain_loss_weight` and `class_align_weight`.
        """
        f_s, f_t = features
        # In case adaptation mode filters the filters only account for actual features
        f_count = sum([f_i.size(0) for f_i in features])
        domain_transfer_loss : torch.Tensor = self.model.domain_adversarial_loss(f_s=f_s, f_t=f_t)
        
        alignment_label_s, alignment_label_t = target
        if self.model.classifier.returns_multiple_outputs:
            alignment_label_s = alignment_label_s[:,self.model.classifier.test_head_pred_idx]
            alignment_label_t = alignment_label_t[:,self.model.classifier.test_head_pred_idx]
        class_alignment_loss : torch.Tensor = self.model.class_aligner(feat1=f_s, feat2=f_t,
                                                        label1=alignment_label_s, 
                                                        label2=alignment_label_t)
        
        self.meter_transfer_loss.update(domain_transfer_loss.item() + class_alignment_loss.item(), f_count)
        return self.domain_loss_weight * domain_transfer_loss + self.class_align_weight * class_alignment_loss
            
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
    
    # We need to overwrite this as we have to update whether the domain discriminator should be updated
    # which requires access to the current iteration
    ### Only modification to Base_Optimizer is the update of self.update_domain_discriminator ###
    def _train_epoch(self, epoch:Optional[int]=None) -> OrderedDict:
        self.reset_train()
        self._check_modify_mixup()
        epoch_start_time = time.time()
        if epoch:
            self.epoch = epoch
        print(f'Start Training epoch: {self.epoch}')
        print('-'*40)
        self.progress_bar_train.prefix = f"Epoch: [{self.epoch}]"
        
        accum_steps = self.grad_accum_steps
        # Number of accumulations for last batches 
        # Can be different from other accumulations which would cause wrong loss scaling
        last_accum_steps = self.iters_per_epoch % accum_steps
        last_batch_id = self.iters_per_epoch - 1
        last_batch_id_to_accum = self.iters_per_epoch - last_accum_steps
        
        # Switch to train mode
        self.model.train()
        
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        
        end = time.time()
        self.last_time_mark = end
        ### Do Training
        for i in range(self.iters_per_epoch):
            ### ONLY MODIFICATION ###
            self.update_domain_discriminator = (i % self.domain_update_freq == 0)
            ###
            
            # Update required if last iter or accum steps reached
            need_update = (i+1) % accum_steps == 0 or i==last_batch_id
            # Adjust number of accumulation steps on final accumulation
            # as it can be different from self.grad_accum_steps
            if i >= last_batch_id_to_accum:
                accum_steps = last_accum_steps
            
            # Load Data
            data, labels = self._load_data_train()
            # Forward pass, Compute Loss and Param update
            self._train_computation(data=data, 
                                    labels=labels, 
                                    accum_steps=accum_steps, 
                                    need_update=need_update)
            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()
            
            if i % self.print_freq == 0:
                self.progress_bar_train.display(i)
                self.reset_batch_meters(train=True)
        ### End Training
        # Display last results
        self.progress_bar_train.display(self.iters_per_epoch)
        # Epoch gets increased externally (e.g. in process_epoch)
        print(f'Train Epoch took: {time.time() - epoch_start_time}')
        
        return self._get_train_results()