from collections import OrderedDict

import time
from typing import Iterable, List, Optional, Tuple, Sequence
import warnings

import torch

from timm.scheduler.scheduler import Scheduler

from torch.utils.data import DataLoader

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import ADDA_Model, METHOD_MODEL_MAP, save_checkpoint
from PSZS.Utils.io.logger import Logger
from PSZS.Utils.io import filewriter

class ADDA_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: ADDA_Model, 
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
        # No point in using ADDA if domain_loss_weight is 0 (use ERM instead)
        if domain_loss_weight == 0:
            raise ValueError("Specify a domain_loss_weight != 0 for ADDA.")
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
            **optim_kwargs,
        )
        assert isinstance(self.model, METHOD_MODEL_MAP['adda']), \
            f"ADDA_Multiple model must be of type {METHOD_MODEL_MAP['adda'].__name__}"
        self.model : ADDA_Model
        # Dynamically set the max_iters for the GRL based on epoch count if max_iters not given explicitly otherwise.
        # If grl_epochs is not given it will default to half the number of epochs
        # Can not be done during model construction as epochs and iters are not known
        if self.model.grl_max_iters is None:
            if grl_epochs is None:
                grl_epochs = self.num_epochs // 2
            self.model.domain_adversarial_loss.grl.max_iters = grl_epochs * iters_per_epoch
        self.phase = "domain" if self.model.skip_cls_train else "cls"
        self.skipped_cls_train = self.model.skip_cls_train
        
    @property
    def optimizers(self) -> List[torch.optim.Optimizer]:
        """Return only first or only second optimizer depending on the phase.
        If no matching phase return all _optimizers."""
        # If called before constructer is finished (i.e. during super().__init__) return all optimizers
        # If skip_cls_train is set, only a single optimizer exists and thus indexing would fail
        # (see _create_optimizers in ADDA_Model.py)
        if hasattr(self, 'phase') == False or self.model.skip_cls_train:
            return super().optimizers
        if self.phase == 'cls':
            return [super().optimizers[0]]
        elif self.phase == 'domain':
            return [super().optimizers[1]]
        else:
            return super().optimizers
        
    @property
    def lr_schedulers(self) -> List[Scheduler]:
        """Return only first or only second scheduler depending on the phase.
        If no matching phase return all _lr_schedulers."""
        # If skipped_cls_train is set, only a single scheduler exists and thus indexing would fail
        if self.phase == 'cls' or self.skipped_cls_train:
            return [super().lr_schedulers[0]]
        elif self.phase == 'domain':
            return [super().lr_schedulers[1]]
        else:
            return super().lr_schedulers
        
    @property
    def phase(self) -> str:
        return self._phase
    @phase.setter
    def phase(self, phase: str) -> None:
        assert phase in ['cls', 'domain'], f"Phase {phase} not supported"
        self._phase = phase
        self._set_progress_bar(phase)
            
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        """Create the meters for the transfer loss and domain accuracy for the domain adaptation phase.
        Setting of the meters is done in _set_progress_bar."""
        self.meter_transfer_loss = StatsMeter.get_stats_meter_min_max('Transfer Loss', fmt=":3.2f",)
        self.meter_domain_acc = StatsMeter.get_stats_meter_min_max('Domain Acc', fmt=":3.2f",)
        
        # Remove total loss meter as we always only have classification or domain adaptation loss
        # and since we overwrite the _compute_loss method, total loss is not updated.
        train_progress.remove_meter(self.meter_total_loss)
        
        return train_progress, val_progress
    
    def _set_progress_bar(self, phase: Optional[str]=None) -> None:
        if phase is None:
            phase = self.phase
        print(f"Setting progress bar for phase: {phase}")
        if phase == 'cls':
            self.progress_bar_train.add_meter(self.meter_cls_loss, exclude_simple_reset=True)
            # Remove the domain transfer meters as it they are not used in this phase
            self.progress_bar_train.remove_meter(self.meter_transfer_loss)
            self.progress_bar_train.remove_meter(self.meter_domain_acc)
        elif phase == 'domain':
            self.progress_bar_train.add_meter(self.meter_transfer_loss, exclude_simple_reset=True)
            self.progress_bar_train.add_meter(self.meter_domain_acc, exclude_simple_reset=True)
            # Remove the classifier loss meter as it is not used in this phase
            self.progress_bar_train.remove_meter(self.meter_cls_loss)
        else:
            raise ValueError(f"Phase {phase} not supported")
        
    def _compute_loss(self, 
                      pred: TRAIN_PRED_TYPE, 
                      target: Tuple[torch.Tensor,...], 
                      features: Optional[Sequence[torch.Tensor]]=None, 
                      og_labels: Optional[Tuple[torch.Tensor,...]]=None
                      ) -> torch.Tensor:
        """Compute the total loss for the model.
        Depending on `self.phase` the loss is computed differently.
            - For `phase='cls'` the loss is computed as usual i.e. classification loss.
            - For `phase='domain'` only the domain adaptation loss is computed.
        
        Assumes that `target` is already mapped to the correct internal class indices as `self._map_labels` 
        is called in `~_train_computation`.
        
        Loss scaling based on `self.scale_loss_accum` and `self.grad_accum_steps` is handeled in `_train_computation`."""
        if self.phase == 'cls':
            return self._compute_loss_cls(pred=pred, target=target)
        else:
            # No need to check phase='domain' as this is the only other option#
            _, _, features, _ = self._filter_loss_components(pred=pred, target=target, 
                                                            features=features, og_labels=og_labels)
            return self._compute_loss_adaptation(features=features)
    
    def _compute_loss_adaptation(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Computes the domain adaptation loss for ADDA. This is the same as for DANN.
        The features are assumed to be filtered based on `adaptation_filter_mode`.
        This is handeled in the `_compute_loss` method.
        .. note::
            No `**others` argument is required as we overwrite the `_compute_loss` method of Base_Optimizer.
            (as compared to DANN where we only overwrite `_compute_loss_adaptation`)

        Args:
            features (Sequence[torch.Tensor]): Features to compute the DANN loss over.

        Returns:
            torch.Tensor: Domain adaptation loss scaled by `domain_loss_weight`.
        """
        f_s, f_t = features
        transfer_loss : torch.Tensor = self.model.domain_adversarial_loss(f_s=f_s, f_t=f_t)
        self.meter_transfer_loss.update(transfer_loss.item(), self.batch_size)
        return self.domain_loss_weight * transfer_loss
            
    def _compute_eval_metrics_train(self, 
                                    pred: TRAIN_PRED_TYPE, 
                                    target: Tuple[torch.Tensor,...], 
                                    og_labels: Tuple[torch.Tensor,...], 
                                    features: Optional[Sequence[torch.Tensor]]=None,
                                    ) -> None:
        """Computes the evaluation metrics during the training phase.
        During `phase='cls'` the metrics are computed as usual.
        For `phase='domain'` the domain discriminator accuracy is computed in addition
        and stored in `meter_domain_acc`."""
        super()._compute_eval_metrics_train(pred=pred, 
                                            target=target, 
                                            og_labels=og_labels,
                                            features=features,)
        if self.phase == 'domain':
            domain_acc = self.model.domain_adversarial_loss.domain_discriminator_accuracy
            self.meter_domain_acc.update(domain_acc, self.batch_size)
        
    def _get_train_results(self) -> OrderedDict:
        results = super()._get_train_results()
        results.update([(f'Transfer_Loss', self.meter_transfer_loss.get_avg()),
                        (f'Domain_Acc', self.meter_domain_acc.get_avg())])
        return results
        
    def train(self, epochs: Optional[int]=None, start_epoch: Optional[int] = None) -> None:
        if self.model.skip_cls_train:
            # Skip classifier training and directly start domain adversarial phase
            # progress bar is set during the property setter of self.phase
            if self.phase != 'domain':
                self.phase = 'domain'
            # Since _compute_loss and _compute_eval_metrics_train are overwritten
            # the 2nd phase of ADDA behaves "as normal" and is handled by the general training loop.
            print("Skipping classifier training and directly starting domain adversarial phase")
            return super().train(epochs, start_epoch)
        
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
        print("Starting training for classification phase.")
        # self.phase is should already set to 'cls' in the constructor in this setting
        # but check again to be sure
        if self.phase != 'cls':
                self.phase = 'cls'
        for _ in range(epochs):
            # Consider lr for optimizer for classification phase
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
                            save_val_test=False)
            
            best_acc1 = max(acc1, best_acc1)
            filewriter.update_summary(epoch=self.epoch, 
                                      metrics=metrics, 
                                      root=self.logger.out_dir,
                                      write_header=first_epoch)
            first_epoch = False
        print("best_acc1 = {:3.1f}".format(best_acc1)) 
        cls_time = time.time()
        print(f'Total time for training classifier (Phase 1): {cls_time - start_time}')
        
        print("Starting training for domain adaptation phase.")
        # Set epoch to 1 to display the domain adaptation phase correctly
        self.epoch = 1
        self.phase = 'domain'
        for _ in range(epochs):
            # Consider lr for optimizer for domain adaptation phase
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
                            save_val_test=False)
            
            best_acc1 = max(acc1, best_acc1)
            filewriter.update_summary(epoch=self.epoch, 
                                      metrics=metrics, 
                                      root=self.logger.out_dir,
                                      write_header=first_epoch)
        print("best_acc1 = {:3.1f}".format(best_acc1)) 
        print(f'Total time for training domain discriminator (Phase 2): {time.time() - cls_time}')
        print(f'Total time: {time.time() - start_time}')