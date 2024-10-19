from collections import OrderedDict

from typing import Iterable, Optional, Tuple, Sequence
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PSZS.Utils.meters import DynamicStatsMeter, StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Optimizer import FEATURE_TYPE
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import PAN_Model, METHOD_MODEL_MAP
from PSZS.Models.funcs import entropy_multiple, weighted_accum
from PSZS.Utils.io.logger import Logger

class PAN_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: PAN_Model, 
                 iters_per_epoch: int, 
                 print_freq: int, 
                 batch_size: int, 
                 eval_classes: Iterable[int],
                 logger: Logger,
                 device: torch.device, 
                 domain_loss_weight: float = 1.0,
                 grl_epochs: Optional[int] = None,
                 entropy_loss_weights: Tuple[float, float] = [0., 0.01],
                 max_smoothing_epochs: Optional[int] = None,
                 progressive_weight: float = 1,
                 **optim_kwargs,
                 ) -> None:
        self.progressive_weight = progressive_weight
        self.entropy_loss_weights = entropy_loss_weights
        # While negative weights should not be used in practice, we allow it here
        # and thus need to check the absolute values
        self.use_entropy = sum(abs(w) for w in entropy_loss_weights) > 0
        if self.use_entropy == False:
            warnings.warn("Entropy loss weights are 0. No entropy loss will be computed.")
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
        self.domain_loss_weight = domain_loss_weight
        self.source_pred_mask = self._construct_source_pred_mask()
        assert isinstance(self.model, METHOD_MODEL_MAP['pan']), \
            f"PAN_Multiple model must be of type {METHOD_MODEL_MAP['pan'].__name__}"
        self.model : PAN_Model
        # Dynamically set the max_iters for the GRL based on epoch count (always as max_iters not supported by PAN_Model).
        # If grl_epochs is not given it will default to half the number of epochs
        # Can not be done during model construction as epochs and iters are not known
        if grl_epochs is None:
            grl_epochs = self.num_epochs // 2
        self.model.bilin_domain_adversarial_loss.grl.max_iters = grl_epochs * iters_per_epoch
        # Set the max smoothing steps to max_smoothing_epochs epochs if not specified otherwise
        if getattr(self.model, 'max_smoothing_steps', None) is None:
            if max_smoothing_epochs is None:
                max_smoothing_epochs = self.num_epochs // 2
            setattr(self.model, 'max_smoothing_steps', self.iters_per_epoch * max_smoothing_epochs)
            
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        self.meter_mixed_fine_cls_loss = DynamicStatsMeter.get_stats_meter_min_max("Mixed fine loss", 
                                                                fields=self.watched_fields_loss, 
                                                                fmt=":6.2f")
        self.meter_transfer_loss = StatsMeter.get_stats_meter_min_max('Transfer Loss', fmt=":3.2f",)
        self.meter_entropy_loss = StatsMeter.get_stats_meter_min_max('Entropy', fmt=":3.2f",)
        train_progress.add_meter(self.meter_transfer_loss, exclude_simple_reset=True)
        train_progress.add_meter(self.meter_mixed_fine_cls_loss, exclude_simple_reset=True)
        # Only get display/filled with values if eval_during_train is True
        if self.eval_during_train:
            self.meter_domain_acc = StatsMeter.get_stats_meter_min_max('Domain Acc', fmt=":3.2f",)
            train_progress.add_meter(self.meter_domain_acc, exclude_simple_reset=True)
        # Only add entropy meter if entropy is relevant (i.e. weight not 0)
        if self.use_entropy:
            train_progress.add_meter(self.meter_entropy_loss, exclude_simple_reset=True)
        return train_progress, val_progress
    
    def _compute_loss_cls(self, 
                          pred: TRAIN_PRED_TYPE, 
                          target: Tuple[torch.Tensor,torch.Tensor]) -> torch.Tensor:
        loss = super()._compute_loss_cls(pred=pred, target=target)
        
        smoothed_fine_labels = [self.model.smooth_labels(fine_target_one_hot=F.one_hot(t[:, self.model.classifier.test_head_pred_idx], 
                                                                                    desc.num_classes[self.model.classifier.test_head_pred_idx]),
                                                      coarse_logits=p[:self.model.classifier.test_head_pred_idx],
                                                      fine_to_coarse=desc.pred_fine_coarse_map) 
                                for t, p, desc in zip(target, pred, self.train_descriptors)]
        fine_loss_components = [F.kl_div(F.log_softmax(p[self.model.classifier.test_head_pred_idx], dim=1), l[0], reduction='batchmean') / b
                     for p,l, b in zip(pred, smoothed_fine_labels, self.train_batch_sizes)]
        fine_loss = sum(fine_loss_components)
        self.meter_mixed_fine_cls_loss.update(vals=[fine_loss] + fine_loss_components,
                            n=[self.batch_size] + [trg.size(0) for trg in target])
        return loss + self.progressive_weight * fine_loss
    
    def _compute_loss_adaptation(self, pred: TRAIN_PRED_TYPE, 
                                 features: Sequence[torch.Tensor], **others) -> torch.Tensor:
        """Computes the bilinear domain adaptation loss for PAN.
        The features are assumed to be filtered based on `adaptation_filter_mode`.
        This is handeled in the `_compute_loss` method of Base_Optimizer.
        
        Args:
            pred (TRAIN_PRED_TYPE): Unnormalized prediction logits to compute the PAN loss over.
            features (Sequence[torch.Tensor]): Features to compute the PAN loss over.

        Returns:
            torch.Tensor: Domain adaptation loss scaled by `domain_loss_weight`.
        """
        f_s, f_t = features
        # In case adaptation mode filters the filters only account for actual features
        f_count = sum([f_i.size(0) for f_i in features])
        # For hierarchical models, each pred has multiple components
        # Thus we need to get only the relevant component 
        p_s, p_t = pred
        if self.model.classifier.returns_multiple_outputs:
            p_s = p_s[self.model.classifier.test_head_pred_idx]
            p_t = p_t[self.model.classifier.test_head_pred_idx]
        # Filter out the predictions that are not shared with the target domain
        # No need to change target domain predictions p_t
        p_s = p_s[:, self.source_pred_mask]
        transfer_loss : torch.Tensor = self.model.bilin_domain_adversarial_loss(f_s=f_s, 
                                                                                norm_logits_s=p_s.softmax(dim=1),
                                                                                f_t=f_t,
                                                                                norm_logits_t=p_t.softmax(dim=1))
        self.meter_transfer_loss.update(transfer_loss.item(), f_count)
        return self.domain_loss_weight * transfer_loss
    
    def _compute_loss_features(self, features: Sequence[FEATURE_TYPE],
                               target: Tuple[torch.Tensor,torch.Tensor]) -> torch.Tensor:
        """Always computes the entropy loss for PAN weighted by `entropy_loss_weights`.
        If other feature losses are specified, they will be computed as well and added.
        
        .. note::
            No check if entropy is computed twice is performed here.
            So do not specify entropy as a feature loss function separately for PAN."""
        if self.use_entropy:
            entropy_loss = weighted_accum(entropy_multiple([f.softmax(dim=1) for f in features]),
                                        weights=self.entropy_loss_weights)
        else:
            entropy_loss = 0.
        self.meter_entropy_loss.update(entropy_loss.item(), sum([f_i.size(0) for f_i in features]))
        return entropy_loss + super()._compute_loss_features(features=features, target=target)
            
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
        domain_acc = self.model.bilin_domain_adversarial_loss.discriminator_accuracy
        self.meter_domain_acc.update(domain_acc, self.batch_size)
        
    def _get_train_results(self) -> OrderedDict:
        results = super()._get_train_results()
        results.update([(f'Transfer_Loss', self.meter_transfer_loss.get_avg())])
        # Values/Meters are only updated if eval_during_train is True
        if self.eval_during_train:
            results.update([(f'Domain_Acc', self.meter_domain_acc.get_avg())])
        if self.use_entropy:
            results.update([(f'Entropy', self.meter_entropy_loss.get_avg())])
        return results