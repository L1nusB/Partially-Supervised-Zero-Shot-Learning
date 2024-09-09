from collections import OrderedDict

from typing import Iterable, Optional, Tuple, Sequence
import warnings


import torch

from torch.utils.data import DataLoader

from PSZS.Utils.meters import StatsMeter, ProgressMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Optimizer.Base_Multiple import Base_Multiple, TRAIN_PRED_TYPE
from PSZS.Models import BiLin_Domain_Adv_Model, METHOD_MODEL_MAP
from PSZS.Utils.io.logger import Logger

class BiLin_Multiple(Base_Multiple):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: BiLin_Domain_Adv_Model, 
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
        assert isinstance(self.model, METHOD_MODEL_MAP['bilinear']), \
            f"Bilin_Multiple model must be of type {METHOD_MODEL_MAP['bilinear'].__name__}"
        self.model : BiLin_Domain_Adv_Model
        self.source_pred_mask = self._construct_source_pred_mask()
        if self.model.bilin_domain_adversarial_loss.mode == 'simple' or self.model.bilin_domain_adversarial_loss.mode == 'classes':
            # Pass the mask to Bilinear Loss when separating between shared and novel or simple mode (i.e. no separation)
            self.model.bilin_domain_adversarial_loss.mask = self.source_pred_mask
        # Dynamically set the max_iters for the GRL based on epoch count if max_iters not given explicitly otherwise.
        # If grl_epochs is not given it will default to half the number of epochs
        # Can not be done during model construction as epochs and iters are not known
        if self.model.grl_max_iters is None:
            if grl_epochs is None:
                grl_epochs = self.num_epochs // 2
            self.model.bilin_domain_adversarial_loss.grl.max_iters = grl_epochs * iters_per_epoch
            
    def _construct_source_pred_mask(self) -> torch.Tensor:
        """Construct a mask to filter out the predictions of the source domain 
        that are not shared with the target domain.
        Filtering is done based on the dataset descriptor of the source domain.
        If no dataset descriptor is found, the mask will retain the first self.shared_classes entries.

        Returns:
            torch.Tensor: Mask to filter predictions of the source domain.
        """
        source_desc = self.train_iters[0].dataset_descriptor
        if source_desc is None:
            warnings.warn("No dataset descriptor found for source domain. "
                          f"Mask will retain first self.shared_classes [{len(self.shared_classes)}] entries.")
            mask = torch.zeros(self.train_iters[0].num_classes)
            mask[range(len(self.shared_classes))] = 1
        else:
            main_class_index = getattr(self.train_iters[0].dataset, 'main_class_index', -1)
            mapping = source_desc.predIndex_to_targetId[main_class_index]
            mask = [mapping[i] in self.shared_classes for i in range(0, source_desc.num_classes[main_class_index])]
        return torch.tensor(mask, dtype=torch.bool, device=self.device)
            
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
    
    def _compute_loss_adaptation(self, pred: TRAIN_PRED_TYPE, features: Sequence[torch.Tensor], **others) -> torch.Tensor:
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
        # For hierarchical models, each pred has multiple components
        # Thus we need to get only the relevant component 
        p_s, p_t = pred
        if self.model.classifier.returns_multiple_outputs:
            p_s = p_s[self.model.classifier.test_head_pred_idx]
            p_t = p_t[self.model.classifier.test_head_pred_idx]
        
        transfer_loss : torch.Tensor = self.model.bilin_domain_adversarial_loss(f_s=f_s, 
                                                                                norm_logits_s=p_s.softmax(dim=1),
                                                                                f_t=f_t,
                                                                                norm_logits_t=p_t.softmax(dim=1))
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