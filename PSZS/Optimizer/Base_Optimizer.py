from contextlib import suppress
import inspect
import time

from typing import Callable, Iterable, List, Optional, Tuple, Sequence, TypeAlias, overload
from collections import OrderedDict
import warnings

import torch

from timm.scheduler.scheduler import Scheduler

from PSZS.datasets import DatasetDescriptor

from torch.utils.data import DataLoader

from PSZS.Utils.meters import StatsMeter, ProgressMeter, _BaseMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Models import CustomModel, save_checkpoint
from PSZS.Utils.io import filewriter, Logger
from PSZS.Metrics import ConfusionMatrix
from PSZS.Utils.utils import NativeScalerMultiple

DAT_TYPE: TypeAlias = torch.Tensor | Tuple[torch.Tensor, ...]
TRAIN_PRED_TYPE: TypeAlias = Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]]
PRED_TYPE: TypeAlias = torch.Tensor | TRAIN_PRED_TYPE
# FEATURE_TYPE: TypeAlias = torch.Tensor | Sequence[torch.Tensor]
FEATURE_TYPE: TypeAlias = torch.Tensor
LABEL_TYPE: TypeAlias = torch.Tensor |Tuple[torch.Tensor,...]
# LABEL_TYPE: TypeAlias = torch.Tensor | Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]]

class Base_Optimizer():
    def __init__(self, 
                 train_iters: ForeverDataIterator | Sequence[ForeverDataIterator], 
                 val_loader: DataLoader, 
                 model: CustomModel, 
                 iters_per_epoch: int, 
                 batch_size: int, 
                 eval_classes: Sequence[int],
                 logger: Logger,
                 device: torch.device, 
                 print_freq: int = 100, 
                 send_to_device: bool = True,
                 eval_during_train: bool = False, 
                 eval_metrics: Optional[Sequence[str]] = None,
                 grad_accum_steps: int = 1,
                 mixup_off_epoch: int = 0,
                 mixup_fn: Optional[Callable | Sequence[Callable]] = None,
                 loss_scaler: Optional[NativeScalerMultiple] = None,
                 amp_autocast = suppress,
                 scale_loss_accum: bool = True,
                 eval_groups_names: Optional[Sequence[str]] = None,
                 additional_eval_group_classes: Optional[Sequence[int] | Sequence[Sequence[int]]] = None,
                 num_epochs: Optional[int] = None,
                 adaptation_filter_mode: str = 'ignore',
                 create_class_summary: bool = False,
                 feature_loss_weight: Optional[float] = None,
                 logit_loss_weight: Optional[float] = None,
                 max_mixing_epochs: int | Sequence[int] = 10,
                 ) -> None:
        self.train_iters = train_iters
        # Only allow a single validator object
        self.val_loader = val_loader
        # Iterator is set during validate() and used as default in _load_data_val()
        # If no loader is passed to validate() it gets constructed from self.val_loader
        self.val_iter = None
        self.epoch_length_val = len(val_loader)
        self.device = device
        self.model = model
        self.iters_per_epoch = iters_per_epoch
        self.print_freq = print_freq
        self.epoch = 1
        self.batch_size = batch_size
        self.train_batch_sizes = self._get_train_batch_sizes()
        self.single_eval_class = additional_eval_group_classes is None
        # Further group names are added subsequently/or overwritten if necessary
        self.eval_groups_names = ['novel']
        # Ensure uniform datatype for eval_classes
        if self.single_eval_class:
            self.eval_classes:Sequence[Sequence[int]] = [eval_classes]
        else:
            # additional_eval_group_classes is already not None
            if isinstance(additional_eval_group_classes[0], int):
                self.eval_classes:Sequence[Sequence[int]] = [eval_classes, additional_eval_group_classes]
            else:
                self.eval_classes:Sequence[Sequence[int]] = [eval_classes] + additional_eval_group_classes
            
            if eval_groups_names is None:
                self.eval_groups_names = self.eval_groups_names + [f'eval{i}' for i in range(len(eval_classes)-1)]
            else:
                # Note that if single_eval_class is True the names are ignored
                assert self.single_eval_class or len(self.eval_classes) == len(eval_groups_names), \
                        f"Number of eval classes ({len(self.eval_classes)}) must match the number of eval groups names ({len(eval_groups_names)})."
                self.eval_groups_names = eval_groups_names
            
        self.logger = logger
        self.send_to_device = send_to_device
        self.eval_during_train = eval_during_train
        self.grad_accum_steps = grad_accum_steps
        self.mixup_off_epoch = mixup_off_epoch
        # If a single mixup function is given use it for all iterators
        # if None is specified no mixup is used and None itself is stored.
        self.mixup_fns = [mixup_fn for _ in range(len(train_iters))] if isinstance(mixup_fn, Callable) else mixup_fn
        self.scale_loss_accum = scale_loss_accum
        self.loss_scaler = loss_scaler
        self.amp_autocast = amp_autocast
        self.num_epochs = num_epochs
        self.adaptation_filter_mode = adaptation_filter_mode
        self.create_class_summary = create_class_summary
        if feature_loss_weight is None:
            self.feature_loss_weight = self.model.default_feature_loss_weight
        else:
            self.feature_loss_weight = feature_loss_weight
        if logit_loss_weight is None:
            self.logit_loss_weight = self.model.default_logit_loss_weight
        else:
            self.logit_loss_weight = logit_loss_weight
        
        self.num_inputs = self.model.classifier.num_inputs
        self.num_classes = self.model.classifier.num_classes
        
        self.second_order = any([hasattr(optimizer, 'is_second_order') and optimizer.is_second_order 
                                 for optimizer in self.optimizers])
        if self.second_order and not all([hasattr(optimizer, 'is_second_order') and optimizer.is_second_order 
                                          for optimizer in self.optimizers]):
            warnings.warn('Some optimizers are second order but not all. This can lead to unexpected behaviour.')
        # Set eval_metrics (to lower to avoid case sensitivity)
        self.eval_metrics = [metric.lower() for metric in eval_metrics] if eval_metrics else []
        
        self._construct_default_meters()
        self.last_time_mark = time.time()
        self.progress_bar_train, self.progress_bar_val = self._build_progress_bars()
        
        if self.val_descriptor is not None:
            class_names = self.val_descriptor.classes[getattr(self.val_loader.dataset, 'main_class_index', -1)]
        else:
            class_names = None
        self.confusion_matrix = ConfusionMatrix(num_classes=self.model.classifier.effective_head_pred_size.max(),
                                                class_names=class_names)
        
        # Set the max mixing steps to max_mixing_epochs epochs if not specified otherwise
        # Use set attribute as it is not set in all strategies or in non-hierarchicals
        # in theory one could also check for hierarchy and what strategy is used
        # but just set it as there are no real additional costs
        if getattr(self.model, 'max_mixing_steps', None) is None:
            # No need to use a Parameter here as we already have the device and it is not optimized either way
            if isinstance(max_mixing_epochs, int):
                maxing_steps = torch.tensor([self.iters_per_epoch * max_mixing_epochs]*self.model.num_pred, 
                                               device=self.device, dtype=int, requires_grad=False)
            else:
                assert len(max_mixing_epochs) == self.model.num_pred, f'Number of max mixing epochs ({len(max_mixing_epochs)}) must match the number of prediction heads ({self.model.num_pred}).'
                maxing_steps = torch.tensor([self.iters_per_epoch * max_mixing_epochs[i] for i in range(self.model.num_pred)], 
                                               device=self.device,dtype=int, requires_grad=False)
            setattr(self.model, 'max_mixing_steps', maxing_steps)
        
    @classmethod
    def get_optim_kwargs(cls, **kwargs) -> dict:
        """
        Dynamically resolve relevant kwargs for optimizer construction.
        """
        dataset_kwargs = {}
        # Get the kwargs from the base classes
        # getmro returns a tuple with the class itself as first element and object as last
        for base in inspect.getmro(cls)[1:-1]:
            dataset_kwargs.update(base.get_optim_kwargs(**kwargs))
        # Using the inspect module to get the arguments of the constructor
        # This way we can filter out the relevant arguments without hardcoding them
        # and thus no need to update this method when the constructor changes
        # or in subclasses
        arguments = inspect.getfullargspec(cls.__init__).args
        # Only add if given otherwise use default from constructor
        for arg in arguments:
            # Also check for arg with _ replaced by -
            if arg in kwargs or arg.replace("_", "-") in kwargs:
                dataset_kwargs[arg] = kwargs[arg]
        return dataset_kwargs
    
    @property
    def cls_acc_1(self) -> Sequence[float]:
        return [meter.get_avg() for meter in self.meters_cls_acc_1]
    @property
    def cls_acc_5(self) -> Sequence[float]:
        return [meter.get_avg() for meter in self.meters_cls_acc_5]
    @property
    def precision(self) -> Sequence[float]:
        return [meter.get_last() for meter in self.meters_precision]
    @property
    def recall(self) -> Sequence[float]:
        return [meter.get_last() for meter in self.meters_recall]
    @property
    def f1(self) -> Sequence[float]:
        return [meter.get_last() for meter in self.meters_f1]
    
    @property
    def eval_acc_1(self) -> Sequence[float]:
        return self.cls_acc_1[0]
    @property
    def eval_acc_5(self) -> Sequence[float]:
        return self.cls_acc_5[0]
    @property
    def eval_precision(self) -> Sequence[float]:
        return self.precision[0]
    @property
    def eval_recall(self) -> Sequence[float]:
        return self.recall[0]
    @property
    def eval_f1(self) -> Sequence[float]:
        return self.f1[0]
        
    @property
    def optimizers(self) -> List[torch.optim.Optimizer]:
        """Return the optimizers of the model.
        Indirect access to model.optimizers to allow for overwriting in derived classes.
        E.g. only returning a single optimizer in case of ADDA.
        """
        return self.model.optimizers
    
    @property
    def lr_schedulers(self) -> List[Scheduler]:
        """Return the lr_schedulers of the model.
        Indirect access to model.lr_schedulers to allow for overwriting in derived classes.
        E.g. only returning a single scheduler in case of ADDA.
        """
        return self.model.lr_schedulers
    
    @property
    def train_descriptors(self) -> Optional[DatasetDescriptor] | List[Optional[DatasetDescriptor]]:
        if isinstance(self.train_iters, ForeverDataIterator):
            return self.train_iters.dataset_descriptor
        else:
            return [it.dataset_descriptor for it in self.train_iters]
    
    @property
    def val_descriptor(self) -> Optional[DatasetDescriptor]:
        return getattr(self.val_loader.dataset, 'descriptor', None)
    
    @property
    def has_feature_loss(self) -> bool:
        return self.model.feature_loss_func is not None and self.feature_loss_weight > 0
    
    @property
    def has_logit_loss(self) -> bool:
        return self.model.logit_loss_func is not None and self.logit_loss_weight > 0
    
    @property
    def num_eval_groups(self) -> int:
        return len(self.eval_classes)
    
    @property
    def multiple_outputs(self) -> bool:
        return self.model.classifier.returns_multiple_outputs
    
    @property
    def num_head_pred(self) -> int:
        return self.model.num_pred
    
    @property
    def uses_mixup(self) -> bool:
        """Determines if mixup is used in this optimizer. Different from `apply_mixup` as this 
        checks if mixup is used at all and not whether to apply it now.
        """
        if self.send_to_device:
            return self.mixup_fns is not None
        else:
            if isinstance(self.train_iters, ForeverDataIterator):
                return self.train_iters.data_loader.mixup_enabled
            else:
                return any([it.data_loader.mixup_enabled for it in self.train_iters])
    @property
    def do_mixup(self) -> bool:
        """Determines if mixup should be applied for the current batch.
        Can only be true when sending to the device here i.e. not using prefetch loader
        as these apply mixup internally already.
        """
        if self.send_to_device:
            return self.mixup_fns is not None
        else:
            return False
    
    def _get_train_batch_sizes(self) -> int | List[int]:
        if isinstance(self.train_iters, ForeverDataIterator):
            return self.train_iters.batch_size
        else:
            return [it.batch_size for it in self.train_iters]
        
    def _check_modify_mixup(self) -> None:
        """Check if mixup should be turned off and modify the mixup_fn accordingly.
        Currently mixup can not be enabled using this method.
        """
        # Turn off mixup if specified (i.e. not 0) 
        if self.mixup_off_epoch and self.epoch > self.mixup_off_epoch:
            # Only log once 
            if self.epoch-1 == self.mixup_off_epoch:
                print('Turning off mixup (if enabled)')
            # Prefetch Loader is used
            if not self.send_to_device:
                if isinstance(self.train_iters, ForeverDataIterator):
                    self.train_iters.data_loader.mixup_enabled = False
                else:
                    for it in self.train_iters:
                        it.data_loader.mixup_enabled = False
            # Disable Mixup function on non-prefetch loader
            elif self.mixup_fns is not None:
                for mixup_fn in self.mixup_fns:
                    mixup_fn.mixup_enabled = False
    
    ######### Metric meter functions #########      
    def _construct_default_meters(self) -> None:
        # Separate function to clean up the __init__ method
        self.batch_time = StatsMeter.get_stats_meter_time('Batch Time', ':5.2f')
        self.data_time = StatsMeter.get_stats_meter_time('Data Time', ':5.2f')
        if self.single_eval_class:
            self.meters_cls_acc_1 = [StatsMeter.get_stats_meter_min_max('Acc@1', ':3.2f')]
            self.meters_cls_acc_5 = [StatsMeter.get_stats_meter_min_max('Acc@5', ':3.2f')]
            self.meters_precision = [StatsMeter.get_average_meter('Precision', ':3.2f', include_last=True)]
            self.meters_recall = [StatsMeter.get_average_meter('Recall', ':3.2f', include_last=True)]
            self.meters_f1 = [StatsMeter.get_average_meter('F1', ':3.2f', include_last=True)]
        else:
            self.meters_cls_acc_1 = [StatsMeter.get_stats_meter_min_max(f'Acc@1({name})', ':3.2f') 
                                     for name in self.eval_groups_names]
            self.meters_cls_acc_5 = [StatsMeter.get_stats_meter_min_max(f'Acc@5({name})', ':3.2f') 
                                     for name in self.eval_groups_names]
            self.meters_precision = [StatsMeter.get_average_meter(f'Precision({name})', ':3.2f', include_last=True) 
                                     for name in self.eval_groups_names]
            self.meters_recall = [StatsMeter.get_average_meter(f'Recall({name})', ':3.2f', include_last=True) 
                                  for name in self.eval_groups_names]
            self.meters_f1 = [StatsMeter.get_average_meter(f'F1({name})', ':3.2f', include_last=True) 
                              for name in self.eval_groups_names]
        
        self.meter_feature_loss = StatsMeter.get_stats_meter_min_max('Feature Loss', fmt=":3.2f",)
        self.meter_logit_loss = StatsMeter.get_stats_meter_min_max('Logit Loss', fmt=":3.2f",)
        self.meter_total_loss = StatsMeter.get_stats_meter_min_max('Total Loss', fmt=":3.2f",)
    
    def _get_metric_meters(self) -> List[_BaseMeter]:
        """Gets all meter objects for the evaluation metrics.
        Can be called during `~_build_progress_bars` for simple access to all metric meters.
        .. note::
            No direct usage in Base_Optimizer as this is only a helper function for derived classes.
            See `Base_Multiple.py` for an example of usage."""
        meters = self.meters_cls_acc_1
        if 'acc@5' in self.eval_metrics:
            meters.extend(self.meters_cls_acc_5)
        if 'precision' in self.eval_metrics:
            meters.extend(self.meters_precision)
        if 'recall' in self.eval_metrics:
            meters.extend(self.meters_recall)
        # Only added later for better display order
        meters.extend(self.meters_f1)
        return meters
    
    ######### Progress Bars #########  
    def _build_progress_bars(self) -> Tuple[ProgressMeter, ProgressMeter]:
        raise NotImplementedError
    
    def _reset_progress_bars(self, 
                             reset_train:bool = True, 
                             reset_val:bool = False, 
                             reset_all:bool = True) -> None:
        if reset_train:
            self.progress_bar_train.reset(reset_all)
        if reset_val:
            self.progress_bar_val.reset(reset_all)
            
    def reset_batch_meters(self, train: bool = False, val: bool = False) -> None:
        """Resets the batch meters of the progress bar for training and/or validation.
        This resets the 'batch' field of this meter and is done after each display of the progress bar.
        Typically the batch meters are the time meters that contain information that only holds for the current batch."""
        if train:
            self._reset_progress_bars(reset_train=True, reset_val=False, reset_all=False)
        if val:
            self._reset_progress_bars(reset_train=False, reset_val=True, reset_all=False)
            
    def reset_train(self):
        """Resets the progress bars and meters for training.
        This reset is done at the beginning of each epoch
        and resets all meters irrespecive of `exclude_simple_reset`"""
        self._reset_progress_bars(reset_train=True, reset_val=False, reset_all=True)
        
    def reset_val(self):
        """Resets the progress bars and meters for validation.
        This reset is done at the start of a validation run
        and resets all meters irrespecive of `exclude_simple_reset`"""
        self._reset_progress_bars(reset_train=False, reset_val=True, reset_all=True)
    
    ######### Data Loading #########          
    def _load_data_train(self) -> Tuple[DAT_TYPE, DAT_TYPE]:
        return self._load_data(self.train_iters)
    
    def _load_data_val(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._load_data(self.val_iter)
        
    def _load_data(self, iters: ForeverDataIterator | Sequence[ForeverDataIterator]) -> Tuple[DAT_TYPE, DAT_TYPE]:
        raise NotImplementedError
    
    ######### Forward pass #########
    @overload
    def _forward(self, data: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:...
    @overload
    def _forward(self, data: Tuple[torch.Tensor]) -> Sequence[torch.Tensor] | Tuple[Sequence[torch.Tensor], torch.Tensor]:...
    @overload
    def _forward(self, data: Tuple[torch.Tensor]) -> Sequence[Sequence[torch.Tensor]] | Tuple[Sequence[Sequence[torch.Tensor]], torch.Tensor]:...
    def _forward(self, data: DAT_TYPE) -> PRED_TYPE | Tuple[PRED_TYPE, torch.Tensor]:
        if isinstance(data, torch.Tensor):
            data = (data,)
        x = torch.cat(data, dim=0)
        return self.model(x)
    
    def _forward_val(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass for validation.
        Different from `_forward_train` as it always only returns the predictions.
        This is based on the assumption that the model is in eval mode and its forward does not return features.
        Input for validation is also always a single tensor."""
        return self._forward(data)
    
    @overload
    def _forward_train(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:...
    @overload
    def _forward_train(self, data: Tuple[torch.Tensor,...]) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:...
    @overload
    def _forward_train(self, data: Tuple[torch.Tensor,...]) -> Tuple[Sequence[Sequence[torch.Tensor]]]:...
    def _forward_train(self, data: DAT_TYPE) -> Tuple[TRAIN_PRED_TYPE, torch.Tensor]:
        """Forward pass for training. Different from `_forward_val` it can handle multiple tensors as input
        and can return the features (if the model forward function returns them)."""
        return self._forward(data)
    
    ######### Loss Computation ######### 
    @overload
    def _map_labels(self, labels: torch.Tensor, mode: str, train: bool = True) -> torch.Tensor:...
    @overload
    def _map_labels(self, labels: Tuple[torch.Tensor,...], mode: str, train: bool = True) -> List[torch.Tensor]:...
    def _map_labels(self, labels: LABEL_TYPE, mode: str, train: bool = True) -> torch.Tensor | List[torch.Tensor]:
        """Function stub to map the (ground truth) labels based on the dataset descriptors and the specified `mode`.
        To provide mapping functionality this function needs to be overwritten in derived classes.
        In its base form it just returns the labels as they are i.e. no mapping.
        Which descriptors and hierarchy levels (i.e. train vs val) are used is based on the `train` parameter.
        If `train=False` i.e. validation the labels must be single Tensors.

       Args:
            labels (torch.Tensor | Tuple[torch.Tensor]): Labels to be mapped
            mode (str): mode of the mapping is reversed.
            train (bool): Whether the mapping is done for training or validation. Defaults to True.

        Returns:
            torch.Tensor | List[torch.Tensor]: Mapped ground truth labels.
        """
        return labels
    
    def _compute_loss(self, 
                      pred: PRED_TYPE, 
                      target: LABEL_TYPE, 
                      features: Optional[FEATURE_TYPE]=None, 
                      og_labels: Optional[LABEL_TYPE]=None
                      ) -> torch.Tensor:
        """Compute the total loss for the model.
        This is a separate function to allow for easier overwriting in derived classes.
        E.g. if the loss computation is more complex (e.g. ADDA) or has additional steps (e.g. UJDA).
        
        Assumes that `target` is already mapped to the correct internal class indices as `self._map_labels` 
        is called in `~_train_computation`.
        
        Loss scaling based on `self.scale_loss_accum` and `self.grad_accum_steps` is handeled in `_train_computation`."""
        cls_loss = self._compute_loss_cls(pred=pred, target=target)
        
        pred, target, features, og_labels = self._filter_loss_components(pred=pred, target=target, 
                                                                         features=features, og_labels=og_labels)
        
        adaptation_loss = self._compute_loss_adaptation(pred=pred, target=target, 
                                                        features=features, og_labels=og_labels)
        # Only returns non zero if feature loss is used
        feature_loss = self._compute_loss_features(features=features)
        # Only returns non zero if logit loss is used
        logit_loss = self._compute_loss_logits(pred=pred)
        
        total_loss = cls_loss + adaptation_loss + feature_loss + logit_loss
        self.meter_total_loss.update(total_loss.item(), self.batch_size)
        return total_loss
    
    def _compute_loss_cls(self, 
                          pred: PRED_TYPE, 
                          target: LABEL_TYPE) -> torch.Tensor:
        """Compute the classification loss for the model.
        This loss is generally independent of the domain adaptation method but can be modified if necessary.
        E.g. for UJDA which uses two classification losses.
        Requires that the `target` corresponds to the correct internal class indices i.e. after mapping
        using `~self._map_labels` with `mode='pred'`."""
        raise NotImplementedError
    
    def _compute_loss_adaptation(self, 
                                 pred: PRED_TYPE, 
                                 target: LABEL_TYPE,
                                 features: Optional[FEATURE_TYPE]=None,
                                 og_labels: Optional[LABEL_TYPE]=None) -> torch.Tensor:
        """Compute the additional adaptation loss for the model.
        This loss is the main difference of the domain adaptation methods and thus needs to implemented in the subclasses.
        The returned loss should be a scalar and be scaled with the corresponding hyperparamteres as it gets 
        added to the classification loss as is.
        If no adaptation is used this function should just return 0.
        .. note::
            It might be necessary to make sure the loss is on the same device as the model if 
            non zero loss is returned."""
        return torch.tensor(0)
    
    def _compute_loss_features(self, features: Sequence[FEATURE_TYPE]) -> torch.Tensor:
        """Compute an optional additional feature loss for the model.
        The returned loss should be a scalar and be scaled with the corresponding hyperparamteres as it gets 
        added to the classification loss as is.
        If no feature loss is used or `feature_loss_weight` is 0 this function should just return 0."""
        if self.has_feature_loss:
            feature_loss = self.feature_loss_weight * self.model.compute_feature_loss(features=features)
            # In case adaptation mode filters the filters only account for actual features
            f_count = sum([f_i.size(0) for f_i in features])
            self.meter_feature_loss.update(feature_loss.item(), f_count)
            return feature_loss
        else:
            return torch.tensor(0)
    
    def _compute_loss_logits(self, pred: TRAIN_PRED_TYPE) -> torch.Tensor:
        """Compute an optional additional logit loss for the model.
        The returned loss should be a scalar and be scaled with the corresponding hyperparamteres as it gets 
        added to the classification loss as is.
        If no adaptation is used or `logit_loss_weight` is 0 this function should just return 0."""
        if self.has_logit_loss:
            logit_loss =  self.logit_loss_weight * self.model.compute_logit_loss(logits=pred)
            self.meter_logit_loss.update(logit_loss.item(), self.batch_size)
            return logit_loss
        else:
            return torch.tensor(0)
    
    def _filter_loss_components(self,
                                pred: PRED_TYPE,
                                target: LABEL_TYPE,
                                features: Optional[FEATURE_TYPE]=None,
                                og_labels: Optional[LABEL_TYPE]=None,
                                mode: Optional[str]=None,
                                ) -> Tuple[PRED_TYPE, LABEL_TYPE, Optional[PRED_TYPE], Optional[LABEL_TYPE]]:
        """Interface to filter the loss components before computing the adaptation loss.
        This allows to only compute adaptation loss over parts of the predictions e.g. only source or target domain
        or only over the shared classes.
        The base implementation just returns the input values as is which should correspond to the behavior 
        for `mode='ignore'`.
        The `mode` parameter can be used to specify different filtering strategies. If not specified 
        `mode` is set based on `self.adaptation_filter_mode`.

        Args:
            pred (PRED_TYPE): 
                Predictions of the model.
            target (LABEL_TYPE): 
                Target labels after mapping to internal indices.
            features (Optional[PRED_TYPE], optional): 
                Features of the model for prediction. Defaults to None.
            og_labels (Optional[LABEL_TYPE], optional): 
                Original target classes from dataset (before mapping). Defaults to None.
            mode (Optional[str], optional): 
                Filtering mode. If not specified based on `adaptation_filter_mode`. Defaults to None.

        Returns:
            Tuple[PRED_TYPE, LABEL_TYPE, Optional[PRED_TYPE], Optional[LABEL_TYPE]]: 
                Filtered values for pred, target, features and og_labels.
        """
        return pred, target, features, og_labels
    
    ######### Evaluation/Metric computation ######### 
    def _compute_eval_metrics(self, 
                              pred: PRED_TYPE, 
                              target: LABEL_TYPE, 
                              og_labels: LABEL_TYPE, 
                              features: Optional[FEATURE_TYPE]=None,
                              train: bool = False) -> None:
        raise NotImplementedError
    
    ######### Backward pass/Parameter updates ######### 
    def _update_params(self, loss:torch.Tensor, need_update:bool) -> None:
        if self.loss_scaler is not None:
            self.loss_scaler(
                loss=loss,
                optimizers=self.optimizers,
                create_graph=self.second_order,
                need_update=need_update,
            )
            if need_update:
                # optimizer.step() is done in loss_scaler
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
        else:
            loss.backward(create_graph=self.second_order)
            if need_update:
                for optimizer in self.optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
                    
    ### Optimizer and LR Scheduler ###
    def _zero_grad(self) -> None:
        """Resets the gradients of all optimizers of the model."""
        for optimizer in self.model.optimizers:
            optimizer.zero_grad()
    
    def _lr_scheduler_step(self) -> None:
        """Perform a step of all learning rate schedulers."""
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step(epoch=self.epoch)
    
    ######### Evaluation/Metric display, filtering and storage ######### 
    # "Non-private" as this could be useful to be called from the outside
    def get_metrics_str(self) -> str:
        """Constructs a string displaying the values of the current evaluation metrics for the novel classes.
        E.g. Acc@1: 0.13 | F1: 0.09 | Acc@5: 0.93
        Typically called at the end of a validation run to display the results."""
        metrics = [f'Acc@1: {self.eval_acc_1:3.2f}']
        if 'acc@5' in self.eval_metrics:
            metrics.append(f'Acc@5: {self.eval_acc_5:3.2f}')
        if 'precision' in self.eval_metrics:
            metrics.append(f'Precision: {self.eval_precision:3.2f}')
        if 'recall' in self.eval_metrics:
            metrics.append(f'Recall: {self.eval_recall:3.2f}')
        # Add F1 last for consistent display
        metrics.append(f'F1: {self.eval_f1:3.2f}')
        return " | ".join(metrics)
    
    def _get_val_results(self) -> OrderedDict:
        """Gets the current evaluation metrics for all `self.eval_metrics` metrics."""
        raise NotImplementedError
    
    def _get_train_results(self) -> OrderedDict:
        """Gets the current training metrics for all `self.eval_metrics` metrics."""
        raise NotImplementedError
    
    ######### Training #########
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
        for epoch in range(epochs):
            lrl = [param_group['lr'] for opt in self.optimizers for param_group in opt.param_groups]
            print(f'Learning rate: {",".join([f"{i:.4e}" for i in lrl])}')
            # No need to pass epoch as we set it appropriately in right before
            metrics = self.process_epoch()
            save_checkpoint(model=self.model, 
                            logger=self.logger, 
                            optimizer=self,
                            metric='acc1', 
                            current_best=best_acc1,
                            save_val_test=False)
            
            best_acc1 = max(self.eval_acc_1, best_acc1)
            filewriter.update_summary(epoch=epoch+start_epoch, 
                                      metrics=metrics, 
                                      root=self.logger.out_dir,
                                      write_header=first_epoch)
            first_epoch = False
        print("best_acc1 = {:3.1f}".format(best_acc1))
        print(f'Total time for training: {time.time() - start_time}')
    
    # This function could be reasonably called from outside so keep it public
    def process_epoch(self, epoch: Optional[int]=None) ->OrderedDict:
        metrics = OrderedDict()
        lrl = [param_group['lr'] for opt in self.optimizers for param_group in opt.param_groups]
        lr = sum(lrl) / len(lrl)
        metrics.update([('lr',lr)])
            
        train_metrics = self._train_epoch(epoch)
        metrics.update({f'train_{k}':v for k,v in train_metrics.items()})
        
        val_metrics = self.validate()
        metrics.update({f'eval_{k}':v for k,v in val_metrics.items()})
        
        self._lr_scheduler_step()
        # Increase Epoch
        # Done after lr update as this epoch counter starts with 1
        self.epoch += 1
        
        return metrics
    
    def _train_computation(self, 
                           data: DAT_TYPE, 
                           labels: LABEL_TYPE, 
                           accum_steps: int,
                           need_update: bool) -> None:
        """Perform the computation for training a single batch.
        This includes the forward pass, loss computation, evaluation metrics and parameter update.
        This is in a separate function to allow for easier overwriting in derived classes.
        E.g. if the forward pass should be split into multiple parts or has multiple steps (UJDA)."""
        # Use autocast for mixed precision
        with self.amp_autocast():
            # Forward Pass/Compute Output and features
            forward_res, f = self._forward_train(data)

            interal_labels = self._map_labels(labels=labels, mode='pred')
            
            # Remove the values from the target/labels corresponding to levels that are not given in the predictions
            # e.g. if the dataset has labels for 3 levels but the model only predicts for 2 levels
            # Negative indices as we go from coarse to fine.
            # During train this will also affect eval metrics (but not for val or test later)
            # which is fine as it is questionable when one uses eval during train and then
            # why also look at other levels which are not predicted.
            if self.multiple_outputs:
                labels = [l[:,self.model.classifier.label_mask] for l in labels]
                interal_labels = [l[:,self.model.classifier.label_mask] for l in interal_labels]
            
            # Compute Loss and update meters
            # Pass mapped labels to compute loss.
            loss = self._compute_loss(pred=forward_res, 
                                      target=interal_labels, 
                                      og_labels=labels,
                                      features=f)
            # Scale loss if accumulation to preserve same impact in backward
            # (Especially for last accumulation step?) --> Taken from timm
            # Only after updating meters as those need original value
            if self.scale_loss_accum and accum_steps > 1:
                loss /= accum_steps
        
        if self.eval_during_train:
            # Compute Eval Metrics (Accuracy) and Update Meters
            self._compute_eval_metrics(pred=forward_res, 
                                       target=interal_labels,
                                       og_labels=labels, 
                                       features=f, 
                                       train=True)
        
        # Compute Gradient and do SGD/Optimizer step if update required
        self._update_params(loss=loss, need_update=need_update)
    
    def _train_epoch(self, epoch:Optional[int]=None) -> OrderedDict:
        self.reset_train()
        # self.model.loss_component_accum = torch.zeros_like(self.model.loss_component_accum)
        # self.model.loss_component_avg = torch.zeros_like(self.model.loss_component_avg)
        # self.model.loss_component_step_count = 0
        # Check if mixup should be disabled (only if mixup_off_epoch is set to a value > 0)
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
                # print("Loss components hierarchy: " + "|".join([f'{i.item():.2f}' for i in self.model.loss_component_avg]))
                self.reset_batch_meters(train=True)
        ### End Training
        # Update of lr is performed in process_epoch after validation computation
        
        # Display last results
        self.progress_bar_train.display(self.iters_per_epoch)
        # Epoch gets increased externally (e.g. in process_epoch)
        print(f'Train Epoch took: {time.time() - epoch_start_time}')
        
        return self._get_train_results()
    
    
    ######### Validation #########
    def validate(self, 
                 phase: str='Validation',
                 progress_bar_prefix : Optional[str] = None) -> OrderedDict:
        self.reset_val()
        self.confusion_matrix.reset()
        print(f'Start {phase}')
        print('-'*40)
        
        if progress_bar_prefix:
            self.progress_bar_val.prefix = progress_bar_prefix
        else:
            self.progress_bar_val.prefix = f"Val Epoch: [{self.epoch}]"
        
        # Switch to eval mode
        self.model.eval()
        # Construct iterator
        self.val_iter = iter(self.val_loader)
        
        val_start_time = time.time()
        end = time.time()
        with torch.no_grad():
            for i in range(self.epoch_length_val):
                # Load Data
                data, labels = self._load_data_val()
                
                # Use autocast for mixed precision
                with self.amp_autocast():
                    # Forward Pass/Compute Output
                    # Due to model.eval() only the predictios are returned not the features
                    y = self._forward_val(data)
                
                # Compute Eval Metrics (Accuracy) and Update Meters
                self._compute_eval_metrics(pred=y, 
                                           target=self._map_labels(labels=labels, mode='pred', train=False),
                                           og_labels=labels, 
                                           features=None, 
                                           train=False)
                
                # measure elapsed time
                self.batch_time.update(time.time() - end)
                end = time.time()
            
                if i % self.print_freq == 0:
                    self.progress_bar_val.display(i)
                    self.reset_batch_meters(val=True)
        # Display last results in progress bar
        self.progress_bar_val.display(self.epoch_length_val)
        # Epoch gets increased externally (e.g. in process_epoch)
        # self._set_eval_results() # Not needed as we have the properties for the metrics
        # Display final metrics
        print(self.get_metrics_str())
        print(f'{phase} took: {time.time() - val_start_time}')
        if self.create_class_summary:
            self.confusion_matrix.update_class_summary(file='classMetricsFull',
                                                dir=self.logger.out_dir,)
            self.confusion_matrix.update_class_summary(file='classMetricsEval',
                                                dir=self.logger.out_dir,
                                                start_class=-len(self.eval_classes),)
        return self._get_val_results()