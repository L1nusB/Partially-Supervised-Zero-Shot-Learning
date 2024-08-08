import time

from typing import List, Optional, Set, Tuple, Sequence, overload
from collections import OrderedDict
import warnings

import torch

from PSZS.datasets import DatasetDescriptor
from torch.utils.data import DataLoader

from PSZS.Utils.meters import DynamicStatsMeter, ProgressMeter, _BaseMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Utils.evaluation import accuracy, accuracy_hierarchy
from PSZS.Optimizer.Base_Optimizer import Base_Optimizer, TRAIN_PRED_TYPE, LABEL_TYPE, PRED_TYPE
from PSZS.Models import CustomModel
from PSZS.Utils.evaluation import PrecisionRecallF1, MultiplePrecisionRecallF1
from PSZS.Utils.io.logger import Logger
from PSZS.datasets import transform_target

class Base_Multiple(Base_Optimizer):
    def __init__(self, train_iters: Sequence[ForeverDataIterator], 
                 val_loader: DataLoader,
                 model: CustomModel, 
                 iters_per_epoch: int, 
                 batch_size: int, 
                 eval_classes: Sequence[int],
                 logger: Logger,
                 device: torch.device, 
                 iter_names: Optional[Sequence[str]]=None,
                 **optim_kwargs,
                 ) -> None:
        if getattr(val_loader.dataset, 'descriptor', None) is not None:
            self.hierarchy_level_names = val_loader.dataset.descriptor.hierarchy_level_names
        else:
            self.hierarchy_level_names = [f'Level{i+1}' for i in range(model.classifier.num_head_pred)]
        
        if model.classifier.returns_multiple_outputs:
            self.hierarchy_levels = [None] * len(train_iters)
        else:
            # Generally the dataset is a Concat Dataset which has the attribute 'main_class_index'
            # otherwise use -1 as default value
            self.hierarchy_levels = [getattr(it.dataset, 'main_class_index', -1) for it in train_iters]
        # If validation dataset does not have a main_class_index attribute use the 
        # value of the first training dataset
        # as this corresponds to the source domain which should match the validation
        self.hierarchy_level_val: int | None = getattr(val_loader, 'main_class_index', self.hierarchy_levels[0])
        # Needs to be set before calling super().__init__ (needed in _build_progress_bars)
        self.main_metric_field : str = self.hierarchy_level_names[self.hierarchy_level_val]
        
        if iter_names:
            assert len(iter_names)==len(train_iters), \
                f'Number of iterator names {len(iter_names)} does not match number of iterators {len(train_iters)}'
            iter_names = iter_names
        else:
            print("No iter_names given. Set default values")
            if len(train_iters) == 2:
                iter_names = ['source', 'target']
            else:
                iter_names = [f'D{i}' for i in range(len(train_iters))]
        total_fields = ['total']
        # For multiple outputs we have to watch each output separately (for losses meter during training)
        # can not use self.model as it is not yet initialized
        if model.classifier.returns_multiple_outputs:
            # Will have the form name1_lvl1, name1_lvl2, name2_lvl1, name2_lvl2
            iter_names_full = [f'{name}_{lvl}' for name in iter_names for lvl in self.hierarchy_level_names]
            total_fields_full = [f'total_{lvl}' for lvl in self.hierarchy_level_names]   
            self._calculate_metrics_train = self._calculate_metrics_train_multiple_output
            # For multiple outputs we have to watch each total output separately (for losses meter during training)
            # For train we have no eval classes and thus MultiplePrecisionRecallF1 is not needed
            self.PrecRecF1_train_totals = [PrecisionRecallF1(num_classes=num_classes,
                                                            topk=1, device=device)
                                                for num_classes in model.classifier.effective_head_pred_size.max(dim=0).values
                                           ]
            
        else:
            iter_names_full = iter_names
            total_fields_full = total_fields
            self._calculate_metrics_train = self._calculate_metrics_train_single_output
            # Shape of effective_head_pred_size can be ignored as it is only used for multiple outputs
            # For train we have no eval classes and thus MultiplePrecisionRecallF1 is not needed
            self.PrecRecF1_train_total = PrecisionRecallF1(num_classes=model.classifier.effective_head_pred_size.max(),
                                                        topk=1, device=device)
        # For train we have no eval classes and thus MultiplePrecisionRecallF1 is not needed
        self.PrecRecF1_train_components = [PrecisionRecallF1(num_classes=num_classes.item(),
                                                                topk=1, device=device) 
                                           for i in range(model.classifier.num_inputs)
                                           for num_classes in model.classifier.effective_head_pred_size[i]
                                            ]
        # Need to construct and set watched_fields_loss and watched_fields_full before calling super().__init__
        # as they are needed in _build_progress_bars()
        self.watched_fields_loss = total_fields + iter_names
        self.watched_fields_full = total_fields_full + iter_names_full
        super().__init__(
            train_iters=train_iters,
            val_loader=val_loader,
            model=model,
            iters_per_epoch=iters_per_epoch,
            batch_size=batch_size,
            eval_classes=eval_classes,
            logger=logger,
            device=device,
            **optim_kwargs,
        )
        
        # Validation F1 can only be constructed after self.eval_classes is set appropriately i.e. after super().__init__
        self.PrecRecF1_val = MultiplePrecisionRecallF1(num_classes=model.classifier.effective_head_pred_size.max(),
                                                      topk=1, device=device,
                                                      evalClasses=self.eval_classes)
        
        self._expand_progress_bars(train_progress=self.progress_bar_train, val_progress=self.progress_bar_val)
        
        self.shared_classes: List[int] = self._get_shared_classes()
        
        # Typehint correctly for intellisense
        self.train_batch_sizes: List[int]
        self.hierarchy_levels : List[int | None]
        self.train_iters : List[ForeverDataIterator]
        # Avoid checks on every data load call
        if self.send_to_device:
            self._load_data = self._load_data_send_to_device
        else:
            self._load_data = self._load_data_on_device
            
            
    @property
    def num_head_pred(self) -> int:
        return self.model.classifier.num_head_pred
    
    # Overwrite to always work with Sequence[ForeverDataIterator] and thus return a list.
    @property
    def train_descriptors(self) -> List[DatasetDescriptor | None]:
        return [it.dataset_descriptor for it in self.train_iters]
    
    @property
    def multiple_outputs(self) -> bool:
        return self.model.classifier.returns_multiple_outputs
    
    def _get_shared_classes(self) -> Set[int]:
        """The shared classes are based on the second train iterator as this is the target domain."""
        descriptor = self.train_iters[1].dataset_descriptor
        if descriptor is not None:
            main_class_index = getattr(self.train_iters[1].dataset, 'main_class_index', -1)
            return descriptor.targetIDs[main_class_index]
        else:
            warnings.warn("No dataset descriptor found for shared classes. "
                          "Setting shared classes as continuous number of classes.")
            return set(range(self.train_iters[1].num_classes))
    
    ######### Progress Bars #########   
    def _build_progress_bars(self) -> Tuple[ProgressMeter, ProgressMeter]:
        self.meter_cls_loss = DynamicStatsMeter.get_stats_meter_min_max("Cls Loss", 
                                                                fields=self.watched_fields_loss, 
                                                                fmt=":6.2f",
                                                                defaultField=self.main_metric_field)
        # Separate metric meters for train and validation
        # For train we only care about the novel classes
        self.meter_cls_accs_1_train = DynamicStatsMeter.get_stats_meter_min_max("Acc@1", 
                                                                fields=self.watched_fields_full, 
                                                                fmt=":3.2f",
                                                                defaultField=self.main_metric_field)
        self.meter_cls_accs_5_train = DynamicStatsMeter.get_stats_meter_min_max("Acc@5", 
                                                                fields=self.watched_fields_full, 
                                                                fmt=":3.2f",
                                                                defaultField=self.main_metric_field)
        self.meter_precision_train = DynamicStatsMeter.get_average_meter("Precision", 
                                                                fields=self.watched_fields_full, 
                                                                fmt=":3.2f",
                                                                defaultField=self.main_metric_field,
                                                                include_last=True)
        self.meter_recall_train = DynamicStatsMeter.get_average_meter("Recall", 
                                                                fields=self.watched_fields_full, 
                                                                fmt=":3.2f",
                                                                defaultField=self.main_metric_field,
                                                                include_last=True)
        self.meter_f1_train = DynamicStatsMeter.get_average_meter("F1", 
                                                                fields=self.watched_fields_full, 
                                                                fmt=":3.2f",
                                                                defaultField=self.main_metric_field,
                                                                include_last=True)
        
        def _get_metric_meters_train() -> List[_BaseMeter]:
            meters = [self.meter_cls_accs_1_train, self.meter_f1_train]
            if 'acc@5' in self.eval_metrics:
                meters.append(self.meter_cls_accs_5_train)
            if 'precision' in self.eval_metrics:
                meters.append(self.meter_precision_train)
            if 'recall' in self.eval_metrics:
                meters.append(self.meter_recall_train)
            return meters
        
        # Overwrite accuracy meters to show all hierarchy levels (other meters are created in the base class)
        # and only show the main hierarchy level
        if self.single_eval_class:
            self.meters_cls_acc_1 = [DynamicStatsMeter.get_stats_meter_min_max("Acc@1", 
                                                                    fields=self.hierarchy_level_names, 
                                                                    fmt=":3.2f",
                                                                    defaultField=self.main_metric_field)]
            self.meters_cls_acc_5 = [DynamicStatsMeter.get_stats_meter_min_max("Acc@5", 
                                                                    fields=self.hierarchy_level_names, 
                                                                    fmt=":3.2f",
                                                                    defaultField=self.main_metric_field)]
        else:
            self.meters_cls_acc_1 = [DynamicStatsMeter.get_stats_meter_min_max(f"Acc@1({name})", 
                                                                    fields=self.hierarchy_level_names, 
                                                                    fmt=":3.2f",
                                                                    defaultField=self.main_metric_field) 
                                     for name in self.eval_groups_names]
            self.meters_cls_acc_5 = [DynamicStatsMeter.get_stats_meter_min_max(f"Acc@5({name})", 
                                                                    fields=self.hierarchy_level_names, 
                                                                    fmt=":3.2f",
                                                                    defaultField=self.main_metric_field) 
                                     for name in self.eval_groups_names]
        
        metric_meters_train = _get_metric_meters_train()
        metric_meters_val = self._get_metric_meters()
        
        loss_meters = [self.meter_cls_loss, self.meter_total_loss]
        if self.has_feature_loss:
            loss_meters.append(self.meter_feature_loss)
        if self.has_logit_loss:
            loss_meters.append(self.meter_logit_loss)
        
        train_progress = ProgressMeter(num_batches=self.iters_per_epoch,
                                       meters=[self.batch_time, self.data_time] + loss_meters, 
                                       batch_meters=[self.batch_time, self.data_time],
                                       exclude_simple_reset=loss_meters,
                                       prefix=f"Train Epoch: [{self.epoch}]")
        if self.eval_during_train:
            train_progress.add_meter(metric_meters_train, exclude_simple_reset=True)
            
        val_progress = ProgressMeter(num_batches=self.epoch_length_val,
                                     meters=[self.batch_time, self.data_time] + metric_meters_val, 
                                     batch_meters=[self.batch_time, self.data_time],
                                     exclude_simple_reset=metric_meters_val,
                                     prefix=f"Val Epoch: [{self.epoch}]")
        return train_progress, val_progress
    
    def _expand_progress_bars(self, 
                              train_progress: ProgressMeter, 
                              val_progress: ProgressMeter
                              ) -> Tuple[ProgressMeter, ProgressMeter]:
        """Use this function to expand the progress bars with additional meters
        This way inheriting classes only need to implement this function to add additional meters.
        Base implementation does nothing."""
        return train_progress, val_progress
    
    def reset_train(self):
        super().reset_train()
        if self.multiple_outputs:
            for prec_rec_f1 in self.PrecRecF1_train_totals:
                prec_rec_f1.reset()
        else:
            self.PrecRecF1_train_total.reset()
        for prec_rec_f1 in self.PrecRecF1_train_components:
            prec_rec_f1.reset()
            
    def reset_val(self):
        super().reset_val()
        self.PrecRecF1_val.reset()
    
    ######### Data Loading #########  
    # Replace _load_data function in __init__ (depending on self.send_to_device)
    def _load_data_send_to_device(self, iters: Sequence[ForeverDataIterator]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the data from the iterators and sends it to the device.
        Mixup is applied if specified."""
        x, labels = zip(*[next(it)[:2] for it in iters])
        x = [e.to(self.device) for e in x]
        labels = [l.to(self.device) for l in labels]
        # Use mixup (if given)
        if self.mixup_fn is not None:
            x, labels = zip(*[self.mixup_fn(x[i], labels[i]) for i in range(len(x))])
        data_load_time = time.time()
        self.data_time.update(data_load_time - self.last_time_mark)
        self.last_time_mark = data_load_time
        return x, labels
    
    def _load_data_on_device(self, iters: Sequence[ForeverDataIterator]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the data from the iterators. Data is assumed to be on the device already.
        I.e. usage of PrefetchLoader. Mixup is applied inside the PrefetchLoader (if specified)."""
        x, labels = zip(*[next(it)[:2] for it in iters])
        data_load_time = time.time()
        self.data_time.update(data_load_time - self.last_time_mark)
        self.last_time_mark = data_load_time
        return x, labels
    
    def _load_data_val(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the data for validation. For hierarchical models only the main hierarchy level is returned 
        based on `self.model.classifier.test_head_pred_ix`."""
        # Since load data requires iters to be a sequence and returns tuples 
        # they have to be split up here again
        x, labels = self._load_data([self.val_iter])
        # Filter labels to only be the main hierarchy level for test/validation
        if self.model.classifier.test_head_pred_idx is not None:
            labels = [label[:,self.model.classifier.test_head_pred_idx] for label in labels]
        return x[0], labels[0]
    
    ######### Forward pass #########
    @overload
    def _forward_train(self, data: Tuple[torch.Tensor,...]) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:...
    @overload
    def _forward_train(self, data: Tuple[torch.Tensor,...]) -> Tuple[Sequence[Sequence[torch.Tensor]], Sequence[Sequence[torch.Tensor]]]:...
    def _forward_train(self, data: Tuple[torch.Tensor,...]) -> Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE]:
        """Computes forward pass for training data. Returns the predictions and the features which are split
        based on the batch sizes of the iterators to correspond to the respective domains of the iterators.

        Args:
            data (Tuple[torch.Tensor,...]): Input data. Each entry in the tuple corresponds to a different iterator/domain.

        Returns:
            Tuple[TRAIN_PRED_TYPE, TRAIN_PRED_TYPE]: Split predictions and features.
        """
        y, f = super()._forward(data)
        # Use tensor_split to split feature tensor into parts of each domain with respective length
        split_indices = [sum(self.train_batch_sizes[:i]) for i in range(1, len(self.train_batch_sizes))]
        f = f.tensor_split(split_indices, dim=0)
        return y, f
    
    def _forward_val(self, data: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass for validation data. Returns only the predictions.
        Based on the assumption that the model is in evaluation mode and its forward thus only returns predictions.
        .. note::
            No need to split here as validation only has single data loader"""
        return super()._forward(data)
    
    ######### Loss Computation ######### 
    def _map_labels(self, labels: LABEL_TYPE, mode: str, train: bool = True) -> torch.Tensor | List[torch.Tensor]:
        """Map the (ground truth) labels to based on the dataset descriptors and the specified `mode`.
        Which descriptors and hierarchy levels (i.e. train vs val) are used is based on the `train` parameter.
        If `train=False` i.e. validation the labels must be single Tensors.
        Mapping is performed based on `self.hierarchy_levels` or `self.hierarchy_level_val` of the used datasets.
        Available values for `mode` are based on `~PSZS.datasets.transform_target()`
            - 'target': Map the internal prediction index to the target class as in the annotation file
            - 'pred': Map the (original) target class to the internal prediction index
            - 'name': Map the target class to the class name. Needs to be the original target class, not the internal prediction index.
            - 'targetIndex': Map the class id to the index of that class in the target (independent of internal prediction)

        Args:
            labels (torch.Tensor | Tuple[torch.Tensor,...]): Labels to be mapped
            mode (str): mode of the mapping is reversed.
            train (bool): Whether the mapping is done for training or validation. Defaults to True.

        Returns:
            torch.Tensor | List[torch.Tensor]: Mapped ground truth labels.
        """
        if train:
            return [transform_target(target=gt, mode=mode, descriptor=desc, hierarchy_level=lvl) 
                        for gt, desc, lvl in zip(labels, self.train_descriptors, self.hierarchy_levels)]
        else:
            return transform_target(target=labels, mode=mode, descriptor=self.val_descriptor, 
                                    hierarchy_level=self.hierarchy_level_val)
    
    def _compute_loss_cls(self, 
                          pred: TRAIN_PRED_TYPE, 
                          target: Tuple[torch.Tensor,...]) -> torch.Tensor:
        loss, loss_components = self.model.compute_cls_loss(pred, target)
        
        # In case of uneven split between source and target shared we have to scale the loss
        # and set the number of samples based on the actual counts
        self.meter_cls_loss.update(vals=[loss] + loss_components,
                           n=[self.batch_size] + [trg.size(0) for trg in target])
        return loss
    
    def _filter_loss_components(self,
                                pred: TRAIN_PRED_TYPE,
                                target: Tuple[torch.Tensor,torch.Tensor],
                                features: Sequence[torch.Tensor],
                                og_labels: Tuple[torch.Tensor,torch.Tensor],
                                mode: Optional[str]=None,
                                ) -> Tuple[TRAIN_PRED_TYPE, Tuple[torch.Tensor,torch.Tensor], Sequence[torch.Tensor], Tuple[torch.Tensor,torch.Tensor]]:
        """Interface to filter the loss components before computing the adaptation loss.
        This allows to only compute adaptation loss over parts of the predictions e.g. only source or target domain
        or only over the shared classes.
        The `mode` parameter can be used to specify different filtering strategies. If not specified 
        `mode` is set based on `self.adaptation_filter_mode`.
        Available modes are:
            - 'ignore': No filtering is applied
            - 'source': Only the source domain is considered
            - 'target': Only the target domain is considered
            - 'shared': Only the shared classes are considered

        Args:
            pred (TRAIN_PRED_TYPE): 
                Predictions of the model.
            target (Tuple[torch.Tensor,torch.Tensor]): 
                Target labels after mapping to internal indices.
            features (Sequence[torch.Tensor]): 
                Features of the model for prediction. 
            og_labels (Tuple[torch.Tensor,torch.Tensor]): 
                Original target classes from dataset (before mapping).
            mode (Optional[str], optional): 
                Filtering mode. If not specified based on `adaptation_filter_mode`. Defaults to None.

        Returns:
            Tuple[TRAIN_PRED_TYPE, Tuple[torch.Tensor,torch.Tensor], Sequence[torch.Tensor], Tuple[torch.Tensor,torch.Tensor]]: 
                Filtered values for pred, target, features and og_labels.
        """
        if mode is None:
            mode = self.adaptation_filter_mode
            
        if mode == 'ignore':
            return pred, target, features, og_labels
        elif mode == 'source':
            # Filter out target domain i.e. delete entries
            # Not sure if this works/makes sense
            pred = (pred[0],)
            target = (target[0],)
            features = (features[0],)
            og_labels = (og_labels[0],)
        elif mode == 'target':
            # Filter out source domain i.e. delete entries
            # Not sure if this works/makes sense
            pred = (pred[1],)
            target = (target[1],)
            features = (features[1],)
            og_labels = (og_labels[1],)
        elif mode == 'shared':
            if self.multiple_outputs:
                index = self.model.classifier.test_head_pred_idx
                source_mask = torch.tensor([lab.item() in self.shared_classes for lab in og_labels[0][:, index]])
                # Only filter out the source domain as the target domain already contains only shared
                pred = ([p[source_mask==1] for p in pred[0]], pred[1])
                target = (target[0][source_mask==1], target[1])
                features = (features[0][source_mask==1], features[1])
                og_labels = (og_labels[0][source_mask==1], og_labels[1])
            else:
                source_mask = torch.tensor([lab.item() in self.shared_classes for lab in og_labels[0]])
                # Only filter out the source domain as the target domain already contains only shared
                pred = (pred[0][source_mask==1], pred[1])
                target = (target[0][source_mask==1], target[1])
                features = (features[0][source_mask==1], features[1])
                og_labels = (og_labels[0][source_mask==1], og_labels[1])
        else:
            raise ValueError(f'Unknown mode {mode} for filtering loss components.')
        return pred, target, features, og_labels
    
    ######### Evaluation/Metric computation ######### 
    @overload
    def _compute_eval_metrics(self, pred: torch.Tensor, target: torch.Tensor, og_labels: torch.Tensor, 
                              features: Optional[Sequence[torch.Tensor]]=None,
                              train: bool = False) -> None:...
    @overload
    def _compute_eval_metrics(self, pred: Sequence[torch.Tensor], target: Tuple[torch.Tensor,...], og_labels: Tuple[torch.Tensor,...], 
                              features: Optional[Sequence[torch.Tensor]]=None,
                              train: bool = False) -> None:...
    @overload
    def _compute_eval_metrics(self, pred: Sequence[Sequence[torch.Tensor]], target: Tuple[torch.Tensor,...], og_labels: Tuple[torch.Tensor,...], 
                              features: Optional[Sequence[torch.Tensor]]=None,
                              train: bool = False) -> None:...
    def _compute_eval_metrics(self, 
                              pred: PRED_TYPE, 
                              target: LABEL_TYPE, 
                              og_labels: LABEL_TYPE, 
                              features: Optional[Sequence[torch.Tensor]]=None,
                              train: bool = False) -> None:
        if train:
            self._compute_eval_metrics_train(pred=pred,
                                             target=target,
                                             og_labels=og_labels,
                                             features=features,)
        else:
            self._compute_eval_metrics_val(pred=pred,
                                           target=target,
                                           og_labels=og_labels,
                                           features=features,)
            
    def _calculate_metrics_train_single_output(self, 
                                               pred: Sequence[torch.Tensor],
                                               target: Tuple[torch.Tensor,...], 
                                               og_labels: Tuple[torch.Tensor,...],
                                               features: Optional[Sequence[torch.Tensor]],
                                               ) -> Tuple[List[float], List[float], List[int],
                                                           List[float], List[float], List[float],
                                                           List[int]]:
        # Since the predictions for each iterator in train_iter can have a different number of classes
        # i.e. source vs. target shared, we have to pad the tensors to the same length
        # to calculate joint total metrics.
        # New values are added to the end of the tensor and filled with zeros to match the shapes.
        # Individual metrics are calculated for each iterator separately later.
        # For single output we can simply pad the tensors due to preditions being Sequence[torch.Tensor]
        max_len = max(pred[0].shape[1], pred[1].shape[1])
        full_pred = torch.cat((torch.nn.functional.pad(pred[0], (0, max_len - pred[0].shape[1])),
                               torch.nn.functional.pad(pred[1], (0, max_len - pred[1].shape[1]))), 
                               dim=0)
        full_pred_labels = torch.cat(target, dim=0)
        # For accuracy during train we don't have eval classes and thus no need to specify original labels
        (cls_acc_1_total, cls_acc_5_total), num_relevant_total = accuracy(prediction=full_pred, 
                                                                          target=full_pred_labels)
        
        # For F1 etc. we need the original labels even during train to update the correct classes
        self.PrecRecF1_train_total.update(prediction=full_pred, 
                                          target=full_pred_labels,)
        prec_total, recall_total, f1_total = self.PrecRecF1_train_total.compute()
        # For accuracy during train we don't have eval classes and thus no need to specify original labels
        (cls_accs_1, cls_accs_5), num_relevant = zip(*[accuracy(prediction=pred[i], target=target[i]) 
                                                       for i in range(self.num_inputs)])
        for i in range(self.num_inputs):
            # For F1 etc. we need the original labels even during train to update the correct classes
            self.PrecRecF1_train_components[i].update(prediction=pred[i], 
                                                      target=target[i],)
        prec, recall, f1 = zip(*[self.PrecRecF1_train_components[i].compute() for i in range(self.num_inputs)])

        acc1_results = [cls_acc_1_total.item()] + [cls_acc.item() for cls_acc in cls_accs_1]
        acc5_results = [cls_acc_5_total.item()] + [cls_acc.item() for cls_acc in cls_accs_5]
        num_relevants = [num_relevant_total] + list(num_relevant)
        prec_results = [prec_total.item()] + [p.item() for p in prec]
        recall_results = [recall_total.item()] + [r.item() for r in recall]
        f1_results = [f1_total.item()] + [f.item() for f in f1]
        f1_meter_counts = [self.batch_size] + [tgt.size(0) for tgt in target]
        
        return acc1_results, acc5_results, num_relevants, prec_results, recall_results, f1_results, f1_meter_counts
     
       
        
    def _calculate_metrics_train_multiple_output(self, 
                                                 pred: Sequence[Sequence[torch.Tensor]],
                                                 target: Tuple[torch.Tensor,...], 
                                                 og_labels: Tuple[torch.Tensor,...],
                                                 features: Optional[Sequence[torch.Tensor]],
                                                 ) -> Tuple[List[float], List[float], List[int],
                                                           List[float], List[float], List[float],
                                                           List[int]]:
        # Since the predictions for each iterator in train_iter can have a different number of classes
        # i.e. source vs. target shared, we have to pad the tensors to the same length
        # to calculate joint total metrics.
        # New values are added to the end of the tensor and filled with zeros to match the shapes.
        # Individual metrics are calculated for each iterator separately later.
        # For multiple outputs we have to pad each output separately
        # as the predictions are Sequence[Sequence[torch.Tensor]]
        max_lens = [max(pred[i][j].shape[1] for i in range(self.num_inputs)) for j in range(self.num_head_pred)]
        full_pred = [torch.cat([torch.nn.functional.pad(p[lvl], (0, max_lens[lvl] - p[lvl].shape[1])) for p in pred], dim=0) 
                   for lvl in range(self.num_head_pred)]
        
        # Transpose to get desired shape
        full_pred_labels = torch.cat(target, dim=0).t()
        
        # For accuracy during train we don't have eval classes and thus no need to specify original labels
        cls_acc_totals, num_relevant_total = zip(*[accuracy(prediction=full_pred[i], 
                                                            target=full_pred_labels[i]) 
                                                    for i in range(self.num_head_pred)])
        # cls_[acc_1/acc_5]_total contains accuracy for pred1 and pred2 (e.g. make and model)
        cls_acc_1_totals, cls_acc_5_totals = zip(*cls_acc_totals)
        
        for i in range(self.num_head_pred):
            # For F1 etc. we need the original labels even during train to update the correct classes
            self.PrecRecF1_train_totals[i].update(prediction=full_pred[i], 
                                                  target=full_pred_labels[i],)
        prec_total, recall_total, f1_total = zip(*[self.PrecRecF1_train_totals[i].compute() 
                                                   for i in range(self.num_head_pred)])
        
        # Fields have the form name1_lvl1, name1_lvl2, name2_lvl1, name2_lvl2
        # so order of iteration is important
        # Need to transpose to get the correct shape as gt is [[lvl1,lvl2],[lvl1,lvl2],...]
        # but we need [[lvl1,lvl1,...],[lvl2,lvl2,...]]
        # cls_accs will have the following form:
        # [acc1_i1_lvl1, acc5_i1_lvl1], [acc1_i1_lvl2, acc5_i1_lvl2], [acc1_i2_lvl1, acc5_i2_lvl1], [acc1_i2_lvl2, acc5_i2_lvl2]
        # --> Correct order as expected for watched_fields_full
        #
        # For accuracy during train we don't have eval classes and thus no need to specify original labels
        cls_accs, num_relevant = zip(*[accuracy(prediction=pred[i][lvl], 
                                                target=target[i].t()[lvl]) 
                                            for i in range(self.num_inputs) 
                                        for lvl in range(self.num_head_pred)])
        cls_accs_1, cls_accs_5 = zip(*cls_accs)
        for i in range(self.num_inputs):
            for lvl in range(self.num_head_pred):
                # lvl+i*self.num_head_pred to get the correct index for the current level
                # PrecRecF1 components are [input1, input2, ...] * num_head_predictions
                # --> [input1_lvl1, input2_lvl1, input1_lvl2, input2_lvl2, ...]
                # Need to transpose to get the correct shape as gt is [[lvl1,lvl2],[lvl1,lvl2],...]
                # but we need [[lvl1,lvl1,...],[lvl2,lvl2,...]]
                # 
                # For F1 etc. we need the original labels even during train to update the correct classes
                self.PrecRecF1_train_components[lvl+i*self.num_head_pred].update(prediction=pred[i][lvl], 
                                                                                 target=target[i].t()[lvl],)
        # Same order as above so order of iteration is important
        prec, recall, f1 = zip(*[self.PrecRecF1_train_components[i*lvl].compute() 
                                 for i in range(self.num_inputs)
                                 for lvl in range(self.num_head_pred)])
        
        acc1_results = [cls_acc.item() for cls_acc in cls_acc_1_totals] + [cls_acc.item() for cls_acc in cls_accs_1]
        acc5_results = [cls_acc.item() for cls_acc in cls_acc_5_totals] + [cls_acc.item() for cls_acc in cls_accs_5]
        num_relevants = list(num_relevant_total) + list(num_relevant)
        prec_results = [p.item() for p in prec_total] + [p.item() for p in prec]
        recall_results = [r.item() for r in recall_total] + [r.item() for r in recall]
        f1_results = [f.item() for f in f1_total] + [f.item() for f in f1]
        f1_meter_counts = [self.batch_size]*self.num_head_pred + [tgt.size(0) for tgt in target]*self.num_head_pred
        
        return acc1_results, acc5_results, num_relevants, prec_results, recall_results, f1_results, f1_meter_counts
    
    @overload
    def _compute_eval_metrics_train(self, pred: Sequence[torch.Tensor], target: Tuple[torch.Tensor,...], 
                                    og_labels: Tuple[torch.Tensor,...], features: Optional[Sequence[torch.Tensor]]=None,) -> None:...
    @overload
    def _compute_eval_metrics_train(self, pred: Sequence[Sequence[torch.Tensor]], target: Tuple[torch.Tensor,...], 
                                    og_labels: Tuple[torch.Tensor,...], features: Optional[Sequence[torch.Tensor]]=None,) -> None:...
            
    def _compute_eval_metrics_train(self, 
                                    pred: TRAIN_PRED_TYPE, 
                                    target: Tuple[torch.Tensor,...], 
                                    og_labels: Tuple[torch.Tensor,...], 
                                    features: Optional[Sequence[torch.Tensor]]=None,
                                    ) -> None:
        # Separated for readability
        metrics = self._calculate_metrics_train(pred=pred, 
                                                target=target, 
                                                og_labels=og_labels,
                                                features=features,)
        cls_acc1, cls_acc5, num_relevant, prec, recall, f1, f1_meter_counts = metrics
        # Scale batch_size to update counter with number of iters 
        # Only account for relevant samples when accuracy updating meters
        self.meter_cls_accs_1_train.update(vals=cls_acc1, n=num_relevant)
        self.meter_cls_accs_5_train.update(vals=cls_acc5, n=num_relevant)
        # In case of uneven split between source and target shared we have to scale the loss
        # and set the number of samples based on the actual counts
        # For precision, recall and f1 all samples are relevant
        self.meter_precision_train.update(vals=prec, n=f1_meter_counts)
        self.meter_recall_train.update(vals=recall, n=f1_meter_counts)
        self.meter_f1_train.update(vals=f1, n=f1_meter_counts)
    
    def _compute_eval_metrics_val(self, 
                                  pred: torch.Tensor, 
                                  target: torch.Tensor, 
                                  og_labels: torch.Tensor, 
                                  features: Optional[torch.Tensor]=None,
                                  ) -> None:
        self.confusion_matrix.update(prediction=pred, target=target)
        cls_accs, nums_relevant = accuracy_hierarchy(prediction=pred, 
                                                   target=target, 
                                                   hierarchy_map=self.val_descriptor.pred_fine_coarse_map,
                                                   originalTarget=og_labels,
                                                   evalClasses=self.eval_classes)
        self.PrecRecF1_val.update(prediction=pred, 
                                  target=target,)
        prec, recall, f1 = self.PrecRecF1_val.compute()
        
        update_size = self.batch_size
        if self.single_eval_class:
            # Need to rewrap the results for zipping and uniform handling below
            # somewhat unintuitive as accuracy_hierarchy specifically unpacks it
            # but this is for allowing more flexibility of its usage
            accs_results = zip([cls_accs], [nums_relevant])
        else:
            accs_results = zip(cls_accs, nums_relevant)
            
        for eval_group, ((cls_acc_1, cls_acc_5), num_relevant) in enumerate(accs_results):
            # Only account for relevant samples when accuracy updating meters
            self.meters_cls_acc_1[eval_group].update(vals=cls_acc_1, n=num_relevant)
            self.meters_cls_acc_5[eval_group].update(vals=cls_acc_5, n=num_relevant)
            # For precision, recall and f1 all samples are relevant
            self.meters_precision[eval_group].update(prec[eval_group].item(), update_size)
            self.meters_recall[eval_group].update(recall[eval_group].item(), update_size)
            self.meters_f1[eval_group].update(f1[eval_group].item(), update_size)
    
    
    ######### Evaluation/Metric display and filtering#########     
    def _get_val_results(self) -> OrderedDict:
        """Gets the current evaluation metrics for all `self.eval_metrics` 
        from the respective internal properties."""
        results = OrderedDict()
        if self.single_eval_class:
            for lvl in self.hierarchy_level_names:
                results.update([(f'Acc@1_{lvl}', self.meters_cls_acc_1[0].get_avg(lvl))])
                if 'acc@5' in self.eval_metrics:
                    results.update([(f'Acc@5_{lvl}', self.meters_cls_acc_5[0].get_avg(lvl))])
            if 'precision' in self.eval_metrics:
                results.update([('Precision', self.meters_precision[0].get_last())])
            if 'recall' in self.eval_metrics:
                results.update([('Recall', self.meters_recall[0].get_last())])
            results.update([('F1', self.meters_f1[0].get_last())])
        else:
            for i, name in enumerate(self.eval_groups_names):
                for lvl in self.hierarchy_level_names:
                    results.update([(f'Acc@1_{lvl}({name})', self.meters_cls_acc_1[i].get_avg(lvl))])
                    if 'acc@5' in self.eval_metrics:
                        results.update([(f'Acc@5_{lvl}({name})', self.meters_cls_acc_5[i].get_avg(lvl))])
                if 'precision' in self.eval_metrics:
                    results.update([(f'Precision({name})', self.meters_precision[i].get_last())])
                if 'recall' in self.eval_metrics:
                    results.update([(f'Recall({name})', self.meters_recall[i].get_last())])
                results.update([(f'F1({name})', self.meters_f1[i].get_last())])
        return results
    
    def _get_train_results(self) -> OrderedDict:
        """Gets the current training metrics for all `self.eval_metrics` and classification components
        from the respective meters."""
        results = OrderedDict()
        for watched_field in self.watched_fields_loss:
            results.update([(f'Cls loss_{watched_field}', self.meter_cls_loss.get_avg(watched_field))])
        if self.has_feature_loss:
            results.update([('Feature loss', self.meter_feature_loss.get_avg())])
        if self.has_logit_loss:
            results.update([('Logit loss', self.meter_logit_loss.get_avg())])
        results.update([(f'Loss_total', self.meter_total_loss.get_avg())])
        if self.eval_during_train:
            for watched_field in self.watched_fields_full:
                results.update([(f'Acc@1_{watched_field}', self.meter_cls_accs_1_train.get_avg(watched_field)),
                                (f'F1_{watched_field}', self.meter_f1_train.get_last(watched_field))])
                if 'acc@5' in self.eval_metrics:
                    results.update([(f'Acc@5_{watched_field}', self.meter_cls_accs_5_train.get_avg(watched_field))])
                if 'precision' in self.eval_metrics:
                    results.update([(f'Precision_{watched_field}', self.meter_precision_train.get_last(watched_field))])
                if 'recall' in self.eval_metrics:
                    results.update([(f'Recall_{watched_field}', self.meter_recall_train.get_last(watched_field))])
        return results