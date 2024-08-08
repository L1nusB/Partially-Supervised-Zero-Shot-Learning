import time

from typing import List, Optional, Tuple, Sequence, TypeAlias, overload
from collections import OrderedDict

import torch

from PSZS.datasets import DatasetDescriptor
from torch.utils.data import DataLoader

from PSZS.Utils.meters import DynamicStatsMeter, ProgressMeter, _BaseMeter, StatsMeter
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Utils.evaluation import accuracy, accuracy_hierarchy
from PSZS.Optimizer.Base_Optimizer import Base_Optimizer, PRED_TYPE
from PSZS.Models import CustomModel
from PSZS.Utils.evaluation import MultiplePrecisionRecallF1, PrecisionRecallF1
from PSZS.Utils.io.logger import Logger
from PSZS.datasets import transform_target

SINGLE_TRAIN_PRED_TYPE : TypeAlias = torch.Tensor | Sequence[torch.Tensor]

class Base_Single(Base_Optimizer):
    def __init__(self, 
                 train_iter: ForeverDataIterator, 
                 val_loader: DataLoader,
                 model: CustomModel, 
                 iters_per_epoch: int, 
                 batch_size: int, 
                 eval_classes: Sequence[int],
                 logger: Logger,
                 device: torch.device, 
                 **optim_kwargs,
                 ) -> None:
        if getattr(val_loader.dataset, 'descriptor', None) is not None:
            self.hierarchy_level_names = val_loader.dataset.descriptor.hierarchy_level_names
        else:
            self.hierarchy_level_names = [f'Level{i+1}' for i in range(model.classifier.num_head_pred)]

        if model.classifier.returns_multiple_outputs:
            self.hierarchy_level = None
        else:
            # Generally the dataset is a Concat Dataset which has the attribute 'main_class_index'
            # otherwise use -1 as default value
            self.hierarchy_level = getattr(train_iter.dataset, 'main_class_index', -1)
        # Typehint correctly for intellisense
        self.hierarchy_level : int | None
        # If validation dataset does not have a main_class_index attribute use the 
        # value of the first training dataset
        # as this corresponds to the source domain which should match the validation
        self.hierarchy_level_val: int | None = getattr(val_loader.dataset, 'main_class_index', self.hierarchy_level)
        # Needs to be set before calling super().__init__ (needed in _build_progress_bars)
        self.main_metric_field = self.hierarchy_level_names[self.hierarchy_level_val]
        
        
        # For multiple outputs we have to watch each output separately (for losses meter during training)
        # can not use self.model as it is not yet initialized
        if model.classifier.returns_multiple_outputs:
            lvl_names_full = [f'{lvl}' for lvl in self.hierarchy_level_names]
            total_fields_full = [f'total_{lvl}' for lvl in self.hierarchy_level_names]   
            self._calculate_metrics_train = self._calculate_metrics_train_multiple_output
            # For multiple outputs we have to watch each total output separately (for losses meter during training)
            # For train we have no eval classes and thus MultiplePrecisionRecallF1 is not needed
            self.PrecRecF1_train_totals = [PrecisionRecallF1(num_classes=num_classes,
                                                            topk=1, device=device)
                                                for num_classes in model.classifier.effective_head_pred_size.max(dim=0).values
                                           ]
            
        else:
            lvl_names_full = []
            total_fields_full = ['total']
            self._calculate_metrics_train = self._calculate_metrics_train_single_output
            # Shape of effective_head_pred_size can be ignored as it is only used for multiple outputs
            # For train we have no eval classes and thus MultiplePrecisionRecallF1 is not needed
            self.PrecRecF1_train_totals = [PrecisionRecallF1(num_classes=model.classifier.effective_head_pred_size.max(),
                                                        topk=1, device=device)]
        
        # Need to construct and set watched_fields_loss and watched_fields_full before calling super().__init__
        # as they are needed in _build_progress_bars()
        self.watched_fields_full = total_fields_full + lvl_names_full
        super().__init__(
            train_iters=train_iter,
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
        
        
        assert self.num_inputs == 1, "Only single input/source domain supported"
        # Avoid checks on every data load call
        if self.send_to_device:
            self._load_data = self._load_data_send_to_device
        else:
            self._load_data = self._load_data_on_device
    
    # Since we only have a single object passed to train_iters it is not a sequence
    @property
    def train_iter(self) -> ForeverDataIterator:
        return self.train_iters
    
    # Since we only have a single train_iter train_batch_sizes is only an int
    @property
    def train_batch_size(self) -> int:
        return self.train_batch_sizes
            
    @property
    def num_head_pred(self) -> int:
        return self.model.classifier.num_head_pred
    
    @property
    def train_descriptor(self) -> DatasetDescriptor | None:
        return self.train_iter.dataset_descriptor
    
    @property
    def multiple_outputs(self) -> bool:
        return self.model.classifier.returns_multiple_outputs
    
    ######### Progress Bars #########
    def _build_progress_bars(self) -> Tuple[ProgressMeter, ProgressMeter]:
        self.meter_cls_loss = StatsMeter.get_stats_meter_min_max("Cls Loss", fmt=":6.2f")
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
        for prec_rec_f1 in self.PrecRecF1_train_totals:
            prec_rec_f1.reset()
            
    def reset_val(self):
        super().reset_val()
        self.PrecRecF1_val.reset()
    
    ######### Data Loading #########  
    # Replace _load_data function in __init__ (depending on self.send_to_device)
    def _load_data_send_to_device(self, iter: ForeverDataIterator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the data from the iterator and sends it to the device.
        Mixup is applied if specified."""
        x, labels = next(iter)[:2]
        x = x.to(self.device)
        labels = labels.to(self.device)
        # Use mixup (if given)
        if self.mixup_fn is not None:
            x, labels = self.mixup_fn(x, labels)
        data_load_time = time.time()
        self.data_time.update(data_load_time - self.last_time_mark)
        self.last_time_mark = data_load_time
        return x, labels
    
    def _load_data_on_device(self, iter: ForeverDataIterator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the data from the iterators. Data is assumed to be on the device already.
        I.e. usage of PrefetchLoader. Mixup is applied inside the PrefetchLoader (if specified)."""
        x, labels = next(iter)[:2]
        data_load_time = time.time()
        self.data_time.update(data_load_time - self.last_time_mark)
        self.last_time_mark = data_load_time
        return x, labels
    
    def _load_data_val(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads the data for validation. For hierarchical models only the main hierarchy level is returned 
        based on `self.model.classifier.test_head_pred_ix`."""
        x, labels = self._load_data(self.val_iter)
        # Filter labels to only be the main hierarchy level for test/validation
        if self.model.classifier.test_head_pred_idx is not None:
            labels = labels[:,self.model.classifier.test_head_pred_idx]
        return x, labels
    
    ######### Forward pass (Only really for typehints) #########
    @overload
    def _forward_train(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:...
    @overload
    def _forward_train(self, data: torch.Tensor) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:...
    def _forward_train(self, data: torch.Tensor) -> Tuple[SINGLE_TRAIN_PRED_TYPE, SINGLE_TRAIN_PRED_TYPE]:
        """Computes forward pass for training data. Returns the predictions and the features.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            Tuple[SINGLE_TRAIN_PRED_TYPE, SINGLE_TRAIN_PRED_TYPE]: Predictions and features.
        """
        y, f = super()._forward(data)
        # Model forward always returns the predictions in a List for each input
        # since we only have a single input we can safely just return the first element
        return y[0], f
    
    ######### Loss Computation ######### 
    def _map_labels(self, labels: torch.Tensor, mode: str, train: bool = True) -> torch.Tensor:
        """Map the (ground truth) labels to based on the dataset descriptors and the specified `mode`.
        Which descriptors and hierarchy levels (i.e. train vs val) are used is based on the `train` parameter.
        Mapping is performed based on `self.hierarchy_levels` or `self.hierarchy_level_val` of the used datasets.
        Available values for `mode` are based on `~PSZS.datasets.transform_target()`
            - 'target': Map the internal prediction index to the target class as in the annotation file
            - 'pred': Map the (original) target class to the internal prediction index
            - 'name': Map the target class to the class name. Needs to be the original target class, not the internal prediction index.
            - 'targetIndex': Map the class id to the index of that class in the target (independent of internal prediction)

        Args:
            labels (torch.Tensor): Labels to be mapped
            mode (str): mode of the mapping is reversed.
            train (bool): Whether the mapping is done for training or validation. Defaults to True.

        Returns:
            torch.Tensor: Mapped ground truth labels.
        """
        if train:
            return transform_target(target=labels, mode=mode, descriptor=self.train_descriptor, 
                                    hierarchy_level=self.hierarchy_level)
        else:
            return transform_target(target=labels, mode=mode, descriptor=self.val_descriptor, 
                                    hierarchy_level=self.hierarchy_level_val)
    
    def _compute_loss_cls(self, 
                          pred: SINGLE_TRAIN_PRED_TYPE, 
                          target: torch.Tensor) -> torch.Tensor:
        loss, _ = self.model.compute_cls_loss(pred, target)
        
        self.meter_cls_loss.update(val=loss.item(), n=self.batch_size)
        return loss
    
    ######### Evaluation/Metric computation ######### 
    @overload
    def _compute_eval_metrics(self, pred: torch.Tensor, target: torch.Tensor, og_labels: torch.Tensor, 
                              features: Optional[torch.Tensor]=None,
                              train: bool = False) -> None:...
    @overload
    def _compute_eval_metrics(self, pred: Sequence[torch.Tensor], target: torch.Tensor, og_labels: torch.Tensor, 
                              features: Optional[torch.Tensor]=None,
                              train: bool = False) -> None:...
    def _compute_eval_metrics(self, 
                              pred: PRED_TYPE, 
                              target: torch.Tensor, 
                              og_labels: torch.Tensor, 
                              features: Optional[torch.Tensor]=None,
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
                                               pred: torch.Tensor,
                                               target: torch.Tensor, 
                                               og_labels: torch.Tensor,
                                               features: Optional[torch.Tensor],
                                               ) -> Tuple[List[float], List[float], List[int],
                                                           List[float], List[float], List[float],
                                                           List[int]]:
        # For accuracy during train we don't have eval classes and thus no need to specify original labels
        (cls_acc_1_total, cls_acc_5_total), num_relevant_total = accuracy(prediction=pred, 
                                                                          target=target)
        
        # For F1 etc. we need the original labels even during train to update the correct classes
        self.PrecRecF1_train_totals[0].update(prediction=pred, 
                                          target=target,)
        prec_total, recall_total, f1_total = self.PrecRecF1_train_totals[0].compute()

        acc1_results = [cls_acc_1_total.item()]
        acc5_results = [cls_acc_5_total.item()]
        num_relevants = [num_relevant_total]
        prec_results = [prec_total.item()]
        recall_results = [recall_total.item()]
        f1_results = [f1_total.item()]
        f1_meter_counts = [self.batch_size, target.size(0)]
        
        return acc1_results, acc5_results, num_relevants, prec_results, recall_results, f1_results, f1_meter_counts
     
       
        
    def _calculate_metrics_train_multiple_output(self, 
                                                 pred: Sequence[torch.Tensor],
                                                 target: torch.Tensor, 
                                                 og_labels: torch.Tensor,
                                                 features: Optional[torch.Tensor],
                                                 ) -> Tuple[List[float], List[float], List[int],
                                                           List[float], List[float], List[float],
                                                           List[int]]:
        # For accuracy during train we don't have eval classes and thus no need to specify original labels
        cls_acc_totals, num_relevant_total = zip(*[accuracy(prediction=pred[i], 
                                                            target=target[i]) 
                                                    for i in range(self.num_head_pred)])
        # cls_[acc_1/acc_5]_total contains accuracy for pred1 and pred2 (e.g. make and model)
        cls_acc_1_totals, cls_acc_5_totals = zip(*cls_acc_totals)
        
        for i in range(self.num_head_pred):
            # For F1 etc. we need the original labels even during train to update the correct classes
            self.PrecRecF1_train_totals[i].update(prediction=pred[i], 
                                                  target=target[i],)
        prec_total, recall_total, f1_total = zip(*[self.PrecRecF1_train_totals[i].compute() 
                                                   for i in range(self.num_head_pred)])
        
        acc1_results = [cls_acc.item() for cls_acc in cls_acc_1_totals]
        acc5_results = [cls_acc.item() for cls_acc in cls_acc_5_totals]
        num_relevants = list(num_relevant_total)
        prec_results = [p.item() for p in prec_total]
        recall_results = [r.item() for r in recall_total]
        f1_results = [f.item() for f in f1_total]
        f1_meter_counts = [self.batch_size]*self.num_head_pred + [target.size(0)]*self.num_head_pred
        
        return acc1_results, acc5_results, num_relevants, prec_results, recall_results, f1_results, f1_meter_counts
    
    @overload
    def _compute_eval_metrics_train(self, pred: torch.Tensor, target: torch.Tensor, 
                                    og_labels: torch.Tensor, features: Optional[torch.Tensor]=None,) -> None:...
    @overload
    def _compute_eval_metrics_train(self, pred: Sequence[torch.Tensor], target: torch.Tensor, 
                                    og_labels: torch.Tensor, features: Optional[torch.Tensor]=None,) -> None:...
    def _compute_eval_metrics_train(self, 
                                    pred: SINGLE_TRAIN_PRED_TYPE, 
                                    target: torch.Tensor, 
                                    og_labels: torch.Tensor, 
                                    features: Optional[torch.Tensor]=None,
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
    
    
    # ######### Evaluation/Metric display and filtering#########     
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
        results.update([(f'Cls loss', self.meter_cls_loss.get_avg())])
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