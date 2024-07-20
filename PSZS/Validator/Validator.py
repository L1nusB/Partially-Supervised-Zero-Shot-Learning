from collections import OrderedDict
from contextlib import suppress

import time
from typing import Iterable, Optional, Sequence, Tuple, Callable
import warnings

import torch
from torch.utils.data import DataLoader

from timm.data.loader import PrefetchLoader

from PSZS.Models import CustomModel
from PSZS.Utils.utils import NativeScalerMultiple
from PSZS.Utils.dataloader import ForeverDataIterator
from PSZS.Utils.meters import StatsMeter, ProgressMeter, _BaseMeter, DynamicStatsMeter
from PSZS.Utils.evaluation import PrecisionRecallF1, accuracy, accuracy_hierachy
from PSZS.datasets import transform_target, DatasetDescriptor
from PSZS.Utils.io import Logger
from PSZS.Metrics import ConfusionMatrix

class Validator():
    def __init__(self,
                 model: CustomModel,
                 device: torch.device,
                 batch_size: int,
                 eval_classes: Iterable[int],
                 logger: Optional[Logger] = None,
                 metrics: Sequence[str] = ['acc@1', 'f1'],
                 dataloader: Optional[DataLoader] = None,
                 send_to_device: bool = True,
                 print_freq: int = 100,
                 loss_scaler: Optional[NativeScalerMultiple] = None,
                 amp_autocast = suppress,
                 result_suffix: Optional[str] = None,
                 create_report: bool = False,
                 hierarchy_level_names: Optional[Sequence[str]] = None,
                 **confmat_params,
                 ):
        self.model = model
        self.model.eval() # For Validation the model should always be in eval mode
        self.device = device
        self.batch_size = batch_size
        self.eval_classes = eval_classes
        self.logger = logger
        self.send_to_device = send_to_device
        self.print_freq = print_freq
        self.loss_scaler = loss_scaler
        self.amp_autocast = amp_autocast
        self.dataloader = dataloader
        # Assume that the dataset is ConcatDataset and thus has a descriptor attribute
        self.dataset_descriptor : DatasetDescriptor = getattr(self.dataloader.dataset, 'descriptor', None)
        # For hierarchical heads the load data function already reduces the labels to the main class
        # so no need to check for it/handle differently
        self.hierarchy_level = getattr(self.dataloader.dataset, 'main_class_index', -1)
        if hierarchy_level_names is None:
            self.hierarchy_level_names = self.dataset_descriptor.hierarchy_level_names
        else:
            assert len(hierarchy_level_names) == len(self.dataset_descriptor.hierarchy_level_names), \
                f"Hierarchy level names ({len(hierarchy_level_names)}) must match the number of hierarchy levels in the dataset descriptor ({len(self.dataset_descriptor.hierarchy_level_names)})."
            self.hierarchy_level_names = hierarchy_level_names
        self.main_metric_field = self.hierarchy_level_names[self.hierarchy_level]
        self._create_meters()
        self.metrics = metrics
        # self.metric_meters is already set via metrics setter and filtered to only contain supported metrics
        if len(self.metric_meters) == 0:
            warnings.warn("No (supported) metrics specified. At least one metric must be specified.")
        self.send_to_device = send_to_device
        
        self.PrecRecF1 = PrecisionRecallF1(num_classes=self.model.classifier.effective_head_pred_size.max(),
                                               topk=(1,5), device=self.device,
                                               evalClasses=self.eval_classes)
        if self.dataset_descriptor is not None:
            class_names = self.dataset_descriptor.classes[self.hierarchy_level]
        else:
            class_names = None
        self.confusion_matrix = ConfusionMatrix(num_classes=self.model.classifier.effective_head_pred_size.max(),
                                                class_names=class_names)
        self.result_suffix = "" if result_suffix is None else result_suffix
        self.create_report = create_report
        self.confmat_params = confmat_params
        
    
    @property
    def reduced_confmat_params(self) -> dict:
        params = {
            'last_relevant': -len(self.eval_classes),
            'path': getattr(self.confmat_params, 'reduced_conf_path', 'reducedConfusionMatrix'),
            'out_dir': self.out_dir,
            'sheet_name': self.result_suffix,
            'misclassification_threshold': getattr(self.confmat_params, 'reduced_conf_thres', 0.2),
            'min_predictions': getattr(self.confmat_params, 'reduced_conf_min_pred', 2),
            'show_class_names': True,
            'class_names': None, # Gets inferred automatically from self.confusion_matrix
            'secondary_info': self.dataset_descriptor.fine_coarse_name_map[0] if self.dataset_descriptor is not None else None,
        }
        return params
    
    @property
    def out_dir(self) -> str:
        """Property to allow not specifying a logger and still be able to access the out_dir attribute."""
        if self.logger is not None:
            return self.logger.out_dir
        else:
            return ""
        
    @property
    def dataloader(self) -> DataLoader:
        return self._dataloader
    @dataloader.setter
    def dataloader(self, dataloader):
        """Both set the dataloader as well as setting number of iterations,
        whether to send data to device, and updating the progress bar to reflect the new dataloader"""
        self._dataloader = dataloader
        self.num_iters = len(dataloader)
        if isinstance(dataloader, PrefetchLoader):
            self.send_to_device = False
        else:
            self.send_to_device = True
        # Set the progress bar
        if getattr(self, "_progress_bar", None) is not None and self.progress_bar is not None:
            self.progress_bar.set_num_batches(self.num_iters)
    
    @property 
    def send_to_device(self) -> Callable[[ForeverDataIterator], Tuple[torch.Tensor, torch.Tensor]]:
        # Avoid checks on every data load call
        return self._send_to_device
    @send_to_device.setter
    def send_to_device(self, send_to_device):
        """Adjust the data loading function based on whether data should be sent to device or not"""
        self._send_to_device = send_to_device
        if send_to_device:
            self._load_data = self._load_data_send_to_device
        else:
            self._load_data = self._load_data_on_device
            
    @property
    def progress_bar(self) -> ProgressMeter:
        return self._progress_bar
    @progress_bar.setter
    def progress_bar(self, progress_bar):
        self._progress_bar = progress_bar
        
    @property
    def metrics(self) -> Sequence[str]:
        return self._metrics
    @metrics.setter
    def metrics(self, metrics: Sequence[str]):
        """Also update the metric meters based on the metrics set"""
        self._metrics = [m.lower() for m in metrics]
        meters = []
        if 'acc@1' in self._metrics:
            meters.append(self._meter_cls_acc_1)
        if 'acc@5' in self._metrics:
            meters.append(self._meter_cls_acc_5)
        if 'precision' in self._metrics:
            meters.append(self._meter_precision)
        if 'recall' in self._metrics:
            meters.append(self._meter_recall)
        if 'f1' in self._metrics:
            meters.append(self._meter_f1)
        if 'f1@5' in self._metrics:
            meters.append(self._meter_f1_5)
        self.metric_meters = meters
        
    @property
    def metric_meters(self) -> Sequence[_BaseMeter]:
        return self._meters
    @metric_meters.setter
    def metric_meters(self, meters):
        self._meters = meters
    
    @property
    def cls_acc_1(self) -> float:
        return self._meter_cls_acc_1.get_avg(self.main_metric_field)   
    @property
    def cls_acc_5(self) -> float:
        return self._meter_cls_acc_5.get_avg(self.main_metric_field) 
    @property
    def precision(self) -> float:
        return self._meter_precision.get_last() 
    @property
    def recall(self) -> float:
        return self._meter_recall.get_last() 
    @property
    def f1(self) -> float:
        return self._meter_f1.get_last() 
    @property
    def f1_5(self) -> float:
        return self._meter_f1_5.get_last() 
    
    @property
    def metric_dict(self) -> OrderedDict[str, float]:
        metrics = OrderedDict()
        for lvl in self.hierarchy_level_names:
            if 'acc@1' in self.metrics:
                metrics[f'Acc@1_{lvl}'] = self._meter_cls_acc_1.get_avg(lvl)
            if 'acc@5' in self.metrics:
                metrics[f'Acc@5_{lvl}'] = self._meter_cls_acc_5.get_avg(lvl)
        if 'precision' in self.metrics:
            metrics.update([('Precision', self._meter_precision.get_last())])
        if 'recall' in self.metrics:
            metrics.update([('Recall', self._meter_recall.get_last())])
        if 'f1' in self.metrics:
            metrics.update([('F1', self._meter_f1.get_last())])
        if 'f1@5' in self.metrics:
            metrics.update([('F1@5', self._meter_f1_5.get_last())])
        return metrics
    
    @property
    def metric_str(self) -> str:
        return " | ".join([f'{name}: {value:3.2f}' for name,value in self.metric_dict.items()])
    
        
    def _create_meters(self) -> None:
        # Batch meter is always available
        self.batch_time = StatsMeter.get_stats_meter_time('Batch Time', ':5.2f')
        
        # Metric meters are handeled via property getter only returning relevant meters
        # self._meter_cls_acc_1 = StatsMeter.get_stats_meter_min_max('Acc@1', ':3.2f')
        # self._meter_cls_acc_5 = StatsMeter.get_stats_meter_min_max('Acc@5', ':3.2f')
        # Overwrite accuracy meters to show all hierarchy levels
        self._meter_cls_acc_1 = DynamicStatsMeter.get_stats_meter_min_max("Acc@1", 
                                                                fields=self.hierarchy_level_names, 
                                                                fmt=":3.2f")
        self._meter_cls_acc_5 = DynamicStatsMeter.get_stats_meter_min_max("Acc@5", 
                                                                fields=self.hierarchy_level_names, 
                                                                fmt=":3.2f")
        self._meter_precision = StatsMeter.get_average_meter('Precision', ':3.2f', include_last=True)
        self._meter_recall = StatsMeter.get_average_meter('Recall', ':3.2f', include_last=True)
        self._meter_f1 = StatsMeter.get_average_meter('F1', ':3.2f', include_last=True)
        self._meter_f1_5 = StatsMeter.get_average_meter('F1@5', ':3.2f', include_last=True)
        
    def create_progress_bar(self, prefix: str = "Epoch") -> None:
        if getattr(self, 'dataloader', None) is None:
            warnings.warn("No dataloader set. Create progress bar with zero length.")
            num_iters_batches = 0
        else:
            # When dataloader is set the number of iterations is also set due to property setter
            num_iters_batches = self.num_iters
            
        # Create meters if necessary (batch_time is always created)
        if getattr(self, 'batch_time', None) is None:
            print("No meters found. Creating meters")
            self._create_meters()
            
        self.progress_bar = ProgressMeter(num_batches=num_iters_batches,
                                        meters=[self.batch_time] + self.metric_meters,
                                        batch_meters=[self.batch_time],
                                        exclude_simple_reset=self.metric_meters,
                                        prefix=f"{prefix}: ")
    
    
    def _load_data_send_to_device(self, iter: ForeverDataIterator) -> Tuple[torch.Tensor, torch.Tensor]:
        x, labels = next(iter)[:2]
        x = x.to(self.device)
        labels = labels.to(self.device)
        return x, labels
    
    def _load_data_on_device(self, iter: ForeverDataIterator) -> Tuple[torch.Tensor, torch.Tensor]:
        # Assumes that data already is on device after next
        return next(iter)[:2]
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x, labels = self._load_data(self.dataiter)
        # Filter labels to only be the main hierarchy level for test/validation
        if self.model.classifier.test_head_pred_idx is not None:
            labels = labels[:,self.model.classifier.test_head_pred_idx]
        return x, labels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _map_labels(self, labels: torch.Tensor, mode: str) -> torch.Tensor:
        return transform_target(target=labels, mode=mode, 
                                descriptor=self.dataset_descriptor, 
                                hierarchy_level=self.hierarchy_level)
    
    def compute_metrics(self, 
                        pred: torch.Tensor, 
                        target: torch.Tensor,
                        og_labels: torch.Tensor,
                        ) -> None:
        self.confusion_matrix.update(target=target, prediction=pred)
        cls_accs, num_relevant = accuracy_hierachy(prediction=pred, 
                                                   target=target, 
                                                   hierarchy_map=self.dataset_descriptor.pred_fine_coarse_map,
                                                   originalTarget=og_labels,
                                                   evalClasses=self.eval_classes)
        cls_acc_1, cls_acc_5 = zip(*cls_accs)
        
        
        self.PrecRecF1.update(prediction=pred, 
                              target=target,)
        prec, recall, f1 = self.PrecRecF1.compute()
        
        # Currently no interest for Recall@5 or Precision@5
        prec_1, _ = prec
        recall_1, _ = recall
        f1_1, f1_5 = f1
            
        # Only account for relevant samples when accuracy updating meters
        self._meter_cls_acc_1.update(vals=cls_acc_1, n=num_relevant)
        self._meter_cls_acc_5.update(vals=cls_acc_5, n=num_relevant)
        # For precision, recall and f1 all samples are relevant
        self._meter_precision.update(prec_1.item(), self.batch_size)
        self._meter_recall.update(recall_1.item(), self.batch_size)
        self._meter_f1.update(f1_1.item(), self.batch_size)
        self._meter_f1_5.update(f1_5.item(), self.batch_size)
    
    @torch.no_grad()
    def run(self, prefix: str = "", name: str = "Validation") -> OrderedDict[str, float]:
        tx = name + f" ({prefix})" if prefix != "" else name
        print(f"Starting  {tx}")
        
        if len(self.metric_meters) == 0:
            print("No supported metrics registered. At least one metric must be specified.")
            return
        # Create progress bar for configuration and reset all meters
        self.create_progress_bar(f'{prefix} Epoch')
        self.progress_bar.reset(reset_all=True)
        # Reset F1 metrics object
        self.PrecRecF1.reset()
        # Reset confusion matrix
        self.confusion_matrix.reset()
        print("Create data iterator")
        self.dataiter = iter(self.dataloader)
        
        start_time = end = time.time()
        print("Start validation loop")
        print('-'*40)
        for i in range(self.num_iters):
            # Load data
            x, labels = self.load_data()
            
            # Use autocast for mixed precision
            with self.amp_autocast():
                # Forward Pass/Compute Output
                # Due to model.eval() only the predictios are returned not the features
                y = self.forward(x)

            # Compute Eval Metrics (Accuracy, F1) and Update Meters
            self.compute_metrics(pred=y, 
                                 target=self._map_labels(labels=labels, mode='pred'),
                                 og_labels=labels)
            
            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()
        
            if i % self.print_freq == 0:
                self.progress_bar.display(i)
                self.progress_bar.reset(reset_all=False)
        # Display last results
        self.progress_bar.display(self.num_iters)
        print(self.metric_str)
        print(f'Validation took: {time.time() - start_time}')
        if self.create_report:
            # Divide Acc@5 by 100 to get percentage in range [0,1]
            self.confusion_matrix.create_report(fileName=f'reportFull{self.result_suffix}',
                                                dir=self.out_dir,
                                                show_class_names=True,
                                                overall_stats={'Overall ACC@5': self.cls_acc_5 / 100})
            self.confusion_matrix.create_report(fileName=f'reportEval{self.result_suffix}',
                                                dir=self.out_dir,
                                                start_class=-len(self.eval_classes),
                                                show_class_names=True,
                                                overall_stats={'Overall ACC@5': self.cls_acc_5 / 100})
        
        self.confusion_matrix.save_reduced_conf(**self.reduced_confmat_params)
        return self.metric_dict