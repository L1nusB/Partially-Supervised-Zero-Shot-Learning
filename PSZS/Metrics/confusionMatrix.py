from numbers import Number
import os
from typing import Dict, List, Optional, Sequence
import warnings
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .pycmWrapper import pycmConfMat
from PSZS.Utils.io import export_reduced_conf

class ConfusionMatrix:
    def __init__(self, num_classes, class_names: Optional[List[str]] = None) -> None:
        self.mat = None
        self.pycmConfMat: pycmConfMat = None
        self.pred = None
        self.gt = None
        self.num_classes = num_classes
        self.class_names = np.array(class_names) if class_names is not None else None
        
    @property
    def num_classes(self):
        return self._num_classes
    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
        # Create objects if the number of classes changes
        if self.mat is None or self.mat.size(0) != value:
            self.mat = np.zeros((value, value), dtype=np.int32)
            
    def reset(self):
        self.mat = np.zeros_like(self.mat, dtype=np.int32)
    
    @torch.no_grad()    
    def update(self, target: torch.Tensor, prediction: torch.Tensor) -> None:
        assert target.ndim == 1, f'Expected target to have 1 dimension, got {target.ndim}'
        if prediction.ndim != 1:
            # If prediction is one-hot encoded or pure logits, convert to class indices
            assert prediction.ndim == 2, f'Expected prediction to have 1 or 2 dimensions, got {prediction.ndim}'
            prediction = prediction.argmax(dim=1)
        assert target.max() < self.num_classes, f'Given target has more classes ({target.max()}) than confusion matrix object ({self.num_classes-1}).'
        assert prediction.max() < self.num_classes, f'Given prediction has more classes ({prediction.max()}) than confusion matrix object ({self.num_classes-1}).'
        self.mat += confusion_matrix(target.cpu(), prediction.cpu(), labels=range(self.num_classes))
        
    def display(self):
        ConfusionMatrixDisplay(self.mat).plot()
        plt.show()
        
    def compute(self, 
                matrix: Optional[list | np.ndarray] = None,
                classes: Optional[Sequence[str]] = None):
        if matrix is None:
            matrix = self.mat
        if classes is not None:
            assert len(classes) == len(matrix), f'Number of classes ({len(classes)}) and matrix size ({len(matrix)}) do not match.'
        # Suppress warnings from pycm due to divide by zero or overflows etc.
        with warnings.catch_warnings(action='ignore', category=RuntimeWarning):
            warnings.simplefilter('ignore')
            self.pycmConfMat = pycmConfMat(matrix=matrix, classes=classes)
        
    def create_report(self, 
                      fileName: str = 'report',
                      dir: Optional[str] = None,
                      start_class: Optional[int] = None,
                      end_class: Optional[int] = None,
                      show_class_names: bool = False,
                      overall_stats: Optional[Dict[str, Number]] = None):
        """Creates a confusion matrix report as an html format. 
        Based on the pycm library with custom Wrapper."""
        # Remove file extension if present as save_html appends it automatically
        if fileName[-5:] == '.html':
            fileName = fileName[:-5]
        if start_class is None:
            start_class = 0
        if end_class is None:
            end_class = self.num_classes
        
        if show_class_names: # Display class names
            if self.class_names is None:
                warnings.warn('No class names provided. Displaying confusion matrix without class names.')
                cell_size = 2
                classes = None
            else:
                cell_size = 5
                classes = self.class_names[start_class:end_class]
        else:
            cell_size = 2
            classes = None
        self.compute(matrix=self.mat, 
                     classes=self.class_names)
        # If start_class or end_class is not the default values, recompute overall stats
        # especially important because the population needs to be the original population
        if start_class != 0 or end_class != self.num_classes:
            self.pycmConfMat.recompute_overall(class_names=classes)
        if dir is not None:
            os.makedirs(dir, exist_ok=True)
            fileName = os.path.join(dir, fileName)
        self.pycmConfMat.save_html(fileName, 
                                   class_name=classes, 
                                   cell_size=cell_size, 
                                   overall_stats=overall_stats)
        
    def update_class_summary(self, 
                       file: str,
                       dir: Optional[str] = None,
                       start_class: Optional[int] = None,
                       end_class: Optional[int] = None,):
        """Updates the class wise summary of the confusion matrix in the given file."""
        if start_class is None:
            start_class = 0
        if end_class is None:
            end_class = self.num_classes
        
        self.compute(matrix=self.mat,
                     classes=self.class_names)
        if dir is not None:
            os.makedirs(dir, exist_ok=True)
            file = os.path.join(dir, file)
        self.pycmConfMat.write_class_summary(file=file, 
                                             class_names=self.class_names[start_class:end_class])
        
    def reduce_confusion_matrix(self,
                                last_relevant: int,
                                misclassification_threshold: float = 0.2, 
                                min_predictions: int = 2, 
                                class_names: Optional[Sequence[str]] = None,
                                secondary_info: Optional[Dict[str, str]] = None):
        n = self.mat.shape[0]
        
        # Validate class_names if provided
        if class_names is not None:
            if len(class_names) != n:
                raise ValueError(f"Length of class_names ({len(class_names)}) must match the number of classes in conf_matrix ({n})")
        
        # Step 1: Keep only the last num_last_relevant classes
        # reduced_matrix = conf_matrix[-num_last_relevant:, :]
        reduced_matrix = self.mat[last_relevant:, :]
        
        # Step 2: Filter rows based on misclassification threshold
        row_sums = np.sum(reduced_matrix, axis=1)
        # misclassification_rates = 1 - (conf_matrix.diagonal()[-k:] / row_sums)
        misclassification_rates = 1 - (self.mat.diagonal()[last_relevant:] / row_sums)
        rows_to_keep = misclassification_rates > misclassification_threshold
        
        result = []
        original_class_indices = np.arange(n)
        
        def get_class_identifier(idx) -> str:
            main_class = class_names[idx] if class_names is not None else str(idx)
            if secondary_info is not None:
                main_class = f'{main_class}[{secondary_info.get(idx, "unknown")}]'
            return main_class
        
        offset = n + last_relevant if last_relevant < 0 else n - last_relevant
        for i, keep_row in enumerate(rows_to_keep):
            if keep_row:
                row = reduced_matrix[i]
                # Add n-k because i starts at 0 but we only kept the last k rows
                # class_index = original_class_indices[i + n - k]
                class_index = original_class_indices[i + offset]
                
                # Find columns with significant misclassifications
                significant_cols = np.where(row >= min_predictions)[0]
                
                # Create a dictionary for this class with sorted misclassifications
                misclassifications = {
                    get_class_identifier(col): int(row[col])
                    for col in significant_cols
                    if col != class_index
                }
                sorted_misclassifications = dict(sorted(misclassifications.items(), key=lambda x: x[1], reverse=True))
                
                class_dict = {
                    "actual_class": get_class_identifier(class_index),
                    "total_samples": int(row_sums[i]),
                    "correct_predictions": int(row[class_index]),
                    "misclassifications": sorted_misclassifications
                }
                
                
                result.append(class_dict)
        result.sort(key=lambda x: x["actual_class"].lower())
        return result
    
    def save_reduced_conf(self, 
                          last_relevant: int, 
                          path: str='reduced_confusion_matrix.xlsx',
                          out_dir: Optional[str] = None,
                          sheet_name: Optional[str] = None,
                          misclassification_threshold: float = 0.2,
                          min_predictions: int = 2,
                          show_class_names: bool = True,
                          class_names: Optional[Sequence[str]] = None,
                          secondary_info: Optional[Dict[str, str]] = None):
        if path[-5:] != '.xlsx':
            path += '.xlsx'
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, path)
        if show_class_names:
            if class_names is None:
                class_names = self.class_names
        reduced_results = self.reduce_confusion_matrix(last_relevant=last_relevant,
                                                       misclassification_threshold=misclassification_threshold,
                                                       min_predictions=min_predictions,
                                                       class_names=class_names,
                                                       secondary_info=secondary_info)
        export_reduced_conf(reduced_data=reduced_results, 
                            filename=path,
                            sheet_name=sheet_name)