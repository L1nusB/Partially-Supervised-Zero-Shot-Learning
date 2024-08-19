from typing import Any, Dict, Optional, Sequence, Type, overload
import warnings

import torch.nn as nn
import torch
import numpy as np
import numpy.typing as npt

from PSZS.Classifiers.Heads import CustomHead, SimpleHead
from PSZS.Classifiers.custom_classifier import CustomClassifier

PRED_TYPE = Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]]

class MaskedClassifier(CustomClassifier):
    def __init__(self, 
                 num_classes: int | npt.NDArray[np.int_],
                 num_inputs: int,
                 in_features: int,
                 head_type: Type[CustomHead] = SimpleHead,
                 auto_split: bool = True,
                 auto_split_indices: Optional[Sequence[int]] = None,
                 num_classes_pref: str = 'inputs',
                 test_head_pred_idx: Optional[int] = -1,
                 hierarchy_level_names: Optional[Sequence[str]] = None,
                 allow_mix_num_classes: bool = False,
                 allow_non_strict_num_classes_order: bool = True,
                 head_params: dict = {},
                 ) -> None:
        """
        Args:
            num_classes (int | np.ndarray): Number of classes for each head. Gets expanded/must match shape (num_inputs, num_head_predictions).
            num_inputs (int): Number of inputs to the classifier during training forward pass.
            in_features (int): Number of features before the head layer produces by the backbone.
            head_type (Type[CustomHead]): Type of head to use. (default: ``SimpleHead``)
            num_features (Optional[int]): Number of features before the head layer (Can be determined from backbone)
            auto_split (bool): Whether to automatically split the incoming features into num_inputs parts. (default: True)
            auto_split_indices (Optional[Sequence[int]]): Indices to split the features when auto_split is True
            num_classes_pref (str): Whether given num_classes represent values for each input or each head prediction. \
                    Only relevant if num_classes has len(shape)=1 and num_inputs==in_features. Either 'inputs' or 'heads' (default: 'inputs')
            test_head_pred_idx (Optional[int]): Index of the prediction the head produce to use for testing/validation. \
                    For heads returning single outputs `CustomClassifier` constructor sets this to None. \
                    Must be provided when Head returns multiple outputs. (default: -1)
            allow_mix_num_classes (bool): Allow the largest number of classes to be a mix of the provided num_classes. (default: False)
            allow_non_strict_num_classes_order (bool): Allow the largest number of classes to not match the first entry in num_classes. (default: True)
            head_params (dict): Keyword arguments for the head
            
        Inputs:
            - f (torch.Tensor): Input data fed to `heads` obtained from the backbone
            
        Outputs:
            - predictions: Classifier's predictions for each head
        """
        super(MaskedClassifier, self).__init__(num_classes=num_classes,
                                               num_inputs=num_inputs,
                                               in_features=in_features,
                                               head_type=head_type,
                                               auto_split=auto_split,
                                               auto_split_indices=auto_split_indices,
                                               num_classes_pref=num_classes_pref,
                                               test_head_pred_idx=test_head_pred_idx,
                                               hierarchy_level_names=hierarchy_level_names,
                                               head_params=head_params,
                                               )
        self.allow_mix_num_classes = allow_mix_num_classes
        self.allow_non_strict_num_classes_order = allow_non_strict_num_classes_order

        self.head = head_type(in_features=self.in_features, 
                              out_features=self.largest_head_num_classes, 
                              **head_params)    

    def forward(self, 
                f: torch.Tensor | Sequence[torch.Tensor], 
                *other_features: torch.Tensor
                ) -> Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]]:
        """Forward pass
        Take the features from the backbone and pass them through the heads
        Features can be passed as a single tensor, sequence of tensors or as multiple individual tensors.
        The number of features must be equal to the number of classes unless a single feature is passed.
        When auto_split is True and a single feature is passed it gets split into num_classes features via tensor_split.
        .. note::
            Apply head to each feature and accumulate in list
            Each list item represents a feature and the corresponding predictions of the head
            [head(f0), head(f1), ...]"""
        if len(other_features) > 0:
            assert isinstance(f, torch.Tensor), "When passing multiple features they must be tensors"
            # Add one for the default feature f
            assert len(other_features) == self.num_inputs - 1, f"Number of given features ({len(other_features)+1}) does not match expected inputs: ({self.num_inputs})"
            predictions = [self.head(feature) for feature in ((f,) + other_features)]
        elif isinstance(f, torch.Tensor):
            if self.auto_split:
                f = torch.tensor_split(f, self.auto_split_indices, dim=0)
                predictions = [self.head(feature) for feature in f]
            else:
                # Note that the features are NOT split here so each prediction has the full batch size dimension
                predictions = [self.head(f) for _ in range(self.num_inputs)]
        elif isinstance(f, Sequence):
            assert len(f) == self.num_inputs, f"Number of given features {len(f)} does not match expected inputs: ({self.num_inputs})"
            predictions = [self.head(feature) for feature in f]
            
        predictions = self.mask_predictions(predictions)
        return predictions
    
    @overload
    def mask_predictions(self,predictions: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:...
    
    @overload
    def mask_predictions(self,predictions: Sequence[Sequence[torch.Tensor]]) -> Sequence[Sequence[torch.Tensor]]:...
    
    def mask_predictions(self, predictions: PRED_TYPE) -> PRED_TYPE:
        """Mask the predictions according to the number of classes.
        Separate function to allow for more sophisticated masking for hierarchies/multiple heads.
        Stacking into single tensor or similar is not possible as the batch sizes of each prediction can differ.
        This leads to potential shape mismatch when trying to stack them into a single tensor.
        """
        if self.head.returns_multiple_outputs:
            # If the head returns multiple outputs, each head output must be masked individually
            return [[head_pred[:, :self.num_classes[input_idx,head_idx]] for head_idx, head_pred in enumerate(prediction)] 
                    for input_idx, prediction in enumerate(predictions)]
        else:
            # If the head returns a single output, num_classes has shape (num_inputs,1) (e.g. [[281],[251]])
            # Thus remove extra dimension via squeeze (e.g. [281, 251])
            unmasked_classes = self.num_classes.squeeze()
            return [prediction[:, :unmasked_classes[idx]] for idx, prediction in enumerate(predictions)]