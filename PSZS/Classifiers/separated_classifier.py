from typing import Dict, Any, Optional, Sequence, Type

import torch
import torch.nn as nn

import numpy as np
import numpy.typing as npt

from PSZS.Classifiers.Heads import CustomHead, SimpleHead, HierarchicalHead
from PSZS.Classifiers.custom_classifier import CustomClassifier

class SeparatedClassifier(CustomClassifier):
    def __init__(self, 
                 num_classes: int | npt.NDArray[np.int_],
                 num_inputs: int,
                 in_features: int,
                 head_type: Type[CustomHead] = SimpleHead,
                 auto_split: bool = True,
                 auto_split_indices: Optional[Sequence[int]] = None,
                 num_classes_pref: str = 'inputs',
                 test_head_idx: int = 0,
                 test_head_pred_idx: Optional[int] = -1,
                 hierarchy_level_names: Optional[Sequence[str]] = None,
                 share_coarse_heads: bool = True,
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
            test_head_idx (int): Index of the head to use for testing/validation (default: 0)
            test_head_pred_idx (Optional[int]): Index of the prediction the head produce to use for testing/validation. \
                    For heads returning single outputs `CustomClassifier` constructor sets this to None. \
                    Must be provided when Head returns multiple outputs. (default: -1)
            head_params (dict): Keyword arguments for the head
            
        Inputs:
            - f (torch.Tensor): Input data fed to `heads` obtained from the backbone
            
        Outputs:
            - predictions: Classifier's predictions for each head
        """
        super(SeparatedClassifier, self).__init__(num_classes=num_classes,
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
        
        # If head does not return multiple outputs we have shape (num_inputs, 1)
        # These heads need a scalar value for out_features and thus we remove thus single dimension
        if head_type.returns_multiple_outputs == False:
            # Specify axis so that only the single dimension is removed
            head_num_classes = self.num_classes.squeeze(axis=1)
        else:
            head_num_classes = self.num_classes
        
        # If hierarchical head and a shared head for some hierarchy levels
        # construct this head here and pass it via hierarchical_params    
        # Done by a temporary head to get the heads for the hierarchical head
        # Only necessary for separated_classifier as masked or default share by default
        hierarchical_params = {}
        if self.head_type.returns_multiple_outputs:
            if share_coarse_heads:
                print("Hierarchical Head with shared heads for coarse hierarchy levels.")
                # Only for type annotations
                head_type : Type[HierarchicalHead] = head_type
                temp_heads = head_type(self.in_features, head_num_classes[0], **head_params).heads
                hierarchical_params = {'heads': temp_heads[:-1]}
            else:
                print("Hierarchical Head with separated heads for all hierarchy levels.")
            
        self.head = torch.nn.ModuleList([head_type(self.in_features, num_class, **head_params, **hierarchical_params) 
                                         for num_class in head_num_classes])
        self.test_head_idx = test_head_idx
        
    @property
    def val_test_head(self) -> CustomHead:
        # Can be overwritten by subclasses if multiple heads are used during training (e.g. SeparatedClassifier)
        return self.head[self.test_head_idx]
        
    def val_test_state_dict(self) -> Dict[str, Any]:
        """Return state dict for the classifier for only validation/testing part.
        Only return the state dict of the head specified by test_head_idx.
        """
        # Append head.0. to the keys to ensure that the head is correctly loaded
        return {f"head.0.{k}":v for k,v in self.head[self.test_head_idx].state_dict().items()}
        
    # def _get_effective_head_pred_size(self) -> torch.Tensor: 
    #     """Currently only a wrapper around num_classes.
    #     Also turns it into a tensor.
    #     Not sure how necessary this is. Mainly used in Base_Multiple for eval train components of F1"""
    #     return torch.tensor(self.num_classes)
    #     # if self.head_type.returns_multiple_outputs == False:
    #     #     return self.num_classes.squeeze(axis=1)
    #     # else:
    #     #     return self.num_classes[:, self.test_head_pred_idx]  
        
    def forward(self, 
                f: torch.Tensor | Sequence[torch.Tensor], 
                *other_features: torch.Tensor) -> Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]]:
        """Forward pass
        Take the features from the backbone and pass them through the heads
        Features can be passed as a single tensor, sequence of tensors or as multiple individual tensors.
        The number of features must be equal to the number of heads unless a single feature is passed.
        When auto_split is True and a single feature is passed it gets split into len(self.num_classes) features via tensor_split."""
        if len(other_features) > 0:
            assert isinstance(f, torch.Tensor), "When passing multiple features they must be tensors"
            # Add one for the default feature f
            assert len(other_features) == self.num_inputs - 1, f"Number of given features ({len(other_features)+1}) does not match expected inputs: ({self.num_inputs})"
            # len(heads) == self.nm_features == len(other_features) + 1 --> We can use zip() instead of indexing
            predictions = [head(feature) for head, feature in zip(self.head, (f,) + other_features)]
        elif isinstance(f, torch.Tensor):
            if self.auto_split:
                f = torch.tensor_split(f, self.auto_split_indices, dim=0)
                predictions = [head(feature) for head, feature in zip(self.head, f)]
            else:
                # Note that the features are NOT split here so each prediction has the full batch size dimension
                predictions = [head(f) for head in self.head]
        elif isinstance(f, Sequence):
            assert len(f) == self.num_inputs, f"Number of given features {len(f)} does not match expected inputs: ({self.num_inputs})"
            predictions = [head(feature) for head, feature in zip(self.head, f)]
        # Don't rejoin predictions but keep them separate
        # Joining would require same shape and thus padding which would 
        # be confusion and might lead to wrong backward pass
        # e.g. padding it with zeros is questionable
        return predictions