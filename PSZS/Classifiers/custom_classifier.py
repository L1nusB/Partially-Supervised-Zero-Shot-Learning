from typing import Any, Dict, Optional, Sequence, Type
import warnings
import inspect

import torch.nn as nn
import torch
import numpy as np
import numpy.typing as npt

from PSZS.Classifiers.Heads import CustomHead, SimpleHead

class CustomClassifier(nn.Module):
    def __init__(self, 
                 num_classes: int | npt.NDArray[np.int_],
                 num_inputs: int,
                 in_features: int,
                 head_type: Type[CustomHead] = SimpleHead,
                 auto_split: bool = True,
                 auto_split_indices: Optional[Sequence[int]] = None,
                 test_head_pred_idx: Optional[int] = None,
                 num_classes_pref: str = 'inputs',
                 head_params: dict = {},
                 ) -> None:
        """
        Args:
        num_classes (int | np.ndarray): Number of classes for each head. Gets expanded/must match shape (num_inputs, num_head_predictions).
        num_inputs (int): Number of inputs to the classifier during training forward pass.
        in_features (int): Number of features before the head layer produces by the backbone.
        head_type (Type[CustomHead]): Type of head to use. (default: ``SimpleHead``)
        auto_split (bool): Whether to automatically split the incoming features into num_inputs parts. (default: True)
        auto_split_indices (Optional[Sequence[int]]): Indices to split the features when auto_split is True
        num_classes_pref (str): Whether given num_classes represent values for each input or each head prediction. \
            Only relevant if num_classes has len(shape)=1 and num_inputs==in_features. Either 'inputs' or 'heads' (default: 'inputs')
        test_head_pred_idx (Optional[int]): Index of the prediction the head produce to use for testing/validation. \
                                            Must be provided when Head returns multiple outputs. (default: None)
        head_params (dict): Keyword arguments for the head
        
        """
        super(CustomClassifier, self).__init__()
        
        num_head_pred = head_params[head_type.num_head_pred_key] if head_type.returns_multiple_outputs else 1
        self.head_type = head_type
        if head_type.returns_multiple_outputs:
            assert test_head_pred_idx is not None, "Head returns multiple outputs. test_head_pred_idx must be provided."
            if head_params[head_type.num_head_pred_key] is None:
                warnings.warn("Head returns multiple outputs but desired number of predictions is not provided. "
                              f"Trying to infer all available predictions shape of the dataset based on num_classes ({num_classes}). "
                              "To change the number of predictions set the desired number via "
                              f"--classifier-kwargs ({head_type.num_head_pred_key}=[num_predictions])")
                match len(num_classes.shape):
                    case 0:
                        raise ValueError(f"num_classes ({num_classes}) is a single number. Cannot infer number of predictions.")
                    case 1:
                        # E.g. [68, 281]
                        if num_inputs != 1:
                            warnings.warn("num_inputs != 1 but num_classes is single array. "
                                          "Try to use len(num_classes) as number of predictions "
                                          "and expand to all inputs.")
                        num_head_pred = len(num_classes)
                        num_classes = np.full((num_inputs, num_head_pred), num_classes)
                    case 2:
                        num_head_pred = num_classes.shape[1]
                assert len(num_classes.shape) == 2, ("num_classes must be of shape (num_inputs, num_head_predictions) "
                                                     f"({num_inputs}, {num_head_pred}) "
                                                     "after inferring num_head_predictions. "
                                                     f"Got shape {num_classes.shape}.")
                num_head_pred = num_classes.shape[1]
                print(f"Using {num_head_pred} predictions based on num_classes.")
        else:
            # If head doesn't return multiple outputs, test_head_pred_idx needs to be None to avoid wrong selection of predictions
            test_head_pred_idx = None
            
        self.num_head_pred = num_head_pred
        
        self.num_inputs = num_inputs
        
        self.num_classes_pref = num_classes_pref
        assert self.num_classes_pref in ['inputs', 'heads'], "num_classes_pref must be 'inputs' or 'heads'"
        
        # Ensure num_classes is a np.array
        if isinstance(num_classes, int):
            num_classes = np.array(num_classes)
        # Expand num_classes to appropriate shape (num_inputs, num_head_predictions)
        match len(num_classes.shape):
            case 0:
                # Expand single value to all inputs and head predictions
                num_classes = np.full((num_inputs, num_head_pred), num_classes)
            case 1:
                if num_inputs == num_head_pred:
                    print("num_inputs == num_head_predictions. "
                          f"Construct num_classes based on num_classes_pref: ({self.num_classes_pref})"
                          "To change this behavior you can set num_classes_pref to 'heads' or 'inputs'")
                # Values for each input --> Expand to all head predictions
                if num_classes.shape[0] == num_inputs and self.num_classes_pref == 'inputs':
                    # [a1, a2, a3] -> [[a1, a1, a1], [a2, a2, a2], [a3, a3, a3]]
                    num_classes = np.full((num_head_pred, num_inputs), num_classes).T
                # Values for each head prediction --> Expand to all inputs
                # No need to explicitly check num_classes_pref here again. Important only in case num_inputs==num_head_pred
                # and then it is already assured that it is not 'inputs'
                # Otherwise it is irrelevant
                elif num_classes.shape[0] == num_head_pred:
                    # [a1, a2, a3] -> [[a1, a2, a3], [a1, a2, a3], [a1, a2, a3]]
                    num_classes = np.full((num_inputs, num_head_pred), num_classes)
                else:
                    raise ValueError("num_classes must be of shape (num_inputs,) or (num_head_predictions,)")
            case 2:
                # Already in correct shape
                assert num_classes.shape == (num_inputs, num_head_pred), f"num_classes must be of shape ({num_inputs}, {num_head_pred}), but is {num_classes.shape}"
        self.num_classes : npt.NDArray[np.int_] = num_classes
        
        self.in_features = in_features
            
        if self.num_inputs == 1:
            print("Only a single input provided. Disabling auto_split.")
            # If only a single class there is nothing to split in the features
            self.auto_split = False
            self.auto_split_indices = None
        else:
            self.auto_split = auto_split
            if auto_split_indices is None:
                # If no split indices are provided do an even split
                print(f"No auto_split_indices provided. Splitting evenly into {len(self.num_classes)} parts.")
                self.auto_split_indices = len(self.num_classes)
            else:
                assert len(auto_split_indices) == len(self.num_classes)-1, (f"Number of auto_split_indices ({len(self.auto_split_indices)}) "
                                                                    f"must be equal to number of classes-1 ({len(self.num_classes)-1})")
                self.auto_split_indices = auto_split_indices
        
        # Required for supporting hierarchical heads
        self.test_head_pred_idx = test_head_pred_idx
        # Needs to be set before calling _get_effective_head_pred_size as it uses this attribute
        self.largest_head_num_classes = self.get_largest_head_num_classes()
        self.smallest_head_num_classes = self.get_smallest_head_num_classes()
        # Can be changed in subclasses otherwise it is a wrapper around num_classes
        self.effective_head_pred_size = self._get_effective_head_pred_size()
        
        # Define class attribute here already to make it accessible by all CustomClassifiers
        self.head : CustomHead | nn.ModuleList[CustomHead] 
        
    @property
    def val_test_head(self) -> CustomHead:
        # Can be overwritten by subclasses if multiple heads are used during training (e.g. SeparatedClassifier)
        return self.head
    
    @property
    def returns_multiple_outputs(self) -> bool:
        return self.head_type.returns_multiple_outputs
    
    @property
    def largest_num_classes_test_lvl(self) -> int:
        """Returns the largest number of classes across all inputs
        for the head prediction relevant for the final prediction
        (that is also used during validation/testing).
        For heads returning single outputs this corresponds to just 
        largest_head_num_classes. For heads returning multiple outputs
        only the relevant head prediction based on test_head_pred_idx is considered."""
        if self.head_type.returns_multiple_outputs:
            # Pick out the head prediction index to use for testing
            # and convert to scalar value
            return self.largest_head_num_classes[self.test_head_pred_idx].item()
        else:
            # If head only returns a single output, we either already 
            # have a scalar value or a tensor containing only a single value
            return self.largest_head_num_classes.item()
        
    @property
    def smallest_num_classes_test_lvl(self) -> int:
        """Returns the smallest number of classes across all inputs
        for the head prediction relevant for the final prediction
        (that is also used during validation/testing).
        For heads returning single outputs this corresponds to just 
        smallest_head_num_classes. For heads returning multiple outputs
        only the relevant head prediction based on test_head_pred_idx is considered."""
        if self.head_type.returns_multiple_outputs:
            # Pick out the head prediction index to use for testing
            # and convert to scalar value
            return self.smallest_head_num_classes[self.test_head_pred_idx].item()
        else:
            # If head only returns a single output, we either already 
            # have a scalar value or a tensor containing only a single value
            return self.smallest_head_num_classes.item()
    
    def val_test_state_dict(self) -> Dict[str, Any]:
        """Return state dict for the classifier for only validation/testing part.
        """
        # Append head.0. to the keys to ensure that the head is correctly loaded
        return {f"head.0.{k}":v for k,v in self.val_test_head.state_dict().items()}
    
    def _get_effective_head_pred_size(self) -> torch.Tensor:
        return torch.tensor(self.num_classes)
        
    def forward(self, 
                f: torch.Tensor | Sequence[torch.Tensor], 
                *other_features: torch.Tensor) -> Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]]:
        raise NotImplementedError("Forward method must be implemented in subclasses")
    
    def forward_test(self, f: torch.Tensor) -> torch.Tensor:
        """Forward pass for testing/validation
        Take the features from the backbone and pass them through the head.
        If a test_head_pred_idx is provided only return the corresponding prediction.
        """
        # This ensures that only a single tensor is returned and not a list of all hierarchical predictions.
        if self.test_head_pred_idx is not None:
            return self.val_test_head(f)[self.test_head_pred_idx]
        else:
            return self.val_test_head(f)
    
    def get_largest_head_num_classes(self) -> torch.Tensor:
        # After super num_classes has shape (num_inputs, num_head_predictions)
        # Determine the entry in the num_classes array with the largest value for each head prediction
        # 
        # [[2],[5],[3]] --> [5]   |  [[5,23],[3,17],[4,20]] --> [5,23]  |  [[5,17],[3,25],[4,20]] --> [5,23]
        major_class : npt.NDArray[np.int_] = self.num_classes.max(0)
        # Check if major_class is even an existing entry in num_classes
        # otherwise no clear maximal number of classes exists across all heads    
        # .tolist() is required as exact existence check does not work directly with numpy arrays
        if major_class.tolist() not in self.num_classes.tolist():
            if getattr(self, 'allow_mix_num_classes', False):
                warnings.warn(f"Largest found classes ({major_class}) is not an entry in num_classes.")
            else:
                raise ValueError(f"Largest found classes ({major_class}) is not an entry in num_classes. "\
                                    "Enable allow_mix_num_classes to ignore this.")    
        # Check if the first entry in num_classes corresponds to the largest number of classes
        elif all(major_class != self.num_classes[0]):
            if getattr(self, 'allow_non_strict_num_classes_order', True):
                warnings.warn(f"Largest class ({major_class}) does not match the first class ({self.num_classes[0]}).")
            else:
                raise ValueError(f"Largest class ({major_class}) does not match the first class ({self.num_classes[0]}). "\
                                    "Enable allow_non_strict_num_classes_order to ignore this.")
        
        # Remove extra dimension if only one head prediction is used to create scalar
        # [5] --> 5  |  [5,23] --> [5,23]  
        return torch.from_numpy(major_class.squeeze())
    
    def get_smallest_head_num_classes(self) -> torch.Tensor:
        # After super num_classes has shape (num_inputs, num_head_predictions)
        # Determine the entry in the num_classes array with the smallest value for each head prediction
        # 
        # [[2],[5],[3]] --> [2]   |  [[5,23],[3,17],[4,20]] --> [3,17]  |  [[5,17],[3,25],[4,20]] --> [3,17]
        minor_class : npt.NDArray[np.int_] = self.num_classes.min(0)
        # Check if minor_class is even an existing entry in num_classes
        # otherwise no clear minimal number of classes exists across all heads    
        # .tolist() is required as exact existence check does not work directly with numpy arrays
        if minor_class.tolist() not in self.num_classes.tolist():
            if getattr(self, 'allow_mix_num_classes', False):
                warnings.warn(f"Smallest found classes ({minor_class}) is not an entry in num_classes.")
            else:
                raise ValueError(f"Smallest found classes ({minor_class}) is not an entry in num_classes. "\
                                    "Enable allow_mix_num_classes to ignore this.")    
        # Remove extra dimension if only one head prediction is used to create scalar
        # [5] --> 5  |  [5,23] --> [5,23]  
        return torch.from_numpy(minor_class.squeeze())
    
    
    
    @classmethod
    def classifier_kwargs(cls, head_params: dict, **kwargs) -> dict:
        # Head params are treated separately as they are required for the head
        # but not for the classifier itself
        classifier_kwargs = {'head_params': head_params}
        # Using the inspect module to get the arguments of the constructor
        # This way we can filter out the relevant arguments without hardcoding them
        # and thus no need to update this method when the constructor changes
        # or in subclasses
        arguments = inspect.getfullargspec(cls.__init__).args
        # Only add if given otherwise use default from constructor
        for arg in arguments:
            if arg in kwargs:
                classifier_kwargs[arg] = kwargs[arg]
        return classifier_kwargs