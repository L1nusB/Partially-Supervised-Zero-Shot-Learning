import math
import torch.nn as nn
import torch
import warnings

from typing import Optional, Sequence

def calculate_intermediate_feature_sizes(in_features: int, 
                                         out_features: int, 
                                         head_depth: int, 
                                         intermediate_features: Optional[Sequence[int]]=None,
                                         reduction_factor: float | Sequence[float] = 2.0) -> Sequence[int]:
    if intermediate_features is None:
        features = [in_features]
        next_features = in_features
        if isinstance(reduction_factor, float):
            reduction_factor = [reduction_factor] * (head_depth-1)
        for i in range(head_depth-1):
            if math.ceil(next_features / reduction_factor[i]) >= out_features:
                features.append(math.ceil(next_features / 2))
                next_features = math.ceil(next_features / 2)
            else:
                features.append(out_features)
                next_features = out_features
    else:
        assert all([f >= out_features for f in intermediate_features]), "All intermediate features must be larger/equal than out_features"
        assert len(intermediate_features) >= head_depth-1, f"Number of intermediate features ({len(intermediate_features)}) must be at least head_depth-1 ({head_depth-1})"
        if len(intermediate_features) > head_depth-1:
            warnings.warn(f"Warning: Number of intermediate features ({len(intermediate_features)}) is larger than head_depth-1 ({head_depth-1}). Ignoring additional intermediate features.")
            intermediate_features = intermediate_features[:head_depth-1]
        features = [in_features] + intermediate_features + [out_features]
    return features

class CustomHead(nn.Module):
    returns_multiple_outputs = False
    num_head_pred_key = ""
    
    def __init__(self, in_features: int, 
                 out_features: int | Sequence[int], 
                 head_depth: int | Sequence[int] = 1, 
                 intermediate_features: Optional[Sequence[int] | Sequence[Sequence[int]]]=None,
                 reduction_factor: float | Sequence[float] | Sequence[Sequence[float]] = 2.0) -> None:
        super(CustomHead, self).__init__()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Method forward must be implemented by subclass")
    
    @property
    def num_head_pred(self) -> int:
        """Returns the number of heads in the hierarchy.
        Default is 1, but can be overwritten by subclasses."""
        return 1
    
    def head_kwargs(**kwargs) -> dict:
        """Do not use inspect here as hierarchical head is somewhat special"""
        head_params = dict(
            head_depth=kwargs.get("head_depth",1),
            intermediate_features=kwargs.get("intermediate_features"),
            reduction_factor=kwargs.get("reduction_factor",2.0),
        )
        
        return head_params