import torch
import torch.nn as nn

from typing import Optional, Sequence

from PSZS.Classifiers.Heads.heads import calculate_intermediate_feature_sizes, CustomHead

# Just a simple wrapper for a simple MLP
class SimpleHead(CustomHead):
    returns_multiple_outputs = False
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 head_depth: int = 1, 
                 intermediate_features: Optional[Sequence[int]]=None,
                 reduction_factor: float | Sequence[float] = 2.0) -> None:
        """Simple MLP head with ReLU activation
        Args:
            in_features (int): Dimensionality/Number of input features
            out_features (int): Dimensionality/Number of output features
            head_depth (int): head_depth of the head. Defaults to 1.
            intermediate_features (Optional[Sequence[int]]): Explicit number of intermediate features. \
                                                            If not specified determined based on reduction_factor. \
                                                            Defaults to None.
            reduction_factor (float | Sequence[float]): Reduction factor for automatic determination of intermediate_features. \
                                                    When single float gets applied on each reduction step. \
                                                    When a Sequence is given each reduction step takes respective factor. \
                                                    Defaults to 2.0.
        
        .. note::
            [in, out] --> head_depth 1
            [in, int1, out] --> head_depth 2
            [in, int1, int2, out] --> head_depth 3
            [in, int1, int2, int3, out] --> head_depth 4
        """
        super(SimpleHead, self).__init__(in_features=in_features,
                                         out_features=out_features,
                                         head_depth=head_depth,
                                         intermediate_features=intermediate_features,
                                         reduction_factor=reduction_factor)
        # If a tensor is given, extract the value
        if isinstance(out_features, torch.Tensor):
            out_features = out_features.item()
        self.act = nn.ReLU(inplace=True)
        # [in, out] --> head_depth 1
        # [in, int1, out] --> head_depth 2
        # [in, int1, int2, out] --> head_depth 3
        # [in, int1, int2, int3, out] --> head_depth 4
        
        # Auto split into half as long as still larger than out_features
        # For head_depth = 1 nothing will be done
        features = calculate_intermediate_feature_sizes(in_features=in_features, 
                                                        out_features=out_features, 
                                                        head_depth=head_depth, 
                                                        intermediate_features=intermediate_features,
                                                        reduction_factor=reduction_factor)
        # For head_depth = 1, fc_layers will be empty
        self.fc_layers = nn.ModuleList([nn.Linear(features[i], features[i+1]) for i in range(head_depth-1)])
        self.fc_final = nn.Linear(features[-1], out_features)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for fc in self.fc_layers:
            x = self.act(fc(x))
            
        x = self.fc_final(x)
        return x