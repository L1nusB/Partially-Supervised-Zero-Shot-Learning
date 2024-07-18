import torch
import torch.nn as nn

from typing import Optional, Sequence, List

from typeguard import check_type

from PSZS.Classifiers.Heads.heads import calculate_intermediate_feature_sizes, CustomHead

class HierarchicalHead(CustomHead):
    returns_multiple_outputs = True
    num_head_pred_key = 'hierarchy_levels'
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int | Sequence[int], 
                 hierarchy_levels: Optional[int] = None,
                 head_depth : int | Sequence[int] = 1,
                 heads: Optional[Sequence[nn.Module]] = None,
                 head_indices: Optional[Sequence[int]] = None,
                 intermediate_features: Optional[Sequence[int] | Sequence[Sequence[int]]] = None,
                 reduction_factor: float | Sequence[float] | Sequence[Sequence[float]] = 2.0) -> None:
        """Construct a hierarchical head that returns a prediction for each level of the hierarchy.
        The number of hierarchy levels can be specified via hierarchy_levels or inferred from the length of out_features.
        If both are given they must match.
        The depth of each level is specified via head_depth. This can be uniform or specified for each level.
        A depth of 1 is equivalent to a SimpleHead with a single layer.
        For deeper subnetworks intermediate_features can be specified. If not specified they are calculated based on reduction_factor.
        These can be specified as a single value or sequences and get expanded as necessary.

        Args:
            in_features (int): Number of features in the input tensor comming from the backbone.
            out_features (int | Sequence[int]): Number of output features for each hierarchy level.
            hierarchy_levels (Optional[int]): Number of hierarchy levels. Defaults to None.
            head_depth (int | Sequence[int]): Depth of each subhead i.e. each hierarchy level. Defaults to 1.
            heads (Optional[Sequence[nn.Module]]): List of heads to use instead of constructing new ones. \
                Replace a head for a certain hierarchy level (specified via `head_indices`). Defaults to None.
            head_indices (Optional[Sequence[int]]): Indices of the hierarchy level a given head should replace. \
                If not specified the first `len(heads)` levels are replaced. Defaults to None.
            intermediate_features (Optional[Sequence[int] | Sequence[Sequence[int]]]): Intermediate feature sizes if head_depth>1. Defaults to None.
            reduction_factor (float | Sequence[float] | Sequence[Sequence[float]]): Reduction factor for automatic calculation of intermediate_features if head_depth>1. Defaults to 2.0.
            
        .. note::
            [in, out] --> Depth 1
            [in, int1, out] --> Depth 2
            [in, int1, int2, out] --> Depth 3
            [in, int1, int2, int3, out] --> Depth 4
        """
        super(HierarchicalHead, self).__init__(in_features=in_features,
                                               out_features=out_features,
                                               head_depth=head_depth,
                                               intermediate_features=intermediate_features,
                                               reduction_factor=reduction_factor)
        
        # Some sanity checks and setting of class attributes (with types)
        self._validate_and_set_attributes(in_features=in_features, 
                                          out_features=out_features, 
                                          hierarchy_levels=hierarchy_levels,
                                          head_depth=head_depth, 
                                          intermediate_features=intermediate_features,
                                          reduction_factor=reduction_factor)
        
        # We probably do not want inplace here as we don't want to modify the input
        self.act = nn.ReLU(inplace=False)
        
        # For each hierarchy level calculate the intermediate features using intermediate_features and reduction_factor
        # For depth = 1 nothing will be done
        # Store separately as in_features and out_features are added here. (In case someone wants to inspect it later.)
        self.features : List[Sequence[int]] = [calculate_intermediate_feature_sizes(in_features=self.in_features, 
                                                                                    out_features=self.out_features[i], 
                                                                                    head_depth=self.head_depth[i], 
                                                                                    intermediate_features=self.intermediate_features[i],
                                                                                    reduction_factor=self.reduction_factor[i]) for i in range(self.hierarchy_levels)
                                                for i in range(self.hierarchy_levels)]
        
        # For depth = 1, body_layers will be empty
        body_layers = [[nn.Linear(self.features[i][j], self.features[i][j+1]) for j in range(self.head_depth[i]-1)]
                                    for i in range(self.hierarchy_levels)]
        fc_finals = [nn.Linear(self.features[i][-1], self.out_features[i]) for i in range(self.hierarchy_levels)]
        # The activation function is inserted applied after each layer
        # and all layers are flattened into a single list
        # Two step code example:
        # bodies = [[[body, self.act] for body in head] for head in self.heads]
        # fl_bodies = [[l for part in body for l in part] for body in bodies]
        flattened_bodies = [[layer for part in body for layer in part] for body in 
                            [[[body, self.act] for body in layer] for layer in body_layers]]
        # Add final layer to each body
        head_components = [body + [fc_finals[i]] for i, body in enumerate(flattened_bodies)]
        # Temporary storage of the heads before optional replacement
        head_modules = [nn.Sequential(*body) for body in head_components]
        
        # If heads are given replace the heads at the specified indices
        if heads is not None:
            # If no head_indices are given replace the first len(heads) heads
            if head_indices is None:
                head_indices = list(range(len(heads)))
            assert len(heads) == len(head_indices), f"Number of heads ({len(heads)}) and head_indices ({len(head_indices)}) must match"
            for i, idx in enumerate(head_indices):
                head_modules[idx] = heads[i]
            
        # Store the (potentially replaced) heads in the class attribute
        self.heads = nn.ModuleList(head_modules)
        
    @property
    def num_head_pred(self) -> int:
        """Returns the number of heads in the hierarchy."""
        return self.hierarchy_levels
        
    def forward(self, x:torch.Tensor) -> List[torch.Tensor]:
        """Computes forward pass for each head in the hierarchy levels and returns the results as a list of tensors.
        One can not construct a single tensor by stacking the individual results, as each prediction can/will have a different shape.
        """
        y = [x.clone()] * self.hierarchy_levels
        for i in range(self.hierarchy_levels):
            y[i] = self.heads[i](y[i])
        return y
    
    def head_kwargs(**kwargs) -> dict:
        """Returns the head construction parameters
        Note that heads and head_indices are not part of the head parameters.
        Because they are somewhat special and are passed directly to the constructor.
        (e.g. in separated_classifier.py)"""
        head_params = CustomHead.head_kwargs(**kwargs)
        head_params["hierarchy_levels"] = kwargs.get(HierarchicalHead.num_head_pred_key)
        return head_params
    
    def _validate_and_set_attributes(self, 
                                     in_features: int , 
                                     out_features: int | Sequence[int], 
                                     hierarchy_levels: Optional[int]=None,
                                     head_depth: int | Sequence[int] = 1, 
                                     intermediate_features: Optional[Sequence[int] | Sequence[Sequence[int]]]=None,
                                     reduction_factor: float | Sequence[float] | Sequence[Sequence[float]] = 2.0) -> None:
        # Ensure hierarchy_levels is set correctly
        if hierarchy_levels is None:
            assert isinstance(out_features, int)==False, "If hierarchy_levels is not specified, out_features must be a sequence"
            hierarchy_levels = len(out_features)
        assert hierarchy_levels == len(out_features), f"Number of hierarchy levels ({hierarchy_levels}) must match number of out_features ({len(out_features)})"
        if hierarchy_levels == 1:
            print("Hierarchy levels is 1, this is equivalent to a SimpleHead. Use SimpleHead instead.")
        
        # Pad head_depth into a list if it is an integer    
        if isinstance(head_depth, int):
            head_depth = [head_depth] * hierarchy_levels
        assert hierarchy_levels == len(head_depth), f"Number of hierarchy levels ({hierarchy_levels}) must match number of head_depth entries ({len(head_depth)})"
        
        # Pad out_features into a list if it is an integer
        if isinstance(out_features, int):
            out_features = [out_features] * hierarchy_levels
            
        #  Pad intermediate_features into Sequence[Sequence[int]] for each hierarchy level and intermediate feature length.
        if intermediate_features is not None:
            if isinstance(intermediate_features[0], int):
                # Note that hierarchy_levels == len(head_depth) ensuring correct inner lenght of intermediate_features
                # Subtract one as in_features and out_features are not part of intermediate_features
                intermediate_features = [intermediate_features[:d-1] for d in head_depth]
            else:
                check_type(intermediate_features, Sequence[Sequence[int]])
                assert len(intermediate_features) == hierarchy_levels, f"Number of intermediate_features sequences ({len(intermediate_features)}) must match hierarchy levels ({hierarchy_levels})"
                # Ensure each intermediate_features sequence has the correct length
                # Other than for reduction factor we do not allow longer Sequences here and simple cut them off
                # If you specify intermediate feature you must make sure it is correct
                for idx, intermediate_feature in enumerate(intermediate_features):
                    assert len(intermediate_feature) == head_depth[idx], f"Length of intermediate_features sequence ({len(intermediate_feature)}) must match head_depth of that level ({head_depth[idx]})" 
        else:
            # Need to pad this up because of for loop when calculating features. Otherwise error that None is not subscriptable
            intermediate_features = [None] * hierarchy_levels
        
        # Pad intermediate_features into a Sequence[Sequence[float]] for each hierarchy level and intermediate feature length.    
        if isinstance(reduction_factor, float):
            # Reduction factor is the same for all hierarchy levels and intermediate feature levels.
            # Note that hierarchy_levels == len(head_depth) ensuring correct inner lenght of reduction_factor
            reduction_factor = [[reduction_factor] * (hd-1) for hd in head_depth]
        elif isinstance(reduction_factor[0], float):
            # Apply same reduction factor to all hierarchy levels.
            # Cut of based on the maximum head_depth of each level.
            # Note that hierarchy_levels == len(head_depth) ensuring correct inner lenght of reduction_factor
            # Use head_depth-1 as we do not apply reduction factor between input_feature to output_feature i.e. no reduction for depth = 1
            assert len(reduction_factor) >= (max(head_depth)-1), f"Length of reduction_factor ({len(reduction_factor)}) must be >= the maximum head_depth-1 ({max(head_depth)-1})"         
            reduction_factor = [reduction_factor[:hd-1] for hd in head_depth]
        else:
            check_type(reduction_factor, Sequence[Sequence[float]])
            # Note that hierarchy_levels == len(head_depth)
            assert len(reduction_factor) >= hierarchy_levels, f"Number of reduction factor sequences ({len(reduction_factor)}) must be >= hierarchy_levels ({hierarchy_levels})"
            # Ensure each reduction_factor sequence has the correct length
            reduction_factor = [rf[:hd-1] for (rf, hd) in zip(reduction_factor[:hierarchy_levels], head_depth)]
        
        # Set class attributes with types
        self.hierarchy_levels : int = hierarchy_levels
        self.head_depth : Sequence[int] = head_depth
        self.in_features : int = in_features
        self.out_features : Sequence[int] = out_features
        self.intermediate_features : Sequence[Sequence[int]] = intermediate_features
        self.reduction_factor : Sequence[Sequence[float]] = reduction_factor