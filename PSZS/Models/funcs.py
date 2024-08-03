import math
from numbers import Number
from typing import Callable, Sequence, Optional, Tuple

import torch
import torch.nn.functional as F

def construct_weights(weights: Sequence[float], 
                      length: Optional[int] = None, 
                      normalized: bool = True,
                      same_weight: bool = False,
                      default_weight: float = 0.1) -> torch.Tensor:
        """
        Create a weight vector based on the specified ratios. 
        Weight components are set to the given ratios and remaining components are filled with appropriate values.
        Additional components can be specified by either setting `length` to a value greater than the number of ratios
        or by setting `normalized` to False and have `sum(weights)` less than 1 in which case one additional component is added.
        Additional components are added in the front of the final weights to represent coarse to fine weight components.
        If `normalized` is True, the weights are normalized/constructed to sum to 1 with equal weight being distributed to the remaining components.
        Otherwise new weights are set to `default_weight`.
        
        Parameters:
            weights (Sequence[float]): 
                The values for the initial components.
            length (int, optional): 
                The total length of the weight vector. If `None`, the length will be the length of ratios if `normalize` is `False` or `weights`
                sum up to 1. If `normalized` is `True` and the sum of weights is less than 1, the length will be the length of ratios + 1.
            normalized (bool): 
                Normalize the weights to sum to 1 and other constraints. Defaults to True.
            same_weight (bool):
                Set all remaining additional components to the same value. Otherwise scale by depth of component. Defaults to False.
            default_weight (float): 
                The default weight value to use for unspecified components if `normalized` is False.
        
        Returns:
            torch.Tensor: A weight vector with the specified weights.
        """
        if normalized:
            if any(w < 0 or w > 1 for w in weights):
                raise ValueError("All weights must be between 0 and 1.")
            if sum(weights) > 1:
                raise ValueError(f"Sum of weights must not exceed 1. Got {sum(weights)}")
        
        num_weights = len(weights)
        if length is None:
            # If normalized ensure weights can be expanded to 1 by adding one more component
            if normalized and sum(weights) < 1:
                length = num_weights + 1
            else:
                # Either normalized=False or sum(weights) == 1
                length = num_weights
        if length < num_weights:
            raise ValueError(f"Length/Number of weights {(length)} be at least the number of ratios {(num_weights)}.")
        
        # Create the weight vector
        weight_vector = torch.zeros(length)
        
        # Set the 'final' components to the given ratios
        for i, ratio in enumerate(weights):
            weight_vector[length - num_weights + i] = ratio
        remaining_length = length - num_weights
        if normalized:
            # Distribute the remaining weight evenly
            # remaining_weight = round((1 - sum(weights)) / remaining_length, 4)
            remaining_weight = round((1 - sum(weights)) / remaining_length, 4)
            # When using normalized weights special scaling is needed
            for i in range(remaining_length):
                # Scale reminaining weight by index if same_weight is False
                weight_vector[i] = remaining_weight if same_weight else remaining_weight * (i*2+1) / remaining_length
        else:
            remaining_weight = default_weight
            # When using default weight directly scale it up by index
            for i in range(remaining_length):
                weight_vector[i] = remaining_weight if same_weight else remaining_weight*(i+1)
        
        
        if normalized:
            assert math.isclose(sum(weight_vector), 1), f"Sum of weight vector is not 1: {sum(weight_vector)}"
        return weight_vector

### Accumulate functions ###
def avg_accum(components: Sequence[torch.Tensor],
              batch_accum_func: Callable[[torch.Tensor], Number]=torch.sum) -> torch.Tensor:
    return sum([batch_accum_func(c) for c in components]) / len(components)

def sum_accum(components: Sequence[torch.Tensor],
              batch_accum_func: Callable[[torch.Tensor], Number]=torch.sum) -> torch.Tensor:
    return sum([batch_accum_func(c) for c in components])

def weighted_accum(components: Sequence[torch.Tensor], 
                  weights: Sequence[Number] | torch.Tensor,
                  batch_accum_func: Callable[[torch.Tensor], Number]=torch.sum) -> torch.Tensor:
    """Computes the weighted sum of the input components. Weights can be given as a list of numbers or a tensor."""
    if isinstance(weights, torch.Tensor) == False:
        weights = torch.Tensor(weights)
    assert len(components) == len(weights), f"Number of components and weights must be equal: {len(components)} != {len(weights)}"
    return sum([batch_accum_func(components[i]) * weights[i] for i in range(len(components))])

def dynamic_weighted_accum(components: Sequence[torch.Tensor],
                           weight_func: Callable[..., Sequence[Number] | torch.Tensor]
                           ) -> torch.Tensor:
    weights = weight_func()
    return weighted_accum(components, weights)

def coarse_sum_accum(components: Sequence[torch.Tensor], fine_index: int, 
                     batch_accum_func: Callable[[torch.Tensor], Number]=torch.sum) -> torch.Tensor:
    # Handle negative indices
    exclude_index = fine_index if fine_index >= 0 else len(components) + fine_index
    return sum([batch_accum_func(c) for i, c in enumerate(components) if i != exclude_index])

def coarse_avg_accum(components: Sequence[torch.Tensor], fine_index: int, 
                     batch_accum_func: Callable[[torch.Tensor], Number]=torch.sum) -> torch.Tensor:
    # Handle negative indices
    exclude_index = fine_index if fine_index >= 0 else len(components) + fine_index
    return sum([batch_accum_func(c) for i, c in enumerate(components) if i != exclude_index]) / (len(components) - 1)

def indexed_accum(index: int, components: Sequence[torch.Tensor]) -> torch.Tensor:
    return components[index]

### Feature loss functions ###
def nothing(features: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    return features

def bsp(features: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    """Batch spectral penalization loss from `Transferability vs. Discriminability: Batch
        Spectral Penalization for Adversarial Domain Adaptation (ICML 2019)
        <http://ise.thss.tsinghua.edu.cn/~mlong/doc/batch-spectral-penalization-icml19.pdf>.
        
        Computes the loss as the squared largest singular value of the input features.
        
        Each feature in `features` is expected to be a tensor of shape (C, F) with minibatch size `C` 
        feature dimension `F`."""
    # Need to ensure it is float32 if mixed precision is used as svd does not work on that
    singular_values = [torch.svd(f.float())[1] for f in features]
    return [torch.pow(s[0], 2) for s in singular_values]

def afn(features: Sequence[torch.Tensor], radius: int = 10) -> Sequence[torch.Tensor]:
    """The `Stepwise Adaptive Feature Norm loss (ICCV 2019) <https://arxiv.org/pdf/1811.07456v2.pdf>`.
    The Hard AFN loss is used as the Soft AFN Loss effectively does nothing.
    
    Computes the l2 norm of the input features and penalizes the squared difference 
    from a specified radius.
    Radius is given under `radius`. Defaults to 10.
    
    Each feature in `features` is expected to be a tensor of shape (C, F) with minibatch size `C` 
    feature dimension `F`."""
    return [(f.norm(p=2, dim=1).mean() - radius)**2 for f in features]

### Entropy functions ###
def entropy(x: torch.Tensor, eps: float = 1e-5, mean: bool = False) -> torch.Tensor:
    """Computes the (Shannon) entropy of the given element.
    An epison value `eps` is added to the input to prevent log(0) errors. Defaults to 1e-5.
    If `mean` is True, the mean entropy is computed. Defaults to False.
    """
    ent = -(x * torch.log(x+eps)).sum(dim=1)
    if mean:
        return ent.mean()
    else:
        return ent

def entropy_multiple(components: Sequence[torch.Tensor], eps: float = 1e-5) -> Sequence[torch.Tensor]:
    """Computes the (Shannon) entropy of the given components.
    
    An epison value is added to the input to prevent log(0) errors. Defaults to 1e-5.
    
    Each component in `components` is expected to be a tensor of shape (C, F) with minibatch size `C` 
    feature dimension `F`."""
    return [-(c * torch.log(c+eps)).sum(dim=1) for c in components]

def cond_entropy(components: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    """Computes the conditional entropy of the input components.
    The joint distribution is modeled as a simple concatenation of the input components."""
    # NOT CLEAR IF THIS IS CORRECT
    joint_components = torch.concatenate(components)
    return [-joint_components * torch.log(joint_components / c).sum(dim=1) for c in components]

def cross_entropy(p: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the cross entropy between the two input features.
    CE(p,q) and CE(q,p) are computed and returned as a tuple."""
    return [-(p * torch.log(q)).sum(dim=1), -(q * torch.log(p)).sum(dim=1)]

def kl_div(p: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the KL Divergence between the two input features.
    KL(p||q) and KL(q||p) are computed and returned as a tuple.
    Requires that exactly two input features are provided."""
    return [(p * (p / q).log()).sum(dim=1), (q * (q / p).log()).sum(dim=1)]

### Misc function ###
def pad(x: torch.Tensor, pad_size: int) -> torch.Tensor:
    """Pads the given tensor with zeros to the specified size.
    The zeros are appended to the right in the last dimension.
    A clone of the tensor is returned."""
    return F.pad(x.clone(), (0, pad_size-x.shape[-1]))

def mixing_progress(strategy: str, step: int, max_step: int) -> float:
    """Computes the mixing progress of the given strategy at the current step.
    
    Strategies are:
        - curriculum: Proposed curriculum schedule of PAN paper
        - step: linear steps
        - exponential: sigmoidal curve
        - fixed: always returns 1
        - linear: linear interpolation
        - sigmoid: sigmoid function (not tested)
        - cosine: cosine function (not tested)
        
    Returns:
        float: The mixing progress at the current step.
    """
    match strategy:
        case 'curriculum':
            mix_progress = 2 / (1 + math.exp(-10 * step / max_step)) -1
        case 'step':
            mix_progress = step // (max_step // 10) * 0.1
        case 'exponential':
            mix_progress = 2 / (1 + math.exp(-10 * step / max_step)) - 1
        case 'fixed':
            mix_progress = 1
        case "linear":
            mix_progress = step / max_step
        case "cosine":
            mix_progress = 0.5 * (1 + math.cos(step / max_step * math.pi))
        case "sigmoid":
            mix_progress = 1 / (1 + math.exp(-5 * (step / max_step - 0.5)))
        case _:
            raise ValueError(f"Unknown mixing strategy: {strategy}")
    return mix_progress