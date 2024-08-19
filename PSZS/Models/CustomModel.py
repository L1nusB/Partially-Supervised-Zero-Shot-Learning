import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Self, TypeAlias
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from functools import partial

from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.scheduler.scheduler import Scheduler

import PSZS.Classifiers as Classifiers
import PSZS.Classifiers.Heads as Heads
from PSZS.Models.funcs import *
import PSZS.Models.Losses as losses

DAT_TYPE: TypeAlias = torch.Tensor | Tuple[torch.Tensor, ...]
TRAIN_PRED_TYPE: TypeAlias = Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]]
PRED_TYPE: TypeAlias = torch.Tensor | TRAIN_PRED_TYPE
FEATURE_TYPE: TypeAlias = torch.Tensor | Sequence[torch.Tensor]
LABEL_TYPE: TypeAlias = torch.Tensor |Tuple[torch.Tensor,...]

DEFAULT_ACCUM = {
    'afn': sum_accum,
    'bsp': sum_accum,
    'entropy': avg_accum,
    'accum': sum_accum,
}
DEFAULT_ACCUM_SPECIFIED = {
    'afn': 'sum',
    'bsp': 'sum',
    'entropy': 'avg',
    'accum': 'sum',
}
DEFAULT_BATCH_ACCUM = {
    'afn': nothing,
    'bsp': nothing,
    'sum': torch.sum,
    'avg': torch.mean,
    'mean': torch.mean,
}
DEFAULT_FEATURE_WEIGHTS_SUM = {
    'afn': 0.05,
    'bsp': 0.01,
    'accum': 3e-3,
    'entropy': 1e-3,
}
DEFAULT_FEATURE_WEIGHTS_AVG = {
    'afn': 0.1,
    'bsp': 0.02,
    'accum': 6e-3,
    'entropy': 3e-3,
}
DEFAULT_LOGITS_WEIGHTS_SUM = {
    'entropy': 0.05,
}
DEFAULT_LOGITS_WEIGHTS_AVG = {
    'entropy': 0.1,
}
MIXING_STRATEGIES = ['curriculum', 'step', 'linear', 'exponential', 'fixed', 'cosine', 'sigmoid']
MIXING_STRATEGY_MAP = {
        'c': 'curriculum',
        's': 'step',
        'l': 'linear',
        'e': 'exponential',
        'f': 'fixed',
        'cos': 'cosine',
        'sig': 'sigmoid',
}
class CustomModel(nn.Module):
    """A generic model class for custom models."""
    allow_non_strict_checkpoint: bool = False
    def __init__(self,
                 backbone: nn.Module,
                 num_classes: npt.NDArray[np.int_],
                 num_inputs: int,
                 classifier_type: str,
                 opt_kwargs: Dict[str, Any],
                 sched_kwargs: Dict[str, Any],
                 head_type: str = "SimpleHead",
                 cls_loss: str = "ce",
                 bottleneck_dim: Optional[int] = None,
                 hierarchy_accum_func: Callable[[Sequence[torch.Tensor]], torch.Tensor] | str = "sum",
                 feature_loss_func: Optional[Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] | str] = None,
                 feature_accum_func: Optional[Callable[[Sequence[torch.Tensor]], torch.Tensor] | str] = None,
                 logit_loss_func: Optional[Callable[[TRAIN_PRED_TYPE], torch.Tensor] | str] = None,
                 logit_accum_func: Optional[Callable[[Sequence[torch.Tensor]], torch.Tensor] | str] = None,
                 additional_params: Dict[str, Any] = {},
                 **classifier_kwargs) -> None:
        """
        Args:
            backbone (torch.nn.Module): 
                Backbone to extract 2-d features from data
            num_classes (np.ndarray): 
                Number of classes. Gets expanded to (num_inputs, num_head_predictions).
            num_inputs (int): 
                Number of inputs to the classifier during training forward pass.
            classifier_type (Optional[str]): 
                Type of the classifier to use. Must be one of the available classifiers. 
                 If `None`, the model is the backbone itself
            head_type (str): 
                Type of the head to use. Must be one of the available heads. Defaults to `'SimpleHead'`
            cls_loss (str):
                Type of the classification loss to use specified by name. Defaults to `'ce'`
            bottleneck_dim (Optional[int]): 
                Dimension of the bottleneck layer applied after backbone. 
                 If not specified (or <= 0) no bottleneck (identify) is applied. Defaults to None.
            opt_kwargs (dict): 
                Keyword arguments for the optimizer(s)
            sched_kwargs (dict): 
                Keyword arguments for the scheduler(s)
            hierarchy_accum_func (Callable | str): 
                Function to accumulate the loss components for hierarchical heads. If a string is given, 
                 the corresponding function is resolved. Defaults to `None`.
            feature_loss_func (Optional[Callable | str]): 
                Function to compute the feature loss. If a string is given, resolve the function. 
                 If `None` is given no additional feature loss will be used. Defaults to None.
            feature_accum_func (Optional[Callable | str]): 
                Function to accumulate the feature loss components. If a string is given, resolve the function 
                 Only relevant if `feature_loss_func` is not `None`. 
                If not specified tries to resolve based on `DEFAULT_ACCUM` and specified loss. 
                 If loss is given as a callable defaults to `sum`. Defaults to `None`.
            logit_loss_func (Optional[Callable | str]):
                Function to compute the logit loss. If a string is given, resolve the function.
            logit_accum_func (Optional[Callable | str]):
                Function to accumulate the logit loss components. If a string is given, resolve the function 
                 Only relevant if `logit_loss_func` is not `None`. 
                If not specified tries to resolve based on `DEFAULT_ACCUM` and specified loss. 
                 If loss is given as a callable defaults to `sum`. Defaults to `None`.
            additional_params (dict): 
                Additional Parameters any additional loss functions that require additional parameters
                as well as parameters for hierarchy mixing.
            classifier_kwargs (dict): 
                Keyword arguments for the classifier and head
                - num_features (Optional[int]): Number of features before the head layer
                - auto_split (bool): Whether to automatically split the features into len(num_classes). 
                - auto_split_indices (Optional[Sequence[int]]): Indices to split the features when auto_split is True
                - test_head_idx (int): Index of the head to use for testing/validation
                - test_head_pred_idx (Optional[int]): Index of the prediction the head produce to use for testing/validation
                - head_depth (int | Sequence[int]): Depth of the heads. Defaults to 1.
                - reduction_factor (float | Sequence[float]): Reduction factor for the heads in each depth step. Defaults to 2.
                - hierarchy_level (Optional[int]): Level of hierarchy for hierarchical head. (default: None)
        """
        # Fix ciruclar import
        from PSZS.Models.models import construct_classifier
        # Ensure given classifier_type is valid
        if classifier_type not in Classifiers.__all__:
            raise ValueError(f"Invalid classifier type {classifier_type}. Valid types are: {','.join(Classifiers.__all__)}")
        # Ensure given head_type is valid
        if head_type not in Heads.__all__:
            raise ValueError(f"Invalid head type {head_type}. Valid types are: {','.join(Heads.__all__)}")

        # Initialize true custom Model
        super(CustomModel, self).__init__()
        self.backbone = backbone
        self.feature_dim = self.backbone.num_features
        self.num_classes = num_classes
        self.num_inputs = num_inputs
        if bottleneck_dim is not None and bottleneck_dim > 0:
            print(f"Applying bottleneck with dimension {bottleneck_dim}.")
            self.bottleneck = nn.Sequential(
                                    nn.Linear(self.feature_dim, bottleneck_dim),
                                    nn.BatchNorm1d(bottleneck_dim),
                                    nn.ReLU()
                                )
            # Overwrite feature_dim with bottleneck_dim as the new feature dimension
            self.feature_dim = bottleneck_dim
        else:
            self.bottleneck = nn.Identity()
        self.classifier_type : Type[Classifiers.CustomClassifier] = getattr(Classifiers, classifier_type)
        self.head_type : Type[Heads.CustomHead] = getattr(Heads, head_type)
        # Use dynamic construction of classifier from given kwargs
        self.classifier = construct_classifier(classifier_type=self.classifier_type, 
                                               head_type=self.head_type, 
                                               num_classes=self.num_classes,
                                               num_inputs=self.num_inputs,
                                               in_features=self.feature_dim, 
                                               **classifier_kwargs)
        
        self.hierarchy_accum_func = hierarchy_accum_func
        self.feature_loss_func = feature_loss_func
        self.feature_accum_func = feature_accum_func
        self.feature_softmax = False
        self.logit_loss_func = logit_loss_func
        self.logit_accum_func = logit_accum_func
        self.logit_softmax = False
        self.additional_params = additional_params
        self.batch_accum_func = torch.sum
        self.default_feature_loss_weight = 0
        self.default_logit_loss_weight = 0
        
        self._resolve_loss(cls_loss, **classifier_kwargs)
        self.cls_loss_accum_func = sum
        
        # Register DA components and other Parameters to be included in optimizers and schedulers
        self._register_da_components(**classifier_kwargs)
        self._register_hierarchy_accum()
        self._register_feature_loss_func()
        self._register_feature_loss_accum()
        self._register_logit_loss_func()
        self._register_logit_loss_accum()
        
        self.optimizers = self.create_optimizers(**opt_kwargs)
        self.lr_schedulers = self.create_schedulers(**sched_kwargs)
    
    @staticmethod
    def resolve_mixing_strategy(strategy: str) -> str:
        if strategy.lower() in MIXING_STRATEGIES:
            return strategy.lower()
        elif strategy.lower() in MIXING_STRATEGY_MAP:
            strategy = MIXING_STRATEGY_MAP[strategy.lower()]
            return strategy
        else:
            raise ValueError(f"Strategy {strategy} not recognized. "
                             f"Must be one of {MIXING_STRATEGIES} or "
                             f"{list(MIXING_STRATEGY_MAP.keys())}")       
            
    @property
    def num_pred(self) -> int:
        return self.classifier.num_head_pred 
    
    # Weights have nothing with gradients
    @torch.no_grad()
    def mixing_weight(self, increase_step: bool = True) -> Tuple[float, ...]:
        """Computes the mixing weights for each level of the hierarchy.
        The weight is determined by the strategy set in the constructor, initial and final parameters,
        maximal number of steps until mixing is at maximum and the current mixing step.
        If `increase_step` is True, the mixing step is increased by 1.
        
        Strategies are:
        - curriculum: Proposed curriculum schedule of PAN paper
        - step: linear steps
        - linear: linear interpolation
        - exponential: sigmoidal curve
        - fixed: always return the final weight

        Returns:
            Tuple[float,...]: Weight for labels of each hierarchy level.
        """
        mix_progress = torch.tensor([mixing_progress(self.mixing_strategy, self.mixing_step, max_steps) 
                        for max_steps in self.max_mixing_steps], device=self.initial_weights.device)
        if increase_step:
            self.mixing_step += 1
        weights = self.initial_weights + (self.final_weights - self.initial_weights) * mix_progress
        return weights
    
    def _resolve_loss(self, loss: str = 'ce', **kwargs) -> None:
        print(f"Resolving loss {loss}.")
        match loss:
            case 'ce':
                self.cls_loss_func = F.cross_entropy
            case 'binom':
                loss_type = losses.BinomialLoss
                loss_kwargs = loss_type.resolve_params(**kwargs)
                self.cls_loss_func = loss_type(**loss_kwargs)
            case 'contrastive':
                loss_type = losses.ContrastiveLoss
                loss_kwargs = loss_type.resolve_params(**kwargs)
                self.cls_loss_func = loss_type(**loss_kwargs)
            case 'lifted':
                loss_type = losses.LiftedStructureLoss
                loss_kwargs = loss_type.resolve_params(**kwargs)
                self.cls_loss_func = loss_type(**loss_kwargs)
            case 'margin':
                loss_type = losses.MarginLoss
                loss_kwargs = loss_type.resolve_params(nClasses=self.classifier.largest_num_classes_test_lvl, 
                                                       **kwargs)
                self.cls_loss_func = loss_type(**loss_kwargs)
            case 'ms':
                loss_type = losses.MultiSimilarityLoss
                loss_kwargs = loss_type.resolve_params(**kwargs)
                self.cls_loss_func = loss_type(**loss_kwargs)
            case 'triplet':
                loss_type = losses.HardMineTripletLoss
                loss_kwargs = loss_type.resolve_params(**kwargs)
                self.cls_loss_func = loss_type(**loss_kwargs)
            case _:
                warnings.warn(f"Loss {loss} not recognized. Using default Cross-Entropy loss. "
                              f"Available losses: {losses.get_losses_names()}")
                self.cls_loss_func = F.cross_entropy
        print(f"Using loss function {self.cls_loss_func.__class__.__name__}.")
        
    def _register_hierarchy_accum(self) -> None:
        """Sets values of `hierarchy_accum_func` and `hierarchy_weights` based on the current `hierarchy_accum_func`.
        If a string is given, the corresponding function is resolved otherwise the passed Callable is used.
        Valid values for `hierarchy_accum_func` are: sum, avg, mean, make, model, weighted.
        """
        # If a string is given, resolve the function otherwise use the given function
        # already in self.hierarchy_accum_func
        if isinstance(self.hierarchy_accum_func, str):
            match self.hierarchy_accum_func:
                case "sum":
                    self.hierarchy_accum_func = sum_accum
                case "avg" | "mean":
                    self.hierarchy_accum_func = avg_accum
                case "make":
                    self.hierarchy_accum_func = partial(indexed_accum, -2)
                case "model":
                    self.hierarchy_accum_func = partial(indexed_accum, -1)
                case "coarse" | "coarse_sum":
                    self.hierarchy_accum_func = partial(coarse_sum_accum, 
                                                        fine_index=self.classifier.test_head_pred_idx)
                case "coarse_avg" | "coarse_mean":
                    self.hierarchy_accum_func = partial(coarse_avg_accum, 
                                                        fine_index=self.classifier.test_head_pred_idx)
                case "weighted":
                    weights = self.additional_params.get('hierarchy_weights', None)
                    # Check if weights are given, if not use default weights
                    if weights is None:
                        weights = [0.8]
                        # Use Normalization by default if no weights given (but allow to be overwritten)
                        normalize = self.additional_params.get('normalize_weights', True)
                    else:
                        # Do not use Normalization by default if weights given (but allow to be overwritten)
                        normalize = self.additional_params.get('normalize_weights', False)
                    # Parameter so it is sent to device as well (but no gradients)
                    self.hierarchy_weights = nn.Parameter(construct_weights(weights=weights, 
                                                                            length=self.num_pred,
                                                                            normalized=normalize),
                                                          requires_grad=False)
                    assert len(self.hierarchy_weights) == self.num_pred, f"Number of weights ({len(self.hierarchy_weights)}) must match number of components ({self.num_pred})."
                    self.hierarchy_accum_func = partial(weighted_accum, weights=self.hierarchy_weights)
                case _:
                    self.mixing_strategy = self.resolve_mixing_strategy(self.hierarchy_accum_func)
                    if self.mixing_strategy:
                        initial_weight = self.additional_params.get('initial_weights', None)
                        final_weight = self.additional_params.get('final_weights', None)
                        # Check if initial and final are given, if not use default weights for finest level and construct other levels
                        if initial_weight is None:
                            normalize = self.additional_params.get('normalize_weights', True)
                            initial_weight = construct_weights(weights=[0.1], 
                                                               length=self.num_pred,
                                                               normalized=normalize)
                        elif len(initial_weight) != self.num_pred:
                            # Partial initial weights given, construct the rest
                            normalize = self.additional_params.get('normalize_weights', True)
                            initial_weight = construct_weights(weights=initial_weight,
                                                               length=self.num_pred,
                                                               normalized=normalize)
                        if final_weight is None:
                            normalize = self.additional_params.get('normalize_weights', True)
                            final_weight = construct_weights(weights=[0.9], 
                                                               length=self.num_pred,
                                                               normalized=normalize)
                        elif len(final_weight) != self.num_pred:
                            # Partial final weights given, construct the rest
                            normalize = self.additional_params.get('normalize_weights', True)
                            final_weight = construct_weights(weights=final_weight,
                                                               length=self.num_pred,
                                                               normalized=normalize)
                        self.initial_weights = nn.Parameter(torch.tensor(initial_weight), requires_grad=False)
                        self.final_weights = nn.Parameter(torch.tensor(final_weight), requires_grad=False)
                        self.max_mixing_steps = self.additional_params.get('max_mixing_steps', None)
                        if self.max_mixing_steps is not None:
                            self.max_mixing_steps = nn.Parameter(torch.tensor([self.max_mixing_steps] * self.num_pred, dtype=int), 
                                                                 requires_grad=False)
                        self.mixing_step = 0
                        self.hierarchy_accum_func = partial(dynamic_weighted_accum, weight_func=self.mixing_weight)
                    else:
                        raise ValueError(f"Invalid hierarchy-accum {self.hierarchy_accum_func}. Valid values are: sum, avg, make, model, weighted, dynamic")
    
    def _auto_resolve_accum(self, 
                            loss_specifier: str, 
                            accum_specifier: str, 
                            batch_accum: bool = False) -> None:
        """Automatically resolve the accumulation function based on the given loss specifier.
        If the accumulation is already set nothing is modified.
        If a batch accumulation function is required, `batch_accum_func` is set based on the loss specifier
        and the accum function will be a partial that uses `batch_accum_func`.
        Resolving is performed based on the DEFAULT_ACCUM and DEFAULT_BATCH_ACCUM dictionaries.
        If the loss_specified is not found in `DEFAULT_ACCUM` use `sum_accum` as default.
        If batch_accum is True and the loss_specifier is not found in `DEFAULT_BATCH_ACCUM`,
        `batch_accum_func` is not modified."""
        # Only set the accum function if it is not already set
        if getattr(self, accum_specifier) is None:
            accum_func = DEFAULT_ACCUM.get(loss_specifier, sum_accum)
            if batch_accum:
                if loss_specifier in DEFAULT_BATCH_ACCUM:
                    self.batch_accum_func = DEFAULT_BATCH_ACCUM[loss_specifier]
                setattr(self, accum_specifier, partial(accum_func, batch_accum_func=self.batch_accum_func))
            else:
                setattr(self, accum_specifier, accum_func)
                
    def _set_default_weight_features(self, loss_specifier: str) -> None:
        if isinstance(self.feature_accum_func, str):
            defaults = DEFAULT_FEATURE_WEIGHTS_AVG if (self.feature_accum_func == 'avg' or self.feature_accum_func == 'mean') else DEFAULT_FEATURE_WEIGHTS_SUM
            self.default_feature_loss_weight = defaults.get(loss_specifier, 1e-3)
        elif self.feature_accum_func is None:
            accum_type = DEFAULT_ACCUM_SPECIFIED.get(loss_specifier, "sum")
            defaults = DEFAULT_FEATURE_WEIGHTS_AVG if (accum_type == 'avg') else DEFAULT_FEATURE_WEIGHTS_SUM
            self.default_feature_loss_weight = defaults.get(loss_specifier, 1e-3)
        else:
            self.default_feature_loss_weight = 1e-3
    
    def _set_default_weight_logits(self, loss_specifier: str) -> None:
        if isinstance(self.logit_accum_func, str):
            defaults = DEFAULT_LOGITS_WEIGHTS_AVG if (self.logit_accum_func == 'avg' or self.logit_accum_func == 'mean') else DEFAULT_LOGITS_WEIGHTS_SUM
            self.default_logit_loss_weight = defaults.get(loss_specifier, 1e-3)
        elif self.logit_accum_func is None:
            accum_type = DEFAULT_ACCUM_SPECIFIED.get(loss_specifier, "sum")
            defaults = DEFAULT_LOGITS_WEIGHTS_AVG if (accum_type == 'avg') else DEFAULT_LOGITS_WEIGHTS_SUM
            self.default_logit_loss_weight = defaults.get(loss_specifier, 1e-3)
        else:
            self.default_logit_loss_weight = 0.05
            
    def _register_feature_loss_func(self) -> None:
        """Sets value of `feature_loss_func` based on the current `feature_loss_func`.
        If a string is given, the corresponding function is resolved otherwise the passed Callable is used.
        Valid string values for `feature_loss_func` are: `none`, `accum`, `entropy`, `bsp`, `afn`.
        When resolving the function, the `logit_accum_func` is set based on `_auto_resolve_accum` with `batch_accum=True`.
        If `None` is given, the `feature_accum_func` is set to `None` as well.
        If `entropy` is specified `eps` is retrieved from `additional_params` or set to default value.
        If `afn` is specified `radius` is retrieved from `additional_params` or set to default value.
        """
        if self.feature_loss_func is None:
            # Also set feature_accum_func to None
            self.feature_accum_func = None
        # If a string is given, resolve the function otherwise use the given function
        elif isinstance(self.feature_loss_func, str):
            match self.feature_loss_func.lower():
                case 'none':
                    self.feature_loss_func = None
                    # Also set feature_accum_func to None
                    self.feature_accum_func = None
                case 'accum':
                    self.feature_loss_func = nothing
                    self._set_default_weight_features(loss_specifier='accum')
                    self._auto_resolve_accum(loss_specifier='accum', 
                                             accum_specifier='feature_accum_func',
                                             batch_accum=True)
                case 'entropy':
                    if self.additional_params.get('eps', None) is None:
                        # Use default eps value
                        self.feature_loss_func = entropy_multiple
                    else:
                        self.feature_loss_func = partial(entropy_multiple, eps=self.additional_params['eps'])
                    self._set_default_weight_features(loss_specifier='entropy')
                    self._auto_resolve_accum(loss_specifier='entropy', 
                                             accum_specifier='feature_accum_func',
                                             batch_accum=True)
                    self.feature_softmax = True
                case 'bsp':
                    self.feature_loss_func = bsp
                    self._set_default_weight_features(loss_specifier='bsp')
                    # For bsp no batch accumulation is required as reduction already only return single values
                    self._auto_resolve_accum(loss_specifier='bsp', 
                                             accum_specifier='feature_accum_func',
                                             batch_accum=False)
                case 'afn':
                    if self.additional_params.get('radius', None) is None:
                        # Use default radius value
                        self.feature_loss_func = afn
                    else:
                        self.feature_loss_func = partial(afn, radius=self.additional_params['radius'])
                    self._set_default_weight_features(loss_specifier='afn')
                    # For afn no batch accumulation is required as reduction already only return single values
                    self._auto_resolve_accum(loss_specifier='afn', 
                                             accum_specifier='feature_accum_func',
                                             batch_accum=False)
                case _:
                    raise ValueError(f"Feature function {self.feature_loss_func} not recognized. "
                                    "Available values: 'none', 'accum', 'entropy', 'bsp', 'afn'")
    
    def _register_feature_loss_accum(self) -> None:
        """Set value of `feature_accum_func` based on the current `feature_accum_func`.
        If a string is given, the corresponding function is resolved otherwise the passed Callable is used.
        Valid values for `feature_accum_func` are: sum, avg, mean.
        """
        if self.feature_accum_func is None:
            pass # Do Nothing
        elif isinstance(self.feature_accum_func, str):    
            match self.feature_accum_func.lower():
                case 'sum':
                    self.feature_accum_func = partial(sum_accum, batch_accum_func=self.batch_accum_func)
                case 'mean' | 'avg':
                    self.feature_accum_func = partial(avg_accum, batch_accum_func=self.batch_accum_func)
                case _:
                    raise ValueError(f"Feature accumulation method {self.feature_accum_func} not recognized. "
                                    "Available values: 'sum', 'mean', 'avg'")
            
    def _register_logit_loss_func(self) -> None:
        """Sets value of `logit_loss_func` based on the current `logit_loss_func`.
        If a string is given, the corresponding function is resolved otherwise the passed Callable is used.
        Valid string values for `logit_loss_func` are: `none`, `accum`, `entropy`, `condentropy`, 
        `cond_entropy`, `crossentropy`, `cross_entropy`, `kl`.
        When resolving the function, the `logit_accum_func` is set based on `_auto_resolve_accum` with `batch_accum=True`.
        If `None` is given, the `logit_accum_func` is set to `None` as well.
        If `entropy` is specified `eps` is retrieved from `additional_params` or set to default value.
        """
        if self.logit_loss_func is None:
            # Also set logit_accum_func to None
            self.logit_accum_func = None
        # If a string is given, resolve the function otherwise use the given function
        elif isinstance(self.logit_loss_func, str):
            match self.logit_loss_func.lower():
                case 'none':
                    self.logit_loss_func = None
                    # Also set logit_accum_func to None
                    self.logit_accum_func = None
                case 'accum':
                    self.logit_loss_func = nothing
                    self._set_default_weight_logits(loss_specifier='accum')
                    self._auto_resolve_accum(loss_specifier='accum', 
                                             accum_specifier='logit_accum_func',
                                             batch_accum=True)
                case 'entropy':
                    if self.additional_params.get('eps', None) is None:
                        # Use default eps value
                        self.logit_loss_func = entropy_multiple
                    else:
                        self.logit_loss_func = partial(entropy_multiple, eps=self.additional_params['eps'])
                    self._set_default_weight_logits(loss_specifier='entropy')
                    self._auto_resolve_accum(loss_specifier='entropy', 
                                             accum_specifier='logit_accum_func',
                                             batch_accum=True)
                    self.logit_softmax = True
                case 'condentropy' | 'cond_entropy':
                    # Not sure if the implementation is correct
                    self.logit_loss_func = cond_entropy
                    self._set_default_weight_logits(loss_specifier='condentropy')
                    self._auto_resolve_accum(loss_specifier='condentropy', 
                                             accum_specifier='logit_accum_func',
                                             batch_accum=True)
                case 'crossentropy' | 'cross_entropy':
                    self.logit_loss_func = cross_entropy
                    self._set_default_weight_logits(loss_specifier='crossentropy')
                    self._auto_resolve_accum(loss_specifier='crossentropy', 
                                             accum_specifier='logit_accum_func',
                                             batch_accum=True)
                case 'kl':
                    self.logit_loss_func = kl_div
                    self._set_default_weight_logits(loss_specifier='kl')
                    self._auto_resolve_accum(loss_specifier='kl', 
                                             accum_specifier='logit_accum_func',
                                             batch_accum=True)
                case _:
                    raise ValueError(f"Logit loss function {self.logit_loss_func} not recognized. "
                                    "Available values: 'none', 'accum', 'entropy', 'condentropy', "
                                    "'cond_entropy, 'crossentropy', 'cross_entropy, 'kl'")
                    
    def _register_logit_loss_accum(self) -> None:
        """Set value of `feature_accum_func` based on the current `feature_accum_func`.
        If a string is given, the corresponding function is resolved otherwise the passed Callable is used.
        Valid values for `feature_accum_func` are: sum, avg, mean.
        """
        if self.logit_accum_func is None:
            pass # Do Nothing
        elif isinstance(self.logit_accum_func, str):    
            match self.logit_accum_func.lower():
                case 'sum':
                    self.logit_accum_func = partial(sum_accum, batch_accum_func=self.batch_accum_func)
                case 'mean' | 'avg':
                    self.logit_accum_func = partial(avg_accum, batch_accum_func=self.batch_accum_func)
                case _:
                    raise ValueError(f"Feature accumulation method {self.logit_accum_func} not recognized. "
                                    "Available values: 'sum', 'mean', 'avg'")
        else:
            self.logit_accum_func = self.logit_accum_func
        
    def _register_da_components(self, **kwargs) -> None:
        """Register domain adaptation components (if any are required).
        Needs to be called before constructing the optimizer and scheduler."""
        pass
    
    def create_optimizers(self, **params) -> List[torch.optim.Optimizer]:
        """Create optimizers for the model.
        As a list of optimizers to uniformly support multiple optimizers.
        Overwritten in derived classes to provide specific optimizers.
        """
        return [create_optimizer_v2(model_or_params=self.model_or_params(), **params)]
    
    def create_schedulers(self, **params) -> List[Scheduler]:
        """Create schedulers for the model.
        One scheduler is generated for each optimizer.
        Overwritten in derived classes to provide specific schedulers.
        """
        # Only first component is returned as the scheduler as the second is num_epochs
        # which is not interesting for iter based optimizers
        return [create_scheduler_v2(optimizer=opt, **params)[0] for opt in self.optimizers]
        
    def val_test_state_dict(self) -> Dict[str, Any]:
        """"""
        val_test_state_dict = {f'backbone.{k}':v for k,v in self.backbone.state_dict().items()}
        val_test_state_dict.update({f'bottleneck.{k}':v for k,v in self.bottleneck.state_dict().items()})
        val_test_state_dict.update({f'classifier.{k}':v for k,v in self.classifier.val_test_state_dict().items()})
        return val_test_state_dict
    
    def model_or_params(self) -> Self | Dict[str, Any]:
        """Return model or model parameters.
        This allows to specify only a subset of parameters to be optimized 
        or specify individual learning rates for certain components."""
        return self
        
    def forward(self, x: torch.Tensor) -> PRED_TYPE | Tuple[PRED_TYPE, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f = self.bottleneck(f)
        if self.training:
            predictions = self.classifier(f)
            return predictions, f
        else:
            predictions = self.classifier.forward_test(f)
            return predictions
        
    def compute_feature_loss(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.feature_softmax:
            features = [f.softmax(dim=1) for f in features]
        return self.feature_accum_func(self.feature_loss_func(features))
    
    def compute_logit_loss(self, logits: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.logit_softmax:
            logits = [l.softmax(dim=1) for l in logits]
        return self.logit_accum_func(self.logit_loss_func(logits))
    
    def compute_cls_loss(self, 
                         pred: PRED_TYPE, 
                         target: LABEL_TYPE,
                         mixup: bool = False,
                         ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        """Computes the classification loss for the given predictions and targets using `cls_loss_func` and 
        `cls_loss_accum_func`. The loss dynamically determines whether we have single or hierarchical predictions 
        and use a single or multiple input/targets.
        For hierarchical predictions, the loss is computed for each level and accumulated using `hierarchy_accum_func`.
        Several internal assertions are made to ensure the correct input and target shapes.
        In case mixup is used the target is expected to have a different shape thus it needs to be specified via `mixup`.
        Note that mixup is not supported for hierarchical heads.

        Args:
            pred (PRED_TYPE): The predictions of the model.
            target (LABEL_TYPE): The target labels for the predictions.
            mixup (bool, optional): Whether mixup is used, affecting the target tensors. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Sequence[torch.Tensor]]: Accumulated loss and loss components for each input.
        """
        if self.head_type.returns_multiple_outputs:
            # Pad/Support for Base_Single optimizers
            if self.num_inputs < 2:
                assert isinstance(target, torch.Tensor), f'Single input requires single target tensor. Got {type(target)}'
                assert isinstance(pred[0], torch.Tensor), f'Single input requires single prediction tensor. Got {type(pred[0])}'
                # Wrap target into tuple to work for Single and Multiple predictions/inputs (Optimizer_Single, Optimizer_Multiple)
                target = (target,)
                # Wrap pred into tuple to work for Single and Multiple predictions/inputs (Optimizer_Single, Optimizer_Multiple)
                pred = (pred,)
            # Hierarchical head with multiple results per sample (multiple heads) 
            # --> Requires gt_label also contain multiple labels per sample
            # Match split feature predictions with ground truth labels
            assert len(pred[0])<=target[0].shape[-1], (f"Number of predictions ({len(pred[0])}) must "
                                                          f"be less than or equal to number of labels ({target[0].shape[-1]}).")
            # Mixup is not supported for hierarchical heads
            loss_components = [self.hierarchy_accum_func(
                                [self.cls_loss_func(p[level], gt[:,level]) for level in range(len(p))]
                                ) for p, gt in zip(pred, target)]
            loss = self.cls_loss_accum_func(loss_components)
            return loss, loss_components 
        else:
            # Simple head with single result per sample --> Requires gt_label to be single label per sample
            if isinstance(target, torch.Tensor):
                assert target.ndim==1, "Head returns single predictions. Ground truth labels must be single label per sample."
            else:
                # In this case target is a tuple of tensors
                if mixup:
                    assert target[0].ndim==2, f"Head returns single predictions with mixup enabled. But got ndim {target[0].ndim}."
                else:
                    assert target[0].ndim==1, f"Head returns single predictions. Ground truth labels must be single label per sample but got {target[0].ndim}."
            if isinstance(pred, torch.Tensor):
                # This is the most "simple" case where a single prediction is given without any hierarchy
                assert isinstance(target, torch.Tensor), "target must be single tensor."
                loss = loss_components = self.cls_loss_func(pred, target)
                return loss, [loss_components]
            else:
                # Custom Classifier that returns multiple predictions i.e. has multiple heads
                loss_components = [self.cls_loss_func(p, gt) for p, gt in zip(pred, target)]
                loss = self.cls_loss_accum_func(loss_components)
                return loss, loss_components
    
    def get_model_kwargs(**kwargs) -> Tuple[dict, dict]:
        """Dynamically resolve relevant kwargs for model construction.
        The resolved parameters are removed from the original kwargs.
        Overwritten in derived classes to provide specific kwargs.
        
        Returns:
            Tuple[dict, dict]: Tuple of model kwargs filtered kwargs
        """
        params = {}
        params['additional_params'] = {}
        if 'hierarchy_weights' in kwargs:
            params['additional_params']['hierarchy_weights'] = kwargs.get('hierarchy_weights')
            kwargs.pop('hierarchy_weights', None)
        if 'normalize_weights' in kwargs:
            params['additional_params']['normalize_weights'] = kwargs.get('normalize_weights')
            kwargs.pop('normalize_weights', None)
        if 'initial_weights' in kwargs:
            params['additional_params']['initial_weights'] = kwargs.get('initial_weights')
            kwargs.pop('initial_weights', None)
        if 'final_weights' in kwargs:
            params['additional_params']['final_weights'] = kwargs.get('final_weights')
            kwargs.pop('final_weights', None)
        params['additional_params']['max_mixing_steps'] = kwargs.get('max_mixing_steps', None)
        kwargs.pop('max_mixing_steps', None)
        
        if 'hierarchy_accum_func' in kwargs:
            params['hierarchy_accum_func'] = kwargs.get('hierarchy_accum_func')
            kwargs.pop('hierarchy_accum_func')
        if 'feature_loss_func' in kwargs:
            params['feature_loss_func'] = kwargs.get('feature_loss_func')
            kwargs.pop('feature_loss_func')
        if 'feature_accum_func' in kwargs:
            params['feature_accum_func'] = kwargs.get('feature_accum_func')
            kwargs.pop('feature_accum_func')
        if 'logit_loss_func' in kwargs:
            params['logit_loss_func'] = kwargs.get('logit_loss_func')
            kwargs.pop('logit_loss_func')
        if 'logit_accum_func' in kwargs:
            params['logit_accum_func'] = kwargs.get('logit_accum_func')
            kwargs.pop('logit_accum_func')
        if 'eps' in kwargs:
            params['additional_params']['eps'] = kwargs['eps']
            kwargs.pop('eps')
        if 'radius' in kwargs:
            params['additional_params']['radius'] = kwargs['radius']
            kwargs.pop('radius')
        return params, kwargs