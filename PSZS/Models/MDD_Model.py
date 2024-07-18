from typing import Any, Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
from torch.nn.modules import Module
from tllib.modules.grl import WarmStartGradientReverseLayer

from PSZS.Models.CustomModel import CustomModel, PRED_TYPE

EXTENDED_PRED_TYPE = Tuple[PRED_TYPE, torch.Tensor]

class MDD_Model(CustomModel):
    """Custom Model implementing MDD (Margin Disparity Discrepancy)
    Using a custom adversarial classifier to predict the domain of the input data.
    Adapted from tllib.alignment.mdd from the tllib library.
    Based on the paper "Learning to Discover Cross-Domain Relations with Generative Adversarial Networks" by Bousmalis et al.
    
    Inputs:
        - x (tensor): input data fed to `backbone`
    """
    def __init__(self, 
                 backbone: Module, 
                 num_classes: npt.NDArray[np.int_], 
                 num_inputs: int, 
                 classifier_type: str, 
                 opt_kwargs: Dict[str, Any],
                 sched_kwargs: Dict[str, Any],
                 hidden_size: int = 1024,
                 grl_alpha: float = 1.,
                 grl_lo: float = 0.,
                 grl_hi: float = 1.,
                 grl_max_iters: int = 1000,
                 **additional_kwargs) -> None:
        """
        Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (np.ndarray): Number of classes. Gets expanded to (num_inputs, num_head_predictions).
        num_inputs (int): Number of inputs to the classifier during training forward pass.
        classifier_type (Optional[str]): Type of the classifier to use. Must be one of the available classifiers. If None, the model is the backbone itself
        opt_kwargs (dict): Keyword arguments for the optimizer(s)
        sched_kwargs (dict): Keyword arguments for the scheduler(s)
        hidden_size (int): Size of the hidden layer in the domain discriminator. Defaults to 1024.
        grl_alpha (float): Alpha value for the gradient reversal layer. Defaults to 1.
        grl_lo (float): Lower bound for the gradient reversal layer. Defaults to 0.
        grl_hi (float): Upper bound for the gradient reversal layer. Defaults to 1.
        grl_max_iters (int): Maximum number of iterations for the gradient reversal layer. Defaults to 1000.
        additional_kwargs (dict): 
            Keyword arguments of the base class `CustomModel` with default values
            - head_type (str): Type of the head to use. Must be one of the available heads. Defaults to 'SimpleHead'
            - bottleneck_dim (Optional[int]): Dimension of the bottleneck layer applied after backbone. 
            - hierarchy_accum_func (Callable | str): Function/Name to accumulate the hierarchy loss. Defaults to `sum`.
            - hierarchy_accum_weights (Sequence[float]): Weights for the hierarchy loss if `weighted`. Defaults to [0.2, 0.8]
            - feature_loss_func (Optional[Callable] | str): Function/Name to calculate the feature loss (if specified). Defaults to `None`.
            - feature_accum_func (Optional[Callable | str]): Function/Name to accumulate the feature loss. Defaults to `None`.
            - logit_loss_func (Optional[Callable] | str): Function/Name to calculate the logit loss (if specified). Defaults to `None`.
            - logit_accum_func (Optional[Callable | str]): Function/Name to accumulate the logit loss. Defaults to `None`.
            - additional_params (dict): Additional parameters for the model
            
            as well as arguments for the classfier and head (aka `classifier_kwargs`):
            - num_features (Optional[int]): Number of features before the head layer
            - auto_split (bool): Whether to automatically split the features into len(num_classes). 
            - test_head_idx (int): Index of the head to use for testing/validation
            - test_head_pred_idx (Optional[int]): Index of the prediction the head produce to use for testing/validation
            - depth (int | Sequence[int]): Depth of the heads. Defaults to 1.
            - reduction_factor (float | Sequence[float]): Reduction factor for the heads in each depth step. Defaults to 2.
            - hierarchy_level (Optional[int]): Level of hierarchy for hierarchical head. (default: None)
        
        """
        self.hidden_size = hidden_size
        self.grl_alpha = grl_alpha
        self.grl_lo = grl_lo
        self.grl_hi = grl_hi
        self.grl_max_iters = grl_max_iters
        super().__init__(backbone=backbone, 
                         num_classes=num_classes,
                         num_inputs=num_inputs, 
                         classifier_type=classifier_type,
                         opt_kwargs=opt_kwargs,
                         sched_kwargs=sched_kwargs,
                         **additional_kwargs)
    
    def _register_da_components(self, **kwargs) -> None:
        """Construct the gradient reversal layer and the adversarial classifier.
        The weights of the adversarial classifier are initialized randomly.
        Hidden size is updated to be smaller than the feature dimension if necessary."""
        # No need to send to device as this is done in models.py when constructing the model
        self.grl = WarmStartGradientReverseLayer(alpha=self.grl_alpha, 
                                                 lo=self.grl_lo, 
                                                 hi=self.grl_hi, 
                                                 max_iters=self.grl_max_iters, 
                                                 auto_step=True)
        # Get the effective head prediction size i.e. the number of classes for the main hierarchy level
        # Use max which should correspond to the source anyways (otherwise it was explicitly allowed or warned against)
        # via allow_mix_num_classes or allow_non_strict_num_classes_order in get_largest_head_num_classes() in custom_classifier.py
        adv_num_classes = self.classifier.largest_num_classes_test_lvl()
        # Try to use given hidden layer as long as sufficient features are available
        # or half of the feature dimension
        self.hidden_size = self.hidden_size if self.feature_dim > self.hidden_size else int(self.feature_dim / 2)
        self.adv_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, adv_num_classes)
        )
        # Fill with random values (initialize)
        # Not sure if this is necessary as pytorch does this by default (but maybe with different scheme)
        for dep in range(2):
            self.adv_classifier[dep * 3].weight.data.normal_(0, 0.01)
            self.adv_classifier[dep * 3].bias.data.fill_(0.0)
                    
    def forward(self, x: torch.Tensor) -> EXTENDED_PRED_TYPE | Tuple[PRED_TYPE, torch.Tensor]:
        """During training additionally return predictions of an adversarial classifier."""
        f = self.backbone(x)
        f = self.bottleneck(f)
        if self.training:
            predictions = self.classifier(f)
            f_adv = self.grl(f)
            # For multiple model the adv prediction still needs to be split
            # according to the respective batch sizes
            predictions_adv = self.adv_classifier(f_adv)
            return (predictions, predictions_adv), f
        else:
            predictions = self.classifier.forward_test(f)
            return predictions
        
    def val_test_state_dict(self) -> Dict[str, Any]:
        """"""
        val_test_state_dict = super().val_test_state_dict()
        val_test_state_dict.update({f'adv_classifier.{k}':v for k,v in self.adv_classifier.state_dict().items()})
        return val_test_state_dict
        
    def get_model_kwargs(**kwargs) -> dict:
        """Dynamically resolve relevant kwargs for model construction.
        """
        params, kwargs = super(MDD_Model, MDD_Model).get_model_kwargs(**kwargs)
        if 'hidden_size' in kwargs:
            params['hidden_size'] = kwargs.get('hidden_size')
            kwargs.pop('hidden_size')
        if 'grl_alpha' in kwargs:
            params['grl_alpha'] = kwargs.get('grl_alpha')
            kwargs.pop('grl_alpha')
        if 'grl_lo' in kwargs:
            params['grl_lo'] = kwargs.get('grl_lo')
            kwargs.pop('grl_lo')
        if 'randomized' in kwargs:
            params['randomized'] = kwargs.get('randomized')
            kwargs.pop('randomized')
        if 'grl_hi' in kwargs:
            params['grl_hi'] = kwargs.get('grl_hi')
            kwargs.pop('grl_hi')
        if 'grl_max_iters' in kwargs:
            params['grl_max_iters'] = kwargs.get('grl_max_iters')
            kwargs.pop('grl_max_iters')
        return params, kwargs