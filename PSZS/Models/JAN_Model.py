from typing import Any, Dict
import numpy as np
import numpy.typing as npt

import torch.nn as nn
from torch.nn.modules import Module

from PSZS.Models.CustomModel import CustomModel
from PSZS.Alignment import JointMultipleKernelMaximumMeanDiscrepancy, Theta, GaussianKernel

class JAN_Model(CustomModel):
    """Custom Model implementing JAN.
    Using the JAN loss to align the features of the source and target domain.
    Based on Joint Multiple Kernel Maximum Mean Discrepancy from the tllib library.
    Based on the paper "Deep Transfer Learning with Joint Adaptation Networks" by Long et al.
    
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
                 adversarial: bool = False,
                 alpha_min: int = -3,
                 alpha_max: int = 2,
                 sigma: float = 0.92,
                 linear: bool = False,
                 **additional_kwargs) -> None:
        """
        Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (np.ndarray): Number of classes. Gets expanded to (num_inputs, num_head_predictions).
        num_inputs (int): Number of inputs to the classifier during training forward pass.
        classifier_type (Optional[str]): Type of the classifier to use. Must be one of the available classifiers. If None, the model is the backbone itself
        opt_kwargs (dict): Keyword arguments for the optimizer(s)
        sched_kwargs (dict): Keyword arguments for the scheduler(s)
        adversarial (bool): Whether to use adversarial theta in the loss
        alpha_min (int): Minimum exponent for the Gaussian kernel
        alpha_max (int): Maximum exponent for the Gaussian kernel
        sigma (float): Sigma for the Gaussian kernel
        linear (bool): Whether to use the linear version of JAN
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
        self.adversarial = adversarial
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.sigma = sigma
        self.linear = linear
        super().__init__(backbone=backbone, 
                         num_classes=num_classes,
                         num_inputs=num_inputs, 
                         classifier_type=classifier_type,
                         opt_kwargs=opt_kwargs,
                         sched_kwargs=sched_kwargs,
                         **additional_kwargs)
    
    def _register_da_components(self, **kwargs) -> None:
        """Construct the jmmd loss and thetas if `adversarial=True`."""
        self.thetas = None
        if self.adversarial:
            print("Using adversarial theta in the loss")
            # Get the effective head prediction size i.e. the number of classes for the main hierarchy level
            # Use max which should correspond to the source anyways (otherwise it was explicitly allowed or warned against)
            # via allow_mix_num_classes or allow_non_strict_num_classes_order in get_largest_head_num_classes() in custom_classifier.py
            theta_num_classes = self.classifier.smallest_num_classes_test_lvl
            self.thetas = nn.ModuleList([Theta(dim) for dim in (self.feature_dim, theta_num_classes)])
        self.jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
                                    kernels=(
                                        [GaussianKernel(alpha=2 ** k) for k in range(self.alpha_min, self.alpha_max)],
                                        (GaussianKernel(sigma=self.sigma, track_running_stats=False),)
                                    ),
                                    linear=self.linear, 
                                    thetas=self.thetas
                                )
        
    def val_test_state_dict(self) -> Dict[str, Any]:
        val_test_state_dict = super().val_test_state_dict()
        val_test_state_dict.update({f'jmmd_loss.{k}':v for k,v in self.jmmd_loss.state_dict().items()})
        return val_test_state_dict
    
    # # Check whether to use higher learning rate for thetas
    # def model_or_params(self) -> Dict[str, Any]:
    #     params = [{"params": self.backbone.parameters(), 'weight_decay':0},
    #               {"params": self.bottleneck.parameters()},
    #               {"params": self.classifier.parameters()}]
    #     if self.thetas is not None:
    #         params += [{'params': theta.parameters(), 'lr': 0.1} for theta in self.thetas]
    #     return params
        
    def get_model_kwargs(**kwargs) -> dict:
        """Dynamically resolve relevant kwargs for model construction.
        """
        # Need for unbound super call
        # https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        params, kwargs = super(JAN_Model, JAN_Model).get_model_kwargs(**kwargs)
        if 'adversarial' in kwargs:
            params['adversarial'] = kwargs['adversarial']
            kwargs.pop('adversarial')
        if 'alpha_min' in kwargs:
            params['alpha_min'] = kwargs.get('alpha_min')
            kwargs.pop('alpha_min')
        if 'alpha_max' in kwargs:
            params['alpha_max'] = kwargs.get('alpha_max')
            kwargs.pop('alpha_max')
        if 'sigma' in kwargs:
            params['sigma'] = kwargs['sigma']
            kwargs.pop('sigma')
        if 'linear' in kwargs:
            params['linear'] = kwargs['linear']
            kwargs.pop('linear')
        return params, kwargs