from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt

import torch
from torch.nn.modules import Module

from timm.optim import create_optimizer_v2

from PSZS.Models.CustomModel import CustomModel, PRED_TYPE

class MCD_Model(CustomModel):
    """Custom Model implementing MCD (Maximum Classifier Discrepancy)
    Adapted from tllib.alignment.mcd from the tllib library.
    Based on the paper "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation" by Saito et al.
    
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
                 **additional_kwargs) -> None:
        """
        Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (np.ndarray): Number of classes. Gets expanded to (num_inputs, num_head_predictions).
        num_inputs (int): Number of inputs to the classifier during training forward pass.
        classifier_type (Optional[str]): Type of the classifier to use. Must be one of the available classifiers. If None, the model is the backbone itself
        head_type (str): Type of the head to use. Must be one of the available heads. Defaults to 'SimpleHead'
        bottleneck_dim (Optional[int]): Dimension of the bottleneck layer applied after backbone. \
            If not specified (or <= 0) no bottleneck (identify) is applied. Defaults to None.
        opt_kwargs (dict): Keyword arguments for the optimizer(s)
        sched_kwargs (dict): Keyword arguments for the scheduler(s)
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
        super().__init__(backbone=backbone, 
                         num_classes=num_classes,
                         num_inputs=num_inputs, 
                         classifier_type=classifier_type,
                         opt_kwargs=opt_kwargs,
                         sched_kwargs=sched_kwargs,
                         **additional_kwargs)
    
    def _register_da_components(self, **classifier_kwargs) -> None:
        """Construct the second classifier for MCD training."""
        # Import here to avoid import issues
        from PSZS.Models.models import construct_classifier
        # Construct a second classifier for mcd training
        self.classifier_2 = construct_classifier(classifier_type=self.classifier_type, 
                                                 head_type=self.head_type, 
                                                 num_classes=self.num_classes,
                                                 num_inputs=self.num_inputs,
                                                 in_features=self.feature_dim, 
                                                 **classifier_kwargs)
    
    
    def model_or_params(self, type: Optional[str]=None) -> Dict[str, Any]:
        """Return model or parameters for the optimizer.
        Either for the backbone (+ bottleneck) or the classifiers (base + 2x joint).
        Which one is returned is determined by the `type` parameter:
        - 'backbone': Return backbone (+ bottleneck) parameters
        - 'cls': Return classifier parameters (classifier + classifier_2)
        - None: Return all parameters (default)"""
        if type is None:
            return super().model_or_params()
        elif type == 'backbone':
            return [{"params": self.backbone.parameters()},
                    {"params": self.bottleneck.parameters()}]
        elif type == 'cls':
            return [{"params": self.classifier.parameters()},
                    {"params": self.classifier_2.parameters()}]
        else: 
            return super().model_or_params()
        
    def create_optimizers(self, **params) -> List[torch.optim.Optimizer]:
        """
        Create one optimizer for the backbone and one for the two classification heads.
        """
        return [create_optimizer_v2(model_or_params=self.model_or_params(type='backbone'), **params),
                create_optimizer_v2(model_or_params=self.model_or_params(type='cls'), **params)]
                    
    def forward(self, x: torch.Tensor) -> Tuple[Tuple[PRED_TYPE, PRED_TYPE], torch.Tensor] | torch.Tensor:
        """During training additionally return predictions of both classifiers."""
        f = self.backbone(x)
        f = self.bottleneck(f)
        if self.training:
            predictions_1 = self.classifier(f)
            predictions_2 = self.classifier_2(f)
            return (predictions_1, predictions_2), f
        else:
            predictions = self.classifier.forward_test(f)
            return predictions
        
    def val_test_state_dict(self) -> Dict[str, Any]:
        """"""
        val_test_state_dict = super().val_test_state_dict()
        val_test_state_dict.update({f'classifier_2.{k}':v for k,v in self.classifier_2.state_dict().items()})
        return val_test_state_dict