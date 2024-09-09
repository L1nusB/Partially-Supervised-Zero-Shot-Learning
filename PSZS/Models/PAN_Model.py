from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import warnings
import numpy as np
import numpy.typing as npt

import torch
from torch.nn.modules import Module

from PSZS.Models.CustomModel import CustomModel
from PSZS.Alignment import BiLinearDomainAdversarialLoss, BilinearDomainDiscriminator
from PSZS.Models.funcs import mixing_progress

__all__ = ['PAN_Model']

class PAN_Model(CustomModel):
    def __init__(self, 
                 backbone: Module, 
                 num_classes: npt.NDArray[np.int_], 
                 num_inputs: int, 
                 classifier_type: str, 
                 opt_kwargs: Dict[str, Any],
                 sched_kwargs: Dict[str, Any],
                 hierarchy_accum_func: Callable[[Sequence[torch.Tensor]], torch.Tensor] | str = "coarse",
                 hidden_size: int = 1024,
                 label_smooth_strategy: str = "curriculum",
                 smooth_intial_coarse: float = 0.9,
                 smooth_final_coarse: float = 0.1,
                 max_smoothing_steps: Optional[int] = None,
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
        if isinstance(hierarchy_accum_func, str) == False and hierarchy_accum_func.startswith('coarse') == False:
            print(f"You are using PAN with hierarchy accumulation ({hierarchy_accum_func}) "
                    "which might cause duplicate loss for fine classes.")
            # warnings.warn(f"You are using PAN with hierarchy accumulation ({hierarchy_accum_func}) "
            #               "which might cause duplicate loss for fine classes. Consider using one of the "
            #               "'coarse' as hierarchy accumulation functions or make sure weights are set appropriately.")
        super().__init__(backbone=backbone, 
                         num_classes=num_classes,
                         num_inputs=num_inputs, 
                         classifier_type=classifier_type,
                         opt_kwargs=opt_kwargs,
                         sched_kwargs=sched_kwargs,
                         hierarchy_accum_func=hierarchy_accum_func, # Overwrite default 'sum' with 'coarse' for PAN
                         **additional_kwargs)
        # Ensure that multiple hierarchy levels exists otherwise PAN can not use any coarse levels
        assert num_classes.ndim >= 2, "Number of classes must be at least 2-dimensional"
        if self.head_type.returns_multiple_outputs==False or self.head_type.num_head_pred==1:
            raise TypeError("PAN requires a classifier that returns multiple outputs.")
            print("Dataset has multiple classes but the classification head only returns one output.")
        # self.num_hierarchy_levels = self.num_classes.shape[1]
        self.num_hierarchy_levels = self.num_pred
        self.num_coarse_levels = self.num_hierarchy_levels - 1
        self.num_fine_classes = self.classifier.largest_num_classes_test_lvl
        self.num_shared_classes = self.classifier.smallest_num_classes_test_lvl
        self.smoothing_strategy = self.resolve_mixing_strategy(label_smooth_strategy)
        self.smooth_initial_coarse = smooth_intial_coarse
        self.smooth_final_coarse = smooth_final_coarse
        self.smooth_step = 0
        # If None will be set in PAN_Multiple
        self.max_smoothing_steps = max_smoothing_steps

    def _register_da_components(self, **kwargs) -> None:
        # self.hidden_size = min(self.hidden_size, self.feature_dim)
        num_shared_classes = self.classifier.smallest_num_classes_test_lvl
        # self.num_shared_classes not intialized yet (called in super().__init__)
        bilin_domain_discriminator = BilinearDomainDiscriminator(in_feature1=self.feature_dim,
                                                                 in_feature2=num_shared_classes,
                                                                 hidden_size=self.hidden_size)
        # Do no construct a GRL layers as we just use default params for the GRL
        self.bilin_domain_adversarial_loss = BiLinearDomainAdversarialLoss(domain_discriminator_s=bilin_domain_discriminator,
                                                                           mode='simple')
    
    def smooth_labels(self, 
                     fine_target_one_hot: torch.Tensor, 
                     coarse_logits: Tuple[torch.Tensor, ...],
                     fine_to_coarse: List[Dict[int, int]]) -> torch.Tensor:
        # Fine_target needs to correspond to the prediction indices of the fine classes
        # Can just use the mixing progress function to get the lambda
        # as the hierarchy accum function should not be dynamic weighted for PAN
        smooth_progress = mixing_progress(self.smoothing_strategy, self.smooth_step, self.max_smoothing_steps)
        self.smooth_step += 1
        coarse_weight = self.smooth_initial_coarse + (self.smooth_final_coarse - self.smooth_initial_coarse) * smooth_progress
        detached_logits = [logits.detach() for logits in coarse_logits] # Remove from computation graph
        coarse_prediction_mix = torch.zeros(fine_target_one_hot.shape, 
                                            device=coarse_logits[0].device, 
                                            dtype=coarse_logits[0].dtype)
        assert len(coarse_logits) == self.num_coarse_levels, f"Number of coarse levels must match the number of coarse logits {len(coarse_logits)} != {self.num_coarse_levels}"
        for lvl in range(self.num_coarse_levels):
            # Only iterate over the fine classes that exist (for shared domain this is only a subset)
            for fine_class in range(fine_target_one_hot.size(1)):
                coarse_class = fine_to_coarse[lvl][fine_class]
                coarse_prediction_mix[:, fine_class] += detached_logits[lvl][:, coarse_class].softmax(dim=0)
        return (1 - coarse_weight) * fine_target_one_hot + coarse_weight * coarse_prediction_mix / self.num_coarse_levels
            
    
    def val_test_state_dict(self) -> Dict[str, Any]:
        val_test_state_dict = super().val_test_state_dict()
        val_test_state_dict.update({f'bilin_domain_adversarial_loss.{k}':v for k,v in self.bilin_domain_adversarial_loss.state_dict().items()})
        return val_test_state_dict
    
    # Check whether to use higher learning rate for domain discriminator?
    def model_or_params(self) -> Dict[str, Any]:
        params = [{"params": self.backbone.parameters(), 'weight_decay':0},
                  {"params": self.bottleneck.parameters()},
                  {"params": self.classifier.parameters()},
                  {'params': self.bilin_domain_adversarial_loss.bilin_domain_discriminator_s.parameters()}]
        if self.bilin_domain_adversarial_loss.bilin_domain_discriminator_o is not None:
            params.append({'params': self.bilin_domain_adversarial_loss.bilin_domain_discriminator_o.parameters()})
        # params.append({'params': self.bilin_domain_adversarial_loss.bilin_domain_discriminator.parameters(), 'lr': 1.})
        return params
        
    def get_model_kwargs(**kwargs) -> dict:
        """Dynamically resolve relevant kwargs for model construction."""
        # Need for unbound super call
        # https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        params, kwargs = super(PAN_Model, PAN_Model).get_model_kwargs(**kwargs)
        if 'hidden_size' in kwargs:
            params['hidden_size'] = kwargs.get('hidden_size')
            kwargs.pop('hidden_size')
        if 'label_smooth_strategy' in kwargs:
            params['label_smooth_strategy'] = kwargs.get('label_smooth_strategy')
            kwargs.pop('label_smooth_strategy')
        if 'smooth_intial_coarse' in kwargs:
            params['smooth_intial_coarse'] = kwargs.get('smooth_intial_coarse')
            kwargs.pop('smooth_intial_coarse')
        if 'smooth_final_coarse' in kwargs:
            params['smooth_final_coarse'] = kwargs.get('smooth_final_coarse')
            kwargs.pop('smooth_final_coarse')
        if 'max_smoothing_steps' in kwargs:
            params['max_smoothing_steps'] = kwargs.get('max_smoothing_steps')
            kwargs.pop('max_smoothing_steps')
        return params, kwargs