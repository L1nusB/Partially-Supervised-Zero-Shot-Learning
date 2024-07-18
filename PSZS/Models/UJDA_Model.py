import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import numpy.typing as npt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module

from timm.optim import create_optimizer_v2

from PSZS.Models.CustomModel import CustomModel, PRED_TYPE
from PSZS.Models.funcs import entropy_multiple, sum_accum

EXTENDED_PRED_TYPE = Tuple[Tuple[PRED_TYPE, torch.Tensor, torch.Tensor], torch.Tensor]

class UJDA_Model(CustomModel):
    """Custom Model for UJDA (Unsupervised Joint Domain Adaptation).
    Based on the paper "Unsupervised Joint Domain Adaptation with Class-wise Alignment" by Zhang et al.
    with some modifications unifiying the losses and not computing forward passes for each loss computation.
    """
    def __init__(self, 
                 backbone: Module, 
                 num_classes: npt.NDArray[np.int_], 
                 num_inputs: int, 
                 classifier_type: str, 
                 opt_kwargs: Dict[str, Any],
                 sched_kwargs: Dict[str, Any],
                 vat_xi: float = 1e-3,
                 vat_radius: float = 0.5,
                 **additional_kwargs) -> None:
        """
        Args:
        backbone (torch.nn.Module): Any backbone to extract 2-d features from data
        num_classes (np.ndarray): Number of classes. Gets expanded to (num_inputs, num_head_predictions).
        num_inputs (int): Number of inputs to the classifier during training forward pass.
        classifier_type (Optional[str]): Type of the classifier to use. Must be one of the available classifiers. If None, the model is the backbone itself
        opt_kwargs (dict): Keyword arguments for the optimizer(s)
        sched_kwargs (dict): Keyword arguments for the scheduler(s)
        vat_xi (float): VAT (Virtual Adversarial Training) xi parameter scaling the normalized random vector for computing \
            the perturbed inputs. Defaults to 1e-3.
        vat_radius (float): VAT (Virtual Adversarial Training) radius parameter scaling the returned perturbed inputs. Defaults to 0.5.
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
        self.vat_xi = vat_xi
        self.vat_radius = vat_radius
        super().__init__(backbone=backbone, 
                         num_classes=num_classes,
                         num_inputs=num_inputs, 
                         classifier_type=classifier_type,
                         opt_kwargs=opt_kwargs,
                         sched_kwargs=sched_kwargs,
                         **additional_kwargs)
    
    def _register_da_components(self, **classifier_kwargs) -> None:
        """Constructs and initializes the joint classifiers for the UJDA model.
        The activation and dropout function objects are also registered here."""
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # Default head depth is 2 for joint classifiers (i.e. one hidden layer)
        head_depth = classifier_kwargs.get('head_depth', 2)
        if isinstance(head_depth, int):
            head_depth = [head_depth, head_depth]
        assert len(head_depth) == 2, f"head_depth must be of length 2 but got {len(head_depth)}"
        # For joint classifiers the prediction is over joint distribution i.e. the largest number of classes
        if self.head_type.returns_multiple_outputs:
            # Pick out the head prediction index to use for testing
            self.out_features_parts = self.classifier.effective_head_pred_size[:,self.classifier.test_head_pred_idx]
        else:
            self.out_features_parts = self.classifier.effective_head_pred_size.squeeze(dim=1)
        self.out_features = sum(self.out_features_parts)
        # Can not use [[self.feature_dim]] *len(head_depth) as this will create a list of the same objects
        feature_nums = [[self.feature_dim] for _ in range(len(head_depth))]
        for i,hd in enumerate(head_depth):
            next_features = self.feature_dim
            # Input layer is already included in feature_nums while output layer is added later
            for _ in range(hd-1):
                if math.ceil(next_features / 2) >= self.out_features:
                    next_features = math.ceil(next_features / 2)
                else:
                    next_features = self.out_features
                feature_nums[i].append(next_features)
                
        features = [[nn.Linear(feature_nums[i][j], feature_nums[i][j+1]) for j in range(len(feature_nums[i])-1)] 
                    for i in range(len(head_depth))]
        
        # The activation function and dropout is inserted applied after each layer
        # and all layers are flattened into a single list
        # Same code as in hierarchical_head.py (just with dropout added)
        flattened_bodies = [[layer for part in body for layer in part] for body in 
                            [[[body, self.act, self.dropout] for body in layer] for layer in features]]
        # Add final layer to each body
        head_components = [body + [nn.Linear(feature_nums[i][-1], self.out_features)] for i, body in enumerate(flattened_bodies)]
        
        
        self.joint_classifier_1 = nn.Sequential(*head_components[0])
        self.joint_classifier_2 = nn.Sequential(*head_components[1]) 
        
        # Set some initial weights
        for dep in range(len(head_depth)):
            self.joint_classifier_1[dep * 3].weight.data.normal_(0, 0.01)
            self.joint_classifier_1[dep * 3].bias.data.fill_(0.0)
            self.joint_classifier_2[dep * 3].weight.data.normal_(0, 0.01)
            self.joint_classifier_2[dep * 3].bias.data.fill_(0.0)
        
    def forward(self, x: torch.Tensor) -> EXTENDED_PRED_TYPE | Tuple[PRED_TYPE, torch.Tensor]:
        """During training additionally return predictions of an adversarial classifier."""
        f = self.backbone(x)
        f = self.bottleneck(f)
        if self.training:
            predictions_cls = self.classifier(f)
            # Prediction needs to be split according to the respective batch sizes
            # For main classifier this is handled in the classifier itself
            # predictions_joint_1 = self.joint_classifier_1(f).tensor_split((predictions_cls[0].size(0),))
            predictions_joint_1 = self.joint_classifier_1(f)
            # predictions_joint_2 = self.joint_classifier_2(f).tensor_split((predictions_cls[0].size(0),))
            predictions_joint_2 = self.joint_classifier_2(f)
            return (predictions_cls, predictions_joint_1, predictions_joint_2), f
        else:
            predictions = self.classifier.forward_test(f)
            return predictions
        
    def _discrepancy(self, out1: torch.Tensor, out2: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(F.softmax(out1, dim =1) - F.softmax(out2, dim = 1)))
    
    def vat(self, inputs: torch.Tensor, radius: Optional[float]=None) -> torch.Tensor:
        if radius is None:
            radius = self.vat_radius
        rand = torch.randn(inputs.data.size())
        eps = self.vat_xi * (rand/torch.norm(rand,dim=(2,3),keepdim=True)).to(inputs.device)
        eps.requires_grad_(True)
        (cls_pred_1, _, _), _ = self(inputs)
        (cls_pred_2, _, _), _ = self(inputs + eps)
        if self.head_type.returns_multiple_outputs:
            cls_pred_1 = [p1[self.classifier.test_head_pred_idx] for p1 in cls_pred_1]
            cls_pred_2 = [p2[self.classifier.test_head_pred_idx] for p2 in cls_pred_2]
        loss_s = self._discrepancy(cls_pred_1[0],cls_pred_2[0])
        loss_t = self._discrepancy(cls_pred_1[1],cls_pred_2[1])
        loss = loss_s + loss_t
        # Instead of calling loss on loss_s and loss_t separately which might delete parts
        # of the gradient for the second call, we call it on the sum of both
        loss.backward(retain_graph=True)

        eps_adv = eps.grad
        eps_adv = eps_adv/torch.norm(eps_adv)
        # If the norm of the gradient is 0 we would have NaNs which are replaced by 0
        image_adv = inputs + radius * (eps_adv).nan_to_num()
        return image_adv
    
    def compute_loss_vat(self,
                         pred_cls: Sequence[torch.Tensor],
                         pred_cls_vat: Sequence[torch.Tensor]) -> torch.Tensor:
        return sum([self._discrepancy(pred_cls[i], pred_cls_vat[i]) for i in range(len(pred_cls))])
    
    def compute_joint_loss_labeled(self, 
                                   pred_joint_1: torch.Tensor,
                                   pred_joint_2: torch.Tensor,
                                   target: Tuple[torch.Tensor, torch.Tensor]
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        # This will compute the loss using also labeled target data
        # This differs from the original UJDA which only uses source data
        # as that is the only available labeled data
        # However the target data needs to correspond to the second half of the joint distribution
        full_target = torch.cat((target[0], target[1] + self.out_features_parts[0]), dim=0)
        loss_1 = self.cls_loss_func(pred_joint_1, full_target)
        loss_2 = self.cls_loss_func(pred_joint_2, full_target)
        return loss_1, loss_2
    
    def compute_joint_loss_unlabeled(self, 
                                     pred_cls: Sequence[torch.Tensor],
                                     pred_joint_1: torch.Tensor,
                                     pred_joint_2: torch.Tensor,
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Effectively add num_classes (source) to the pseudo labels to match the joint classifier output for target
        # As the joint classifiers are over the joint distribution i.e. sum over number of classes
        # and the "unlabeled" data corresponds to the target data and thus to the second half of the joint distribution
        pseudo_labels = torch.cat((torch.argmax(pred_cls[0], dim=1), 
                                   torch.argmax(pred_cls[1], dim=1) + self.out_features_parts[0]), 
                                  dim=0)
        loss_1 = self.cls_loss_func(pred_joint_1, pseudo_labels)
        loss_2 = self.cls_loss_func(pred_joint_2, pseudo_labels)
        return loss_1, loss_2
    
    def compute_loss_entropy(self, pred_cls: torch.Tensor) -> torch.Tensor:
        return sum_accum(entropy_multiple([pred.softmax(dim=1) for pred in pred_cls]))
    
    def compute_loss_adv(self, 
                         pred_cls: Sequence[torch.Tensor],
                         pred_joint_1: torch.Tensor, 
                         pred_joint_2: torch.Tensor,
                         target: Tuple[torch.Tensor, torch.Tensor]
                         ) -> torch.Tensor:
        # Only allow elements that correspond to shared classes (target) 
        # otherwise we will run out of bounds
        # Note that this is(should be) no limitation as all target data will be included anyways
        # and the original UJDA only computes this over target samples
        valid_target_mask = torch.tensor([t < self.out_features_parts[1] for t in torch.cat(target)])
        adv_target = torch.cat((target[0][valid_target_mask[:len(target[0])]] + self.out_features_parts[0], target[1]), dim=0)
        loss_1_l = self.cls_loss_func(pred_joint_1[valid_target_mask], adv_target)
        loss_2_l = self.cls_loss_func(pred_joint_2[valid_target_mask], adv_target)
        
        # Effectively add num_classes to the pseudo labels to match the joint classifier output
        # As the joint classifiers are over the joint distribution i.e. sum over number of classes
        # and the "unlabeled" data corresponds to the target data and thus to the second half of the joint distribution
        adv_pseudo_target = torch.cat((torch.argmax(pred_cls[0], dim=1) + self.out_features_parts[0], 
                                       torch.argmax(pred_cls[1], dim=1)), 
                                      dim=0)
        valid_pseudo_target_mask = torch.tensor([pt < self.out_features for pt in adv_pseudo_target])
        loss_1_ul = self.cls_loss_func(pred_joint_1[valid_pseudo_target_mask], 
                                        adv_pseudo_target[valid_pseudo_target_mask])
        loss_2_ul = self.cls_loss_func(pred_joint_2[valid_pseudo_target_mask], 
                                        adv_pseudo_target[valid_pseudo_target_mask])
        
        return loss_1_l + loss_2_l + loss_1_ul + loss_2_ul
    
    def compute_loss_discrepancy(self,
                                 pred_joint_1: torch.Tensor, 
                                 pred_joint_2: torch.Tensor,
                                 ) -> torch.Tensor:
        # One could/should? multiply by 2 as _discrepancy is already the mean
        # and we compute it over source + target
        # For faithful reproduction of the paper implementation a *2 would be needed
        # as they compute it separately for source and target and add it together
        return self._discrepancy(pred_joint_1, pred_joint_2)
        
        
    def model_or_params(self, cls_type: Optional[str]=None) -> Dict[str, Any]:
        """Return model or parameters for the optimizer.
        Either for the backbone (+ bottleneck) or the classifiers (base + 2x joint).
        Which one is returned is determined by the `cls_type` parameter:
        - 'backbone': Return backbone (+ bottleneck) parameters
        - 'cls': Return base classifier parameters
        - 'joint_cls_1': Return joint classifier 1 parameters
        - 'joint_cls_2': Return joint classifier 2 parameters
        - None: Return all parameters (default)"""
        if cls_type is None:
            return super().model_or_params()
        elif cls_type == 'backbone':
            return [{"params": self.backbone.parameters()},
                    {"params": self.bottleneck.parameters()}]
        elif cls_type == 'cls':
            return [{"params": self.classifier.parameters()}]
        elif cls_type == 'joint_cls_1':
            return [{"params": self.joint_classifier_1.parameters()}]
        elif cls_type == 'joint_cls_2':
            return [{"params": self.joint_classifier_2.parameters()}]
        else: 
            return super().model_or_params()
        
    def create_optimizers(self, **params) -> List[torch.optim.Optimizer]:
        """
        Create one optimizer for each classification head.
        """
        return [create_optimizer_v2(model_or_params=self.model_or_params(cls_type='backbone'), **params),
                create_optimizer_v2(model_or_params=self.model_or_params(cls_type='cls'), **params),
                create_optimizer_v2(model_or_params=self.model_or_params(cls_type='joint_cls_1'), **params),
                create_optimizer_v2(model_or_params=self.model_or_params(cls_type='joint_cls_2'), **params)]
    
    def val_test_state_dict(self) -> Dict[str, Any]:
        """"""
        val_test_state_dict = super().val_test_state_dict()
        val_test_state_dict.update({f'joint_classifier_1.{k}':v for k,v in self.joint_classifier_1.state_dict().items()})
        val_test_state_dict.update({f'joint_classifier_2.{k}':v for k,v in self.joint_classifier_2.state_dict().items()})
        return val_test_state_dict
        
    def get_model_kwargs(**kwargs) -> dict:
        """Dynamically resolve relevant kwargs for model construction.
        """
        params, kwargs = super(UJDA_Model, UJDA_Model).get_model_kwargs(**kwargs)
        if 'head_depth' in kwargs:
            params['head_depth'] = kwargs.get('head_depth')
            kwargs.pop('head_depth')
        return params, kwargs