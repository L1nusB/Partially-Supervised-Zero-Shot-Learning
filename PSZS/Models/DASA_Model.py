import copy
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.nn.modules import Module

from timm.optim import create_optimizer_v2
from timm.layers.classifier import _create_pool

from PSZS.Models.CustomModel import CustomModel, PRED_TYPE
from PSZS.Alignment import DomainDiscriminator, DomainAdversarialLoss, WarmStartGradientReverseLayer

class DASA_Model(CustomModel):
    class ClassAligner(Module):
        def __init__(self, margin=1):
            super().__init__()
            self.margin = margin
            
        def forward(self, feat1: torch.Tensor, 
                    feat2: torch.Tensor, 
                    label1: torch.Tensor, 
                    label2: torch.Tensor) -> torch.Tensor:
            # cosine = F.cosine_similarity()
            # cosine = torch.nn.CosineSimilarity(dim=0)
            result = 0
            num = len(label1) * len(label2)
            if num == 0:
                return torch.tensor(1)
            for i in range(0, len(label1)):
                for j in range(0, len(label2)):
                    if label1[i] == label2[j]:
                        result += max(self.margin - F.cosine_similarity(feat1[i], feat2[j], dim=0), 0)
                    else:
                        result += F.cosine_similarity(feat1[i], feat2[j], dim=0)

            return result / num
        
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
                 grl_max_iters: Optional[int] = None,
                 margin: float = 1,
                 sa_weight: float = 0.2,
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
        grl_max_iters (int): Maximum number of iterations for the gradient reversal layer. If not given will be set as multiple of epochs in Optimizer. Defaults to None.
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
        # Will default resolve to 1000 in WarmStartGradientReverseLayer even if None is pased
        # but enables setting to multiple of epochs in optimizers
        self.grl_max_iters = grl_max_iters 
        self.margin = margin
        self.sa_weight = sa_weight
        super().__init__(backbone=backbone, 
                         num_classes=num_classes,
                         num_inputs=num_inputs, 
                         classifier_type=classifier_type,
                         opt_kwargs=opt_kwargs,
                         sched_kwargs=sched_kwargs,
                         **additional_kwargs)
        # Create a second backbone for the SA module
        # can not be done before super call as we need to initialize the module class itself first
        # Calling a bounded super method is not possible as the init within the super class
        # will reset the module again.
        self.backbone2 = copy.deepcopy(backbone)
        # Need to rebuild optimizers and schedulers with the new backbone
        self.optimizers = self.create_optimizers(**opt_kwargs)
        self.lr_schedulers = self.create_schedulers(**sched_kwargs)
        
    @property
    def feature_extractor(self) -> Module:
        return self.backbone
    
    @property
    def feature_extractor_SA(self) -> Module:
        if hasattr(self, 'backbone2'):
            return self.backbone2
        else:
            return None
        
    def _register_da_components(self, **kwargs) -> None:
        """Construct the domain adversarial loss.
        In the process the domain discriminator and gradient reversal layer are constructed
        but not registered as model components.
        Hidden size is updated to be smaller than the feature dimension if necessary."""
        # Try to use given hidden layer as long as sufficient features are available
        # or half of the feature dimension
        self.hidden_size = self.hidden_size if self.feature_dim > self.hidden_size else int(self.feature_dim / 2)
        domain_discriminator = DomainDiscriminator(in_feature=self.feature_dim,
                                                   hidden_size=self.hidden_size)
        grl = WarmStartGradientReverseLayer(alpha=self.grl_alpha, 
                                            lo=self.grl_lo, 
                                            hi=self.grl_hi, 
                                            max_iters=self.grl_max_iters, 
                                            auto_step=True)
        # No need to send to device as this is done in models.py when constructing the model
        self.domain_adversarial_loss = DomainAdversarialLoss(domain_discriminator=domain_discriminator,
                                                             grl=grl)
        self.class_aligner = self.ClassAligner(margin=self.margin)
        
    def _crop_batch(self, data: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        dtype = features.dtype
        device = features.device
        croppedBatch = []
        features = features.detach().cpu().numpy()
        
        for batch_index in range(len(data)):
            # Average has already been applied to the features
            # Crop values at 0
            np.maximum(features[batch_index], 0, out=features[batch_index])
            # Original Code
            tmp_heat = features[batch_index] / np.max(features[batch_index])
            tmp_heatmap = np.uint8(255 * tmp_heat)
            _, binary = cv2.threshold(tmp_heatmap, 127, 255, cv2.THRESH_BINARY)
            
            contours, _2 = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                tmp_max = -1
                tmp_maxi = -1
                for i in range(len(contours)):
                    cnt = contours[i]
                    _, _, w, h = cv2.boundingRect(cnt)
                    if w * h > tmp_max:
                        tmp_max = w * h
                        tmp_maxi = i
                tmpx, tmpy, tmpw, tmph = cv2.boundingRect(contours[tmp_maxi])
                tmpx1, tmpy1, tmpx2, tmpy2 = int(tmpx * 227 / tmp_heat.shape[0]), int(
                    tmpy * 227 / tmp_heat.shape[0]), int(
                    math.ceil((tmpx + tmpw) * 227 / tmp_heat.shape[0])), int(
                    math.ceil((tmpy + tmph) * 227 / tmp_heat.shape[0]))

                tmp_img = data[batch_index].detach().cpu().numpy().transpose(1, 2, 0)
                tmp_img = Image.fromarray(np.uint8(tmp_img))
                tmp_bbox = (tmpx1, tmpy1, tmpx2, tmpy2)
                tmp_bbox = tuple(tmp_bbox)
                tmp_img = tmp_img.crop(tmp_bbox).resize((227, 227))
                tmpiimg = np.asarray(tmp_img)
            else:
                tmpiimg = data[batch_index].detach().cpu().numpy().transpose(1, 2, 0)

            croppedBatch.append(tmpiimg)

        croppedBatch = np.array(croppedBatch).transpose(0, 3, 1, 2)
        return torch.tensor(croppedBatch, dtype=dtype, device=device)
        
    def forward(self, x: torch.Tensor) -> Tuple[PRED_TYPE, torch.Tensor] | torch.Tensor:
        """During training computes the predictions using the SA module.
        The SA module performs a masking operation on the features
        and returns a weighted combination of the predictions of the original features
        and masked features based on `sa_weight`."""
        f = self.feature_extractor(x)
        f = self.bottleneck(f)
        if self.training:
            masked_features = self.feature_extractor_SA(self._crop_batch(data=x, features=f))
            masked_features = self.bottleneck(masked_features)
            
            predictions = self.classifier(f)
            sa_predictions = self.classifier(masked_features)
            
            if self.classifier.num_inputs > 1:
                if self.classifier.returns_multiple_outputs:
                    # Hierarchical head with multiple inputs returns list of list of predictions
                    return [[self.sa_weight * sa_pred + (1-self.sa_weight)*pred for sa_pred, pred in zip(sa_head, head)] 
                            for sa_head, head in zip(sa_predictions, predictions)], f
                else:
                    # Non-hierarchical head with multiple inputs returns list of predictions
                    return [self.sa_weight * sa_pred + (1-self.sa_weight)*pred for sa_pred, pred in zip(sa_predictions, predictions)], f
            else:
                if self.classifier.returns_multiple_outputs:
                    # Hierarchical head with single input returns list of predictions
                    return [self.sa_weight * sa_pred + (1-self.sa_weight)*pred for sa_pred, pred in zip(sa_predictions, predictions)], f
                else:
                    # Non-hierarchical head with single input returns single prediction
                    return self.sa_weight * sa_predictions + (1-self.sa_weight)*predictions, f
        else:
            predictions = self.classifier.forward_test(f)
            return predictions
    
    def val_test_state_dict(self) -> Dict[str, Any]:
        val_test_state_dict = super().val_test_state_dict()
        val_test_state_dict.update({f'domain_adversarial_loss.{k}':v for k,v in self.domain_adversarial_loss.state_dict().items()})
        return val_test_state_dict
    
    def model_or_params(self, param_type: Optional[str] = None) -> Dict[str, Any]:
        """Return model or parameters for the optimizer.
        Either for the backbones + classifier or the domain discriminator."""
        if param_type is None:
            return super().model_or_params()
        if param_type=='cls':
            params = [{"params": self.backbone.parameters(), 'weight_decay':0},
                        {"params": self.bottleneck.parameters()},
                        {"params": self.classifier.parameters()}]
            # Add SA module if available (not during super().__init__)
            if self.feature_extractor_SA is not None:
                params.append({"params": self.feature_extractor_SA.parameters(), 'weight_decay':0})
            return params
        elif param_type=='domain':
            return [{"params": self.domain_adversarial_loss.domain_discriminator.parameters()}]
        else: 
            return super().model_or_params()
    
    def create_optimizers(self, **params) -> List[torch.optim.Optimizer]:
        """
        Create one optimizer for each classification head.
        """
        return [create_optimizer_v2(model_or_params=self.model_or_params(param_type='cls'), **params),
                create_optimizer_v2(model_or_params=self.model_or_params(param_type='domain'), **params),]
        
    def get_model_kwargs(**kwargs) -> dict:
        """Dynamically resolve relevant kwargs for model construction."""
        # Need for unbound super call
        # https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        params, kwargs = super(DASA_Model, DASA_Model).get_model_kwargs(**kwargs)
        if 'hidden_size' in kwargs:
            params['hidden_size'] = kwargs.get('hidden_size')
            kwargs.pop('hidden_size')
        if 'grl_alpha' in kwargs:
            params['grl_alpha'] = kwargs.get('grl_alpha')
            kwargs.pop('grl_alpha')
        if 'grl_lo' in kwargs:
            params['grl_lo'] = kwargs.get('grl_lo')
            kwargs.pop('grl_lo')
        if 'grl_hi' in kwargs:
            params['grl_hi'] = kwargs.get('grl_hi')
            kwargs.pop('grl_hi')
        if 'grl_max_iters' in kwargs:
            params['grl_max_iters'] = kwargs.get('grl_max_iters')
            kwargs.pop('grl_max_iters')
        if 'margin' in kwargs:
            params['margin'] = kwargs.get('margin')
            kwargs.pop('margin')
        if 'sa_weight' in kwargs:
            params['sa_weight'] = kwargs.get('sa_weight')
            kwargs.pop('sa_weight')
        return params, kwargs