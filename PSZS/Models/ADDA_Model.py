from typing import Any, Dict, List, Optional
import numpy as np
import numpy.typing as npt

import torch
from torch.nn.modules import Module

from tllib.modules.grl import WarmStartGradientReverseLayer

from timm.optim import create_optimizer_v2

from PSZS.Models.CustomModel import CustomModel
from PSZS.Alignment import DomainDiscriminator, DomainAdversarialLoss

class ADDA_Model(CustomModel):
    """Custom Model implementing ADDA.
    The pretraining phase where the classifier is trained can be skipped by setting `skip_cls_train` to True.
    In this case only the domain adaptation phase is performed.
    
    Inputs:
        - x (tensor): input data fed to `backbone`
    """
    # Needs to be true as a model trained via erm will not have a domain discriminator 
    # and thus the state_dicts do not match
    allow_non_strict_checkpoint = True
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
                 grl_hi: float = 2.,
                 grl_max_iters: int = 1000,
                 skip_cls_train: bool = False,
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
        skip_cls_train (bool): Whether to skip the classifier training and directly start the domain adversarial phase. \
            Defaults to False.
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
        self.skip_cls_train = skip_cls_train
        if self.skip_cls_train:
            print("Skipping classifier training. Make sure to load a checkpoint for classification results.")
        
        super().__init__(backbone=backbone, 
                         num_classes=num_classes,
                         num_inputs=num_inputs, 
                         classifier_type=classifier_type,
                         opt_kwargs=opt_kwargs,
                         sched_kwargs=sched_kwargs,
                         **additional_kwargs)
        
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
        # No need to send to device as this is done in models.py when constructing the model
        grl = WarmStartGradientReverseLayer(alpha=self.grl_alpha, lo=self.grl_lo, 
                                            hi=self.grl_hi, max_iters=self.grl_max_iters, 
                                            auto_step=True)
        self.domain_adversarial_loss = DomainAdversarialLoss(domain_discriminator=domain_discriminator,
                                                             grl=grl)
        
    def val_test_state_dict(self) -> Dict[str, Any]:
        """"""
        val_test_state_dict = super().val_test_state_dict()
        val_test_state_dict.update({f'domain_adversarial_loss.{k}':v for k,v in self.domain_adversarial_loss.state_dict().items()})
        return val_test_state_dict
    
    def create_optimizers(self, **params) -> List[torch.optim.Optimizer]:
        """
        Depending on whether the classifier training is skipped or not,
        create one optimizer for the classification phase and one for the domain adaptation phase
        or just one optimizer for the domain adaptation phase.
        """
        if self.skip_cls_train:
            return [create_optimizer_v2(model_or_params=self.model_or_params(phase='domain'), **params)]
        else:
            return [create_optimizer_v2(model_or_params=self.model_or_params(phase='cls'), **params),
                    create_optimizer_v2(model_or_params=self.model_or_params(phase='domain'), **params)]
    
    # Check whether to use higher learning rate for domain discriminator
    def model_or_params(self, phase: Optional[str]=None) -> Dict[str, Any]:
        """Return model or parameters for the optimizer.
        Either for the classifier for the pretraining phase (phase='cls')
        or for the domain discriminator for the domain adaptation phase (phase='domain')."""
        if phase is None:
            return super().model_or_params()
        if phase=='cls':
            return [{"params": self.backbone.parameters(), 'weight_decay':0},
                    {"params": self.bottleneck.parameters()},
                    {"params": self.classifier.parameters()}]
        elif phase=='domain':
            # Backbone becomes explicit weight_decay=0 to avoid weight decay on the backbone (consistency with all other methods)
            return [{"params": self.backbone.parameters(), 'weight_decay':0},
                    {"params": self.bottleneck.parameters()},
                    {"params": self.domain_adversarial_loss.domain_discriminator.parameters()}]
                    # {"params": self.domain_adversarial_loss.domain_discriminator.parameters(), 'lr': 1.}]
        else: 
            return super().model_or_params()
        
        
    def get_model_kwargs(**kwargs) -> dict:
        """Dynamically resolve relevant kwargs for model construction.
        """
        params, kwargs = super(ADDA_Model, ADDA_Model).get_model_kwargs(**kwargs)
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
        if 'skip_cls_train' in kwargs:
            params['skip_cls_train'] = kwargs.get('skip_cls_train')
            kwargs.pop('skip_cls_train')
        return params, kwargs