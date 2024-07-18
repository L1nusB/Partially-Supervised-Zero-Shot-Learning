from typing import Any, Dict, Optional
import numpy as np
import numpy.typing as npt

from torch.nn.modules import Module
# from tllib.alignment.cdan import ConditionalDomainAdversarialLoss
# from tllib.modules.domain_discriminator import DomainDiscriminator

from PSZS.Models.CustomModel import CustomModel
from PSZS.Alignment import DomainDiscriminator, ConditionalDomainAdversarialLoss

class Cond_Domain_Adv_Model(CustomModel):
    """Custom Model using the ConditionalDomainAdversarialLoss and DomainDiscriminator from tllib.
    Used for CDAN.
    
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
                 entropy_conditioning: bool = False,
                 randomized: bool = False,
                 randomized_dim: int = 1024,
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
        entropy_conditioning (bool): Whether to condition the adversarial loss on the entropy of the source domain predictions. Defaults to False.
        randomized (bool): Whether to use randomized features for the domain discriminator. Defaults to False.
        randomized_dim (int): Dimension of the randomized features when using randomized multi-linear map. Defaults to 1024.
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
        self.entropy_conditioning = entropy_conditioning
        self.randomized = randomized
        self.randomized_dim = randomized_dim
        super().__init__(backbone=backbone, 
                         num_classes=num_classes,
                         num_inputs=num_inputs, 
                         classifier_type=classifier_type,
                         opt_kwargs=opt_kwargs,
                         sched_kwargs=sched_kwargs,
                         **additional_kwargs)
    
    def _register_da_components(self, **kwargs) -> None:
        """Construct the conditional domain adversarial loss.
        In the process the domain discriminator and gradient reversal layer are constructed
        but not registered as model components."""
        # Randomized dimension should be at least the hidden size
        self.randomized_dim = self.randomized_dim if self.randomized_dim > self.hidden_size else self.hidden_size
        # Get the effective head prediction size i.e. the number of classes for the main hierarchy level
        # Use max which should correspond to the source anyways (otherwise it was explicitly allowed or warned against)
        # via allow_mix_num_classes or allow_non_strict_num_classes_order in get_largest_head_num_classes() in custom_classifier.py
        dd_num_classes = self.classifier.largest_num_classes_test_lvl()
        
        if self.randomized:
            self.hidden_size = self.hidden_size if self.randomized_dim > self.hidden_size else int(self.randomized_dim / 2)
            domain_discriminator = DomainDiscriminator(in_feature=self.randomized_dim,
                                                        hidden_size=self.hidden_size)
        else:
            # Try to use given hidden layer as long as sufficient features are available
            # or half of the feature dimension
            self.hidden_size = self.hidden_size if self.feature_dim*dd_num_classes > self.hidden_size else int(self.feature_dim / 2)
            domain_discriminator = DomainDiscriminator(in_feature=self.feature_dim*dd_num_classes,
                                                       hidden_size=self.hidden_size)
        # No need to send to device as this is done in models.py when constructing the model
        self.cond_domain_adversarial_loss = ConditionalDomainAdversarialLoss(domain_discriminator=domain_discriminator,
                                                                             entropy_conditioning=self.entropy_conditioning,
                                                                             randomized=self.randomized,
                                                                             num_classes=dd_num_classes,
                                                                             features_dim=self.feature_dim,
                                                                             randomized_dim=self.randomized_dim)
    
    def val_test_state_dict(self) -> Dict[str, Any]:
        """"""
        val_test_state_dict = super().val_test_state_dict()
        val_test_state_dict.update({f'cond_domain_adversarial_loss.{k}':v for k,v in self.cond_domain_adversarial_loss.state_dict().items()})
        return val_test_state_dict
        
    def get_model_kwargs(**kwargs) -> dict:
        """Dynamically resolve relevant kwargs for model construction.
        """
        params, kwargs = super(Cond_Domain_Adv_Model, Cond_Domain_Adv_Model).get_model_kwargs(**kwargs)
        if 'hidden_size' in kwargs:
            params['hidden_size'] = kwargs.get('hidden_size')
            kwargs.pop('hidden_size')
        if 'entropy_conditioning' in kwargs:
            params['entropy_conditioning'] = kwargs.get('entropy_conditioning')
            kwargs.pop('entropy_conditioning')
        if 'randomized' in kwargs:
            params['randomized'] = kwargs.get('randomized')
            kwargs.pop('randomized')
        if 'randomized_dim' in kwargs:
            params['randomized_dim'] = kwargs.get('randomized_dim')
            kwargs.pop('randomized_dim')
        return params, kwargs