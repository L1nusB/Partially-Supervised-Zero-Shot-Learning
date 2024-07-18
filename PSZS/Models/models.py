# Needed to reference Base_Optimizer in type hints
# see https://stackoverflow.com/questions/39740632/python-type-hinting-without-cyclic-imports
from __future__ import annotations 
from argparse import Namespace
from typing import Optional, Type, TYPE_CHECKING
import warnings
import os
import shutil

import timm
import tllib.vision.models as models
import torch
import numpy as np
import numpy.typing as npt

from timm.optim import optimizer_kwargs
from timm.scheduler import scheduler_kwargs

from PSZS.Utils.io.logger import Logger
from PSZS.Models import METHOD_MODEL_MAP, CustomModel
from PSZS.Classifiers.Heads import CustomHead
from PSZS.Classifiers import CustomClassifier

if TYPE_CHECKING:
    from PSZS.Optimizer import Base_Optimizer

def get_model_type(method: str) -> Optional[Type[CustomModel]]:
    method = method.lower()
    for k in METHOD_MODEL_MAP.keys():
        if method == k.lower():
            return METHOD_MODEL_MAP[k.lower()]
    warnings.warn(f'Method {method} not in available methods: {list(METHOD_MODEL_MAP.keys())}')
    return None

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()
    
def load_checkpoint(model: torch.nn.Module,
                    checkpoint_path: str,
                    checkpoint_dir: str = "",
                    file_type: str = ".pth",
                    strict: bool = True):
    if checkpoint_path[-4:] != file_type:
        checkpoint_path = checkpoint_path + file_type
    if os.path.exists(checkpoint_path)==False:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
        assert os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path), f"Checkpoint {checkpoint_path} does not exist."
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    # Filter out the keys that are not in the model state dict
    state_dict : dict = torch.load(checkpoint_path)    
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(pretrained_dict, strict=strict)
    
def save_checkpoint(model: CustomModel, 
                    logger: Logger,
                    optimizer: Base_Optimizer,
                    metric: str,
                    current_best: float,
                    save_val_test: bool = False,):
    torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))
    if save_val_test:
        torch.save(model.val_test_state_dict(), logger.get_checkpoint_path('latest_val_test'))
    match metric:
        case 'acc1':
            cur_val = optimizer.cls_acc_1
        case 'acc5':
            cur_val = optimizer.cls_acc_5
        case 'f1':
            cur_val = optimizer.f1
        case 'f1_5':
            cur_val = optimizer.f1_5
        case 'precision':
            cur_val = optimizer.precision
        case 'recall':
            cur_val = optimizer.recall
        case '_':
            raise ValueError(f'Metric {metric} not supported.')
    # Save best model
    # Do not check for multiple metrics as if the metric stayed the same the change is not significant
    if cur_val > current_best or os.path.exists(logger.get_checkpoint_path('best'))==False:
        shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        if save_val_test:
            shutil.copy(logger.get_checkpoint_path('latest_val_test'), logger.get_checkpoint_path('best_val_test'))
    
# Contains the names of the components in the timm backbone models
# Split based on their depth in the model
# The main use will be to freeze the layers up to a certain depth
# The model matching will be done based on a 'startingwith' match to allow different variations of the same model
# that share the same structure
param_name_depth_map = {
    'resnet50': [['conv1','bn1','layer1'],['layer2'],['layer3'],['layer4']],
    'swinv2': [['patch_embed','layers.0'],['layers.1'],['layers.2'],['layers.3']],
}

def freeze_layers(model: torch.nn.Module, 
                  model_name: str,
                  freeze_depth: int):
    """Freeze the layers of a model up to a certain depth.
    
    Args:
        model (torch.nn.Module): Model to freeze layers of
        freeze_depth (int): Depth up to which to freeze the layers
    """
    print(f'Freezing layers of model {model_name} up to depth {freeze_depth}.')
    if freeze_depth == 0:
        # Freeze all layers
        print('Depth 0 specified. Freezing all layers of the model.')
        for param in model.parameters():
            param.requires_grad = False
        return
    matchingParams = [v for k,v in param_name_depth_map.items() if k.startswith(model_name)]
    if len(matchingParams) == 0:
        warnings.warn(f'No matching depth map found for model {model_name}.')
        return
    elif len(matchingParams) > 1:
        warnings.warn(f'Multiple matching depth maps found for model {model_name}. '
                        'Trying to freeze all matching layers.')
    freeze_names = [name for matchingParam in matchingParams for param_level in matchingParam[:freeze_depth] for name in param_level]
    # freeze_names = [name for param_level in matchingParams[0][:freeze_depth] for name in param_level]
    for name, param in model.named_parameters():
        if any([name.startswith(param_name) for param_name in freeze_names]):
            param.requires_grad = False

def load_backbone(model_name: str, 
                  args: Optional[Namespace] = None,
                  pretrain: bool = True,
                  source: str = "timm", 
                  timm_chk: Optional[str] = None,
                  drop_rate: float = 0.0,
                  drop_path_rate: Optional[float] = None,
                  drop_block_rate: Optional[float] = None,
                  freeze_depth: Optional[int] = None,
                ) -> torch.nn.Module:
    """Load a backbone model from timm or local models.
    
    Args:
        model_name (str): Name of timm model or local model
        args (Optional[Namespace]): Arguments for the model. \
            Overwrites values by other arguments. Defaults to None.
        source (str): Load model from timm or local. Defaults to "timm".
        timm_chk (Optional[str]): Checkpoint from which to load weights. Defaults to None.
        drop_rate (float): Classifier dropout rate for training.. Defaults to 0.0.
        drop_path_rate (Optional[float]): Stochastic depth drop rate for training. Defaults to None.
        drop_block_rate (Optional[float]): Drop block rate. Defaults to None.
        
    Returns:
        torch.nn.Module: Classifier model
    """
    if args is not None:
        source = 'local' if getattr(args, 'use_local', source=='local') else 'timm'
        timm_chk = getattr(args, 'timm_chk', timm_chk)
        drop_rate = getattr(args, 'drop', drop_rate)
        drop_path_rate = getattr(args, 'drop_path', drop_path_rate)
        drop_block_rate = getattr(args, 'drop_block', drop_block_rate)
        freeze_depth = getattr(args, 'freeze_backbone', freeze_depth)
    
    if source == "timm":
        # Need to preserve original model_name for freezing of layers
        expanded_model_name = model_name
        if timm_chk:
            expanded_model_name = f'{model_name}.{timm_chk}'
        backbone = timm.create_model(model_name=expanded_model_name, 
                                     pretrained=pretrain,
                                     drop_rate=drop_rate,
                                     drop_path_rate=drop_path_rate,
                                     drop_block_rate=drop_block_rate,)
        if freeze_depth is not None:
            freeze_layers(backbone, model_name, freeze_depth)
        
    elif source == "local":
        # Local models must support reset_classifier method. Otherwise an exception will be thrown
        if model_name in models.__dict__:
            # load models from tllib.vision.models
            backbone = models.__dict__[model_name](pretrained=pretrain)
        else:
            raise NotImplementedError(f'Model {model_name} not in local models.')
    else:
        raise NotImplementedError(f'Not supported source {source}')
    return backbone

    
def construct_classifier(classifier_type: Type[CustomClassifier], 
                         head_type: Type[CustomHead],
                         **kwargs) -> CustomClassifier:
    head_params = head_type.head_kwargs(**kwargs)
    classifier_params = classifier_type.classifier_kwargs(head_type=head_type, 
                                                     head_params=head_params,
                                                     **kwargs)
    return classifier_type(**classifier_params)

def build_model(backbone: torch.nn.Module,
                device: torch.device,
                num_classes: npt.NDArray[np.int_],
                num_inputs: int,
                args: Optional[Namespace] = None,
                classifier_type: Optional[str] = None,
                head_type: str = "SimpleHead",
                method: str = "erm",
                bottleneck_dim: Optional[int] = None,
                **model_and_classifier_kwargs) -> CustomModel:
    """Construct a Custom model with a given backbone and specified head
    Args:
        backbone (nn.Module): Backbone of the CustomModel
        device (torch.device): device to send the model to
        num_classes (np.ndarray): Number of classes. During model construction gets expanded to (num_inputs, num_head_predictions).
        num_inputs (int): Number of inputs to the classifier during training forward pass.
        args (Optional[Namespace]): Arguments for the model. \
            Overwrites values by other arguments. Defaults to None.
        classifier_type (Optional[str], optional): Classifier type for custom model. If None just use backbone. Defaults to None.
        head_type (str): Type of the head to use in custom model. Defaults to 'SimpleHead'.
        method (str): Type of the method used to resolve which model to build. Defaults to 'erm'.
        model_and_classifier_kwargs (dict): Keyword arguments for the Custom model, classifier and used head_type. \
            classifier_kwargs are filtered out and passed on.
        
        Relevant classifier_kwargs:
        - num_heads (Optional[int]): Number of heads
        - num_features (Optional[int]): Number of features before the head layer
        - auto_split (bool): Whether to automatically split the features into num_heads for separated classifiers
        - test_head_idx (int): Index of the head to use for testing/validation
        - test_head_pred_idx (Optional[int]): Index of the prediction the head produce to use for testing/validation
        - depth (int | Sequence[int]): Depth of the heads. Defaults to 1.
        - reduction_factor (float | Sequence[float]): Reduction factor for the heads in each depth step. Defaults to 2.
        - hierarchy_level (Optional[int]): Level of hierarchy for hierarchical head. (default: None)
            
    Returns:
       PSZS.Models.Custom Model: Classifier model
    """
    if args is not None:
        classifier_type = getattr(args, 'classification_type', classifier_type)
        head_type = getattr(args, 'head_type', head_type)
        method = getattr(args, 'method', method)
        bottleneck_dim = getattr(args, 'bottleneck_dim', bottleneck_dim)
        opt_kwargs = optimizer_kwargs(cfg=args)
        sched_kwargs = scheduler_kwargs(cfg=args)
        sched_kwargs.update({'updates_per_epoch':args.iters_per_epoch})
    else:
        opt_kwargs = {}
        sched_kwargs = {}
    
    model = get_model_type(method)
    assert model is not None, (f'No Model type for method {method} registered. '
                               f'Available models: {list(METHOD_MODEL_MAP.keys())}')
    # Allows for more complex model kwargs unpacking e.g. for Feature_Model
    # classifier_kwargs will only retain the filtered kwargs that were not used for model kwargs
    model_kwargs, classifier_kwargs = model.get_model_kwargs(**model_and_classifier_kwargs)
    print(f'Building model {model.__name__} based on method {method}.')
    if classifier_type is None:
        print('No classifier_type specified. Using default classifier (no differentiation for domains).')
        classifier_type = "DefaultClassifier"
    
    print(f'Using classifier_type ({classifier_type}) with head_type: ({head_type}) for Custom Model.')
    # Delete the head of the backbone and add a custom head via custom model
    # The pooling needs to be kept (or custom pooling needs to be added) to keep the feature size
    # TODO Add custom pooling
    # Local models must support reset_classifier method. Otherwise an exception will be thrown
    backbone.reset_classifier(0)
    return model(backbone=backbone, 
                classifier_type=classifier_type, 
                head_type=head_type,
                num_classes=num_classes, 
                num_inputs=num_inputs,
                bottleneck_dim=bottleneck_dim,
                opt_kwargs=opt_kwargs,
                sched_kwargs=sched_kwargs,
                **model_kwargs,
                **classifier_kwargs
                ).to(device)