"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import List, Dict
import torch.nn as nn

class DomainDiscriminator(nn.Sequential):
    r"""Adapted from `~tllib.modules.domain_discriminator.DomainDiscriminator` to use 
    Logits for binary classification i.e. remove Sigmoid function to enable autocasting (AMP).
    Original implementation can be restored by setting `use_logits=False`.
    
    Domain discriminator model from
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Distinguish whether the input features come from the source domain or the target domain.
    The source domain label is 1 and the target domain label is 0.

    Args:
        in_feature (int): dimension of the input feature
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.

    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, 1)`
    """

    def __init__(self, 
                 in_feature: int, 
                 hidden_size: int, 
                 batch_norm: bool = True, 
                 sigmoid: bool = True,
                 use_logits: bool = True):
        if sigmoid:
            # When use_logits is True, the final layer will output logits instead of probabilities
            # This is useful when using BCEWithLogitsLoss to enable autocasting (AMP)
            # Thus the sigmoid function is removed
            if use_logits:
                final_layer = nn.Linear(hidden_size, 1)
            else:
                final_layer = nn.Sequential(
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
        else:
            final_layer = nn.Linear(hidden_size, 2)
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                final_layer
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                final_layer
            )

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]