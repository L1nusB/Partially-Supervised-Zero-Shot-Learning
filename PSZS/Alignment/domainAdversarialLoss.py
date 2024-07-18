"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from tllib.modules.grl import WarmStartGradientReverseLayer
from PSZS.Metrics import binary_accuracy, binary_accuracy_with_logits, accuracy

__all__ = ['DomainAdversarialLoss']


class DomainAdversarialLoss(nn.Module):
    r"""Adapted from `~tllib.alignment.dann.DomainAdversarialLoss`
    to use BCEWithLogitsLoss instead of binary cross entropy loss to enable autocasting (AMP).
    
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional[nn.Module] = None, binary_pred: bool =True, use_logits: bool = True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.binary_pred = binary_pred
        self.use_logits = use_logits
        # Depending on whether the model outputs logits or probabilities, we need to use the appropriate loss function
        # only relevant for binary_pred=True (i.e. binary classification) and needed to enable autocast
        # Generally using logits is numerically more stable
        # https://discuss.pytorch.org/t/bceloss-are-unsafe-to-autocast/110407
        if use_logits:
            self.binary_accuracy_func = binary_accuracy_with_logits
            self.binary_loss_func = F.binary_cross_entropy_with_logits
        else:
            self.binary_accuracy_func = binary_accuracy
            self.binary_loss_func = F.binary_cross_entropy
        self.reduction = reduction
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f: torch.Tensor = self.grl(torch.cat((f_s, f_t), dim=0))
        d: torch.Tensor = self.domain_discriminator(f)
        if self.binary_pred:
            d_s, d_t = d.tensor_split((f_s.size(0),), dim=0)
            d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
            num_features = f.size(0)
            # Scale the accuracy w.r.t. the number of features in source and target domain
            # If one domain is empty accuracy will be returned as 1
            # but due to scaling this is removed anyways after multiplication 
            # with the number of features (i.e. 0)
            self.domain_discriminator_accuracy = (self.binary_accuracy_func(d_s, d_label_s) * f_s.size(0) + 
                                                  self.binary_accuracy_func(d_t, d_label_t) * f_t.size(0)
                                                  ) / num_features

            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
                
            # Remove NaNs if features are empty
            source_loss = self.binary_loss_func(d_s, d_label_s, weight=w_s.view_as(d_s), 
                                                reduction=self.reduction).nan_to_num()
            
            target_loss = self.binary_loss_func(d_t, d_label_t, weight=w_t.view_as(d_t), 
                                                reduction=self.reduction).nan_to_num()
            # Scale the loss w.r.t. the number of features
            return (source_loss * f_s.size(0) + target_loss * f_t.size(0)) / num_features
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)