from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from PSZS.Metrics import binary_accuracy_with_logits
from PSZS.Alignment import WarmStartGradientReverseLayer, BilinearDomainDiscriminator

__all__ = ['BiLinearDomainAdversarialLoss']


class BiLinearDomainAdversarialLoss(nn.Module):
    r"""Domain Adversarial Loss for Bilinear Domain Discriminator.
    Measure the binary domain discrepancy through training a domain discriminator.
    
    Inspired by `~tllib.alignment.dann.DomainAdversarialLoss`
    using BCEWithLogitsLoss instead of binary cross entropy loss to enable autocasting (AMP).
    
    Args:
        domain_discriminator (BilinearDomainDiscriminator): 
            A bilinear domain discriminator object, which predicts the domains of features.
        reduction (str, optional): 
            Specifies the reduction to apply to the output: `none` | `mean` | `sum`. Default: `none`
        grl (WarmStartGradientReverseLayer, optional): 
            Warmstart Gradient reversal layer. If not specified constructs a new one. Default: None.

    Inputs:
        - f_s (torch.Tensor): feature representations on source domain
        - norm_logits_s (torch.Tensor): normalized logits for source domain
        - f_t (torch.Tensor): feature representations on target domain
        - norm_logits_t (torch.Tensor): normalized logits for target domain
        - w_s (torch.Tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (torch.Tensor, optional): a rescaling weight given to each instance from target domain.
    """

    def __init__(self, 
                 domain_discriminator: BilinearDomainDiscriminator, 
                 reduction: Optional[str] = 'mean',
                 grl: Optional[nn.Module] = None):
        super().__init__()
        if reduction is None:
            reduction = "none"
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.bilin_domain_discriminator = domain_discriminator
        # Generally using logits is numerically more stable and enables autocasting
        # https://discuss.pytorch.org/t/bceloss-are-unsafe-to-autocast/110407
        self.binary_accuracy_func = binary_accuracy_with_logits
        self.binary_loss_func = F.binary_cross_entropy_with_logits
        self.reduction = reduction
        self.discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, 
                norm_logits_s: torch.Tensor, 
                f_t: torch.Tensor,
                norm_logits_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None,
                w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f: torch.Tensor = self.grl(torch.cat((f_s, f_t), dim=0))
        d: torch.Tensor = self.bilin_domain_discriminator(f, torch.cat((norm_logits_s, norm_logits_t), dim=0))
        d_s, d_t = d.tensor_split((f_s.size(0),), dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        num_features = f.size(0)
        # Scale the accuracy w.r.t. the number of features in source and target domain
        # If one domain is empty accuracy will be returned as 1
        # but due to scaling this is removed anyways after multiplication 
        # with the number of features (i.e. 0)
        self.discriminator_accuracy = (self.binary_accuracy_func(d_s, d_label_s) * f_s.size(0) + 
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