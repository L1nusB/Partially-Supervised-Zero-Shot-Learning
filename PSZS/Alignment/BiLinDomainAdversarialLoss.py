from typing import Literal, Optional
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

    Inputs:
        - f_s (torch.Tensor): feature representations on source domain
        - norm_logits_s (torch.Tensor): normalized logits for source domain
        - f_t (torch.Tensor): feature representations on target domain
        - norm_logits_t (torch.Tensor): normalized logits for target domain
        - w_s (torch.Tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (torch.Tensor, optional): a rescaling weight given to each instance from target domain.
    """

    def __init__(self, 
                 domain_discriminator_s: BilinearDomainDiscriminator, 
                 domain_discriminator_o: Optional[BilinearDomainDiscriminator] = None, 
                 reduction: Optional[str] = 'mean',
                 grl: Optional[nn.Module] = None,
                 mask: Optional[torch.Tensor] = None,
                 mode: Literal['simple', 'classes', 'domains'] = 'simple'):
        """
        Args:
        domain_discriminator_s (BilinearDomainDiscriminator): 
            Main bilinear domain discriminator object for source/shared/simple.
        domain_discriminator_o (Optional[BilinearDomainDiscriminator]): 
            Other/Additional bilinear domain discriminator when `mode!=simple`. Is used for target or novel input. Default: None
        reduction (str, optional): 
            Specifies the reduction to apply to the output: `none` | `mean` | `sum`. Default: `none`
        grl (WarmStartGradientReverseLayer, optional): 
            Warmstart Gradient reversal layer. If not specified constructs a new one. Default: None.
        mask (torch.Tensor, optional):
            Mask to separate the shared and novel features and restrict source input in simple mode. Required when `mode` is `simple` or `classes`. 
            If not specified will be set to all True mask based on `domain_discriminator_s`. Default: None
        mode (Literal['simple', 'classes', 'domains'], optional):
            Mode of the domain adversarial loss. `simple` does no separation and only needs single domain discriminator.
            `classes` separates shared and novel features and requires two domain discriminators. `mask` must be provided. 
            `domains` separates source and target domain.
        """
        super().__init__()
        if reduction is None:
            reduction = "none"
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=False) if grl is None else grl
        # Disable auto_step because each forward calls grl twice
        self.grl.auto_step = False
        self.bilin_domain_discriminator_s = domain_discriminator_s
        self.bilin_domain_discriminator_o = domain_discriminator_o
        # Generally using logits is numerically more stable and enables autocasting
        # https://discuss.pytorch.org/t/bceloss-are-unsafe-to-autocast/110407
        self.binary_accuracy_func = binary_accuracy_with_logits
        self.binary_loss_func = F.binary_cross_entropy_with_logits
        self.reduction = reduction
        self.discriminator_accuracy = None
        if mask is None and (mode == 'simple' or mode == 'classes'):
            self.mask = torch.ones(domain_discriminator_s.in_feature2, dtype=torch.bool)
        else:
            self.mask = mask
        self.mode = mode.lower()
        if mode == 'simple':
            self.forward = self.forward_simple
        elif mode == 'classes':
            self.forward = self.forward_sep_classes
        else:
            self.forward = self.forward_sep_domains
    
    def forward_simple(self, f_s: torch.Tensor, 
                norm_logits_s: torch.Tensor, 
                f_t: torch.Tensor,
                norm_logits_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None,
                w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f: torch.Tensor = self.grl(torch.cat((f_s, f_t), dim=0))
        self.grl.step()
        d: torch.Tensor = self.bilin_domain_discriminator_s(f, torch.cat((norm_logits_s[:, self.mask], norm_logits_t), dim=0))
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
        
    def forward_sep_classes(self, f_s: torch.Tensor, 
                            norm_logits_s: torch.Tensor, 
                            f_t: torch.Tensor,
                            norm_logits_t: torch.Tensor,
                            w_s: Optional[torch.Tensor] = None,
                            w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f_s = self.grl(f_s)
        f_t = self.grl(f_t)
        self.grl.step()
        d_s : torch.Tensor = self.bilin_domain_discriminator_s(torch.cat((f_s, f_t), dim=0), torch.cat((norm_logits_s[:,self.mask], norm_logits_t), dim=0))
        d_n : torch.Tensor = self.bilin_domain_discriminator_o(f_s, norm_logits_s[:,~self.mask])

        d_label_s = torch.ones((d_s.size(0), 1)).to(f_s.device)
        d_label_n = torch.zeros((d_n.size(0), 1)).to(f_t.device)
        normalizer = d_s.size(0) + d_n.size(0)
        # Scale the accuracy w.r.t. the number of features in source and target domain
        # If one domain is empty accuracy will be returned as 1
        # but due to scaling this is removed anyways after multiplication 
        # with the number of features (i.e. 0)
        self.discriminator_accuracy = (self.binary_accuracy_func(d_s, d_label_s) * d_s.size(0) + 
                                       self.binary_accuracy_func(d_n, d_label_n) * d_n.size(0)
                                       ) / normalizer

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_n)
            
        # Remove NaNs if features are empty
        shared_loss = self.binary_loss_func(d_s, d_label_s, weight=w_s.view_as(d_s), 
                                            reduction=self.reduction).nan_to_num()
        
        novel_loss = self.binary_loss_func(d_n, d_label_n, weight=w_t.view_as(d_n), 
                                            reduction=self.reduction).nan_to_num()
        # Scale the loss w.r.t. the number of features
        return (shared_loss * d_s.size(0) + novel_loss * d_n.size(0)) / normalizer

    def forward_sep_domains(self, f_s: torch.Tensor, 
                            norm_logits_s: torch.Tensor, 
                            f_t: torch.Tensor,
                            norm_logits_t: torch.Tensor,
                            w_s: Optional[torch.Tensor] = None,
                            w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f_s = self.grl(f_s)
        f_t = self.grl(f_t)
        self.grl.step()
        d_s = self.bilin_domain_discriminator_s(f_s, norm_logits_s)
        d_t = self.bilin_domain_discriminator_o(f_t, norm_logits_t)

        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        num_features = f_s.size(0) + f_t.size(0)
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