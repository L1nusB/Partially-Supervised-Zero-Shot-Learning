"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional, Sequence
import torch
import torch.nn as nn

from PSZS.Alignment.gradientReversalLayer import GradientReverseLayer
    
class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    r"""The Joint Multiple Kernel Maximum Mean Discrepancy (JMMD) used in
    `Deep Transfer Learning with Joint Adaptation Networks (ICML 2017) <https://arxiv.org/abs/1605.06636>`_

    Given source domain :math:`\mathcal{D}_s` of :math:`n_s` labeled points and target domain :math:`\mathcal{D}_t`
    of :math:`n_t` unlabeled points drawn i.i.d. from P and Q respectively, the deep networks will generate
    activations in layers :math:`\mathcal{L}` as :math:`\{(z_i^{s1}, ..., z_i^{s|\mathcal{L}|})\}_{i=1}^{n_s}` and
    :math:`\{(z_i^{t1}, ..., z_i^{t|\mathcal{L}|})\}_{i=1}^{n_t}`. The empirical estimate of
    :math:`\hat{D}_{\mathcal{L}}(P, Q)` is computed as the squared distance between the empirical kernel mean
    embeddings as

    .. math::
        \hat{D}_{\mathcal{L}}(P, Q) &=
        \dfrac{1}{n_s^2} \sum_{i=1}^{n_s}\sum_{j=1}^{n_s} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{sl}) \\
        &+ \dfrac{1}{n_t^2} \sum_{i=1}^{n_t}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{tl}, z_j^{tl}) \\
        &- \dfrac{2}{n_s n_t} \sum_{i=1}^{n_s}\sum_{j=1}^{n_t} \prod_{l\in\mathcal{L}} k^l(z_i^{sl}, z_j^{tl}). \\

    Args:
        kernels (tuple(tuple(torch.nn.Module))): kernel functions, where `kernels[r]` corresponds to kernel :math:`k^{\mathcal{L}[r]}`.
        linear (bool): whether use the linear version of JAN. Default: False
        thetas (list(Theta): use adversarial version JAN if not None. Default: None

    Inputs:
        - z_s (tuple(tensor)): multiple layers' activations from the source domain, :math:`z^s`
        - z_t (tuple(tensor)): multiple layers' activations from the target domain, :math:`z^t`

    Shape:
        - :math:`z^{sl}` and :math:`z^{tl}`: :math:`(minibatch, *)`  where * means any dimension
        - Outputs: scalar

    .. note::
        Activations :math:`z^{sl}` and :math:`z^{tl}` must have the same shape.

    .. note::
        The kernel values will add up when there are multiple kernels for a certain layer.

    Examples::

        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> layer1_kernels = (GaussianKernel(alpha=0.5), GaussianKernel(1.), GaussianKernel(2.))
        >>> layer2_kernels = (GaussianKernel(1.), )
        >>> loss = JointMultipleKernelMaximumMeanDiscrepancy((layer1_kernels, layer2_kernels))
        >>> # layer1 features from source domain and target domain
        >>> z1_s, z1_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # layer2 features from source domain and target domain
        >>> z2_s, z2_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> output = loss((z1_s, z2_s), (z1_t, z2_t))
    """

    def __init__(self, 
                 kernels: Sequence[Sequence[nn.Module]], 
                 linear: Optional[bool] = True, 
                 thetas: Sequence[nn.Module] = None):
        super().__init__()
        self.kernels = kernels
        self.weight_matrix = None
        self.linear = linear
        # Indicates whether to recompute the weight matrix
        # when the batch sizes change (-1 forces recomputation on first pass)
        self.last_source_dim = self.last_target_dim = -1
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, 
                feature_source: torch.Tensor,
                feature_target: torch.Tensor,
                pred_norm_source: torch.Tensor, 
                pred_norm_target: torch.Tensor) -> torch.Tensor:
        # Recompute the weight matrix if the batch sizes changed
        if self.last_source_dim != feature_source.size(0) or self.last_target_dim != feature_target.size(0):
            # If one of the batch sizes is 0, the loss is 0 and no computations are needed
            if feature_source.size(0) == 0 or feature_target.size(0) == 0:
                return torch.tensor(0., device=feature_source.device)
            # Update the weight matrix and last dimensions
            self.last_source_dim = feature_source.size(0)
            self.last_target_dim = feature_target.size(0)
            self.weight_matrix = self.kernel_weight_matrix(device=feature_source.device)
            
        z_s=(feature_source, pred_norm_source)
        z_t=(feature_target, pred_norm_target)
        kernel_matrix = torch.ones_like(self.weight_matrix, device=self.weight_matrix.device)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum(
                [kernel(layer_features) for kernel in layer_kernels])  # Add up the matrix of each kernel

        # Add 1 / (bs_source -1) + 1 / (bs_source -1) to make up for the 
        # value on the diagonal to ensure loss is positive in the non-linear version
        # last_source_dim and last_target_dim are the batch sizes
        loss = (kernel_matrix * self.weight_matrix).sum() + 1. / float(self.last_source_dim - 1) + 1. / float(self.last_target_dim - 1)
        return loss
    
    
    def kernel_weight_matrix(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Constructs a kernel weight matrix to scale the kernel matrix in the JMMD loss.
        If a weight matrix is given and has the correct shape, it is returned without recomputation.
        If `linear` is True, the weight matrix is constructed as a linear kernel matrix based on 
        tllib.alignment.dan ~_update_index_matrix. This requires the batch sizes to match.
        Otherwise the weight matrix is constructed as a non-linear kernel weight matrix 
        where same domain values are positive, cross domain values are negative and diagonal is 0.
        The non-zero values are scaled by the number of samples in the respective domain.

        Args:
            device (Optional[torch.device], optional): device the weight matrix should be put on. Defaults to None.

        Returns:
            torch.Tensor: Weight matrix.
        """
        weight_matrix = torch.zeros(self.last_source_dim+self.last_target_dim, 
                                    self.last_source_dim+self.last_target_dim, 
                                    device=device)
        if self.linear:
            # Implementation of linear kernel weights taken from tllib
            assert self.last_source_dim == self.last_target_dim, f"Linear kernel requires equal batch sizes {self.last_source_dim}!={self.last_target_dim}"
            for i in range(self.last_source_dim):
                    s1, s2 = i, (i + 1) % self.last_source_dim
                    t1, t2 = s1 + self.last_source_dim, s2 + self.last_source_dim
                    weight_matrix[s1, s2] = 1. / float(self.last_source_dim)
                    weight_matrix[t1, t2] = 1. / float(self.last_source_dim)
                    weight_matrix[s1, t2] = -1. / float(self.last_source_dim)
                    weight_matrix[s2, t1] = -1. / float(self.last_source_dim)
            return weight_matrix
        
        # Set same domain values to positive, cross domain values to negative and diagonal to 0
        weight_matrix[:self.last_source_dim, :self.last_source_dim] = 1. / float(self.last_source_dim * (self.last_source_dim - 1))
        weight_matrix[self.last_source_dim:, self.last_source_dim:] = 1. / float(self.last_target_dim * (self.last_target_dim - 1))
        weight_matrix[:self.last_source_dim, self.last_source_dim:] = -1. / float(self.last_source_dim * self.last_target_dim)
        weight_matrix[self.last_source_dim:, :self.last_source_dim] = -1. / float(self.last_source_dim * self.last_target_dim)
        weight_matrix.diagonal().zero_()
        return weight_matrix


class Theta(nn.Module):
    r"""
    maximize loss respect to `theta`
    minimize loss respect to features
    """
    def __init__(self, dim: int):
        super(Theta, self).__init__()
        self.grl1 = GradientReverseLayer()
        self.grl2 = GradientReverseLayer()
        self.layer1 = nn.Linear(dim, dim)
        nn.init.eye_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.grl1(features)
        return self.grl2(self.layer1(features))