# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn

class MultiSimilarityLoss(nn.Module):
    """
    Reference:
        Wang, Xun, et al. "Multi-similarity loss with general pair weighting for deep metric learning.
    
    Imported from `https://github.com/msight-tech/research-ms-loss`
    """
    @staticmethod
    def resolve_params(**kwargs):
        return {
            'scale_pos': kwargs.get('scale_pos', 2),
            'scale_neg': kwargs.get('scale_neg', 50),
            'thres': kwargs.get('thres', 1),
            'margin': kwargs.get('margin', 0.1)}
    
    def __init__(self, scale_pos: float = 2, scale_neg: float = 50, 
                 thres: float = 1, margin: float = 0.1):
        """
        Args:
            scale_pos (float, optional): Alpha parameter for positive weighting. Defaults to 2.
            scale_neg (float, optional): Beta parameter for negative weighting. Defaults to 50.
            thres (float, optional): Lambda Parameter for weighting?. Defaults to 1.
            margin (float, optional): Margin for sample mining. Defaults to 0.1.
        """
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = thres # Originally in the code this was 0.5
        self.margin = margin
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]
            
            # Can happen when all similarities are too large
            # would cause error when trying to index in the next step
            if len(pos_pair_) < 1:
                continue

            # Will cause issue as we cant guarantee at least one positive and negative pair...
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss
