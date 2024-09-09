from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardMineTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/losses/hard_mine_triplet_loss.py>`_.
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    @staticmethod
    def resolve_params(**kwargs):
        return {
            'margin': kwargs.get('margin', 0.3),
            'metric': kwargs.get('metric', 'euclidean')}
    
    def __init__(self, margin: float=0.3, metric: Literal['euclidean', 'cos']='euclidean'):
        super(HardMineTripletLoss, self).__init__()
        self.margin = margin
        self.metric = metric.lower()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.metric == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True, dtype=inputs.dtype).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt() # for numerical stability
        elif self.metric == 'cos':
            dist = inputs.unsqueeze(dim=1).expand(-1, n, -1)
            dist_1 = dist.reshape(n * n, -1)
            dist_2 = dist.transpose(0, 1).reshape(n * n, -1)
            dist = -F.cosine_similarity(dist_1, dist_2, dim=1).view(n, n)
        else:
            raise NotImplementedError()

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # No negative sample (relevant for hierarchical uses)
        if mask.all():
            return torch.tensor(0.0, requires_grad=True)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
