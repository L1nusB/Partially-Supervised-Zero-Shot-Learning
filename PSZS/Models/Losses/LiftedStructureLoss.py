import torch
from torch import nn
import torch.nn.functional as F

class LiftedStructureLoss(nn.Module):
    """
    Reference: 
        https://github.com/phucty/Deep_metric
    """
    @staticmethod
    def resolve_params(**kwargs):
        return {
            'margin': kwargs.get('margin', 1),
            'hard_mining': kwargs.get('hard_mining', False)}
    
    def __init__(self, margin: float = 1, hard_mining: bool = False):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.hard_mining = hard_mining

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = inputs.size(0)

        mag = (inputs ** 2).sum(1).expand(n, n)
        sim_mat = inputs.mm(inputs.transpose(0, 1))
    
        dist_mat = (mag + mag.transpose(0, 1) - 2 * sim_mat)
        dist_mat = F.relu(dist_mat).sqrt().to(inputs.device)
        # split the positive and negative pairs
        eyes_ = torch.tensor(torch.eye(n, n), device=targets.device)
        zeros_ = torch.zeros(n,n, device=targets.device)
        # eyes_ = Variable(torch.eye(n, n))
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) ^ pos_mask
        pos_mask = pos_mask ^ eyes_.eq(1)

        pos_dist = torch.where(pos_mask, dist_mat, zeros_)
        neg_dist = torch.where(neg_mask, dist_mat, zeros_)

        loss = 0

        len_p = pos_mask.nonzero().shape[0]
        if self.hard_mining:
            for ind in torch.triu(pos_mask).nonzero():
                i = ind[0].item()
                j = ind[1].item()

                hardest_k, _ = torch.topk(neg_dist[i][neg_dist[i].nonzero()].squeeze(), 1, largest=False)
                neg_ik_component = torch.sum(torch.exp(self.margin - hardest_k))
                hardest_l, _ = torch.topk(neg_dist[j][neg_dist[j].nonzero()].squeeze(), 1, largest=False)
                neg_jl_component = torch.sum(torch.exp(self.margin - hardest_l))
                Jij_heat = torch.log(neg_ik_component+neg_jl_component) + pos_dist[i,j]
                loss += torch.nn.functional.relu(Jij_heat) ** 2
            return loss / len_p
        
        else:
            for ind in torch.triu(pos_mask).nonzero():
                i = ind[0].item()
                j = ind[1].item()
                neg_ik_component = torch.sum(torch.exp(self.margin - neg_dist[i][neg_dist[i].nonzero()].squeeze()))
                neg_jl_component = torch.sum(torch.exp(self.margin - neg_dist[j][neg_dist[j].nonzero()].squeeze()))
                Jij_heat = torch.log(neg_ik_component+neg_jl_component) + pos_dist[i,j]

                loss += torch.nn.functional.relu(Jij_heat) ** 2

            return loss / len_p