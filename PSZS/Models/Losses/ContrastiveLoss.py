import torch
from torch import nn

def test():
    """Random"""
    pass

class ContrastiveLoss(nn.Module):
    """
    Reference: 
        https://github.com/phucty/Deep_metric
    """
    @staticmethod
    def resolve_params(**kwargs):
        return {
            'margin': kwargs.get('margin', 0.5),
            'mean': kwargs.get('mean', False)}
        
    def __init__(self, margin=0.5, mean: bool = False, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.mean = mean

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, inputs.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n, n, device=targets.device)
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) ^ pos_mask
        # neg_mask = eyes_.eq(eyes_) - pos_mask
        pos_mask = pos_mask ^ eyes_.eq(1)
        # pos_mask = pos_mask - eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        num_instances = len(pos_sim)//n + 1
        num_neg_instances = n - num_instances
        
        pos_sim = pos_sim.reshape(len(pos_sim)//(num_instances-1), num_instances-1)
        neg_sim = neg_sim.reshape(len(neg_sim) // num_neg_instances, num_neg_instances)

        loss = list()
        for i, pos_pair_ in enumerate(pos_sim):
            # print(i)
            pos_pair = pos_pair_
            neg_pair_ = torch.sort(neg_sim[i])[0]

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)
            # pos_pair = torch.masked_select(pos_pair_, pos_pair_ < neg_pair_[-1] + 0.05)
            
            neg_loss = 0
            if self.mean == False:
                # print('sum is ok')
                pos_loss = torch.sum(-pos_pair+1) 
                if len(neg_pair) > 0:
                    neg_loss = torch.sum(neg_pair)
                loss.append(pos_loss + neg_loss)

            else:
                pos_loss = torch.mean(-pos_pair+1) 
                if len(neg_pair) > 0:
                    neg_loss = torch.mean(neg_pair)
                loss.append(pos_loss + neg_loss)

        # loss = torch.sum(torch.cat(loss))/n
        # neg_d = torch.mean(neg_sim).data[0]
        # pos_d = torch.mean(pos_sim).data[0]
        # Each element is a single value (scalar tensor) anyways
        # so no need to concatenate or else...
        return sum(loss)/n