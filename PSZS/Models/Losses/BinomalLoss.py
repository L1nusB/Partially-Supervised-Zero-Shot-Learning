import torch
from torch import nn

class BinomialLoss(nn.Module):
    """
    Reference: 
        https://github.com/phucty/Deep_metric
    """
    @staticmethod
    def resolve_params(**kwargs):
        return {
            'alpha': kwargs.get('alpha', 40),
            'beta': kwargs.get('beta', 1),
            'margin': kwargs.get('margin', 0.5),
            'hard_mining': kwargs.get('hard_mining', False)}
    
    def __init__(self, alpha=40, beta=1, margin=0.5, hard_mining: bool = False):
        super(BinomialLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, inputs.t())
        # split the positive and negative pairs
        eyes_ = torch.eye(n, n, device=targets.device)
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) ^ pos_mask
        pos_mask = pos_mask ^ eyes_.eq(1)

        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        num_instances = len(pos_sim)//n + 1
        num_neg_instances = n - num_instances

        pos_sim = pos_sim.reshape(len(pos_sim)//(num_instances-1), num_instances-1)
        neg_sim = neg_sim.reshape(len(neg_sim) // num_neg_instances, num_neg_instances)

        #  clear way to compute the loss first
        loss = list()
        c = 0

        for i, pos_pair_ in enumerate(pos_sim):
            # print(i)
            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_sim[i])[0]

            if self.hard_mining:
                # print('mining')
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ < neg_pair_[-1] + 0.1)
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ > pos_pair_[0] - 0.1)  
            
                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    c += 1
                    continue        
        
            
                pos_loss = 2.0/self.beta * torch.mean(torch.log(1 + torch.exp(-self.beta*(pos_pair - 0.5))))
                neg_loss = 2.0/self.alpha * torch.mean(torch.log(1 + torch.exp(self.alpha*(neg_pair - 0.5))))

            else:  
                # print('no mining')
                pos_pair = pos_pair_
                neg_pair = neg_pair_ 

                pos_loss = torch.mean(torch.log(1 + torch.exp(-2*(pos_pair - self.margin))))
                neg_loss = torch.mean(torch.log(1 + torch.exp(self.alpha*(neg_pair - self.margin))))

            if len(neg_pair) == 0:
                c += 1
                continue

            loss.append(pos_loss + neg_loss)
            
        # loss = torch.sum(torch.cat(loss))/n
        # prec = float(c)/n
        # neg_d = torch.mean(neg_sim).data[0]
        # pos_d = torch.mean(pos_sim).data[0]

        # Each element is a single value (scalar tensor) anyways
        # so no need to concatenate or else...
        return sum(loss)/n
