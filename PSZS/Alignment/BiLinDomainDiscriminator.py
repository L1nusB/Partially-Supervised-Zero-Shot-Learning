from typing import List, Dict

import torch
import torch.nn as nn

class BilinearDomainDiscriminator(nn.Module):
    r"""Bilinear domain discrimintor adapted from `https://github.com/thuml/PAN` using 
    Logits for binary classification i.e. remove Sigmoid function to enable autocasting (AMP).

    Distinguish whether the input features come from the source domain or the target domain.
    Uses a bilinear layer to model the interactions between the source and target domain features.

    Args:
        in_feature1 (int): dimension of the first input feature corresponding to backbone output
        in_feature1 (int): dimension of the second input feature corresponding to the number of fine classes
        hidden_size (int): dimension of the hidden features needs to be smaller than the first feature dimension
    """

    def __init__(self, 
                 in_feature1: int, 
                 in_feature2: int, 
                 hidden_size: int, ):
        super().__init__()
        self.in_feature1 = in_feature1
        self.in_feature2 = in_feature2
        
        self.bilinear = nn.Bilinear(in_feature1, in_feature2, hidden_size)
        self.bilinear.weight.data.normal_(0, 0.01)
        self.bilinear.bias.data.fill_(0.0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.ad_layer1 = nn.Linear(in_feature1, hidden_size)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(2*hidden_size, hidden_size)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        f1 = self.bilinear(x, y)
        f1 = self.relu(f1)
        f1 = self.dropout(f1)
        f2 = self.fc1(x)
        f = torch.cat((f1, f2), dim=1)
        f = self.fc2_3(f)
        return f

    def get_parameters(self) -> List[Dict]:
        return [{"params": self.parameters(), "lr": 1.}]