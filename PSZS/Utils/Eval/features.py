from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PSZS.Models import CustomModel

@torch.no_grad()
def get_features(*loader: DataLoader, 
                 model: nn.Module,
                 device: torch.device,
                 split_labels: Optional[Sequence[int]] = None,
                 max_feat: Optional[int] = None) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    if split_labels is not None:
        split_labels = torch.tensor(split_labels, device=device)
    model.eval()
    if max_feat is None:
        max_feat = float('inf')
    
    features_split1 = []
    features_split2 = []
    finish = False
    for l in loader:
        loader_features_split1 = []
        loader_features_split2 = []
        for i, (data, labels) in enumerate(l):
            # If max_feat is reached, break the loop
            mask = torch.isin(labels, split_labels) if split_labels is not None else torch.ones_like(labels, dtype=torch.bool)
            if any(mask) == False:
                loader_features_split1.append(model(data).cpu())
            elif all(mask):
                loader_features_split2.append(model(data).cpu())
            else:
                loader_features_split1.append(model(data[~mask]).cpu())
                loader_features_split2.append(model(data[mask]).cpu())
            if (len(loader_features_split1) >= max_feat and split_labels is None) or (
                len(loader_features_split1) >= max_feat and len(loader_features_split2) >= max_feat):
                finish = True
                loader_features_split1 = loader_features_split1[:max_feat]
                loader_features_split2 = loader_features_split2[:max_feat]
                break
        features_split1.extend(loader_features_split1)
        features_split2.extend(loader_features_split2)
        if finish:
            break
        
    if split_labels is None:
        return torch.cat(features_split1, dim=0)
    else:
        return torch.cat(features_split1, dim=0), torch.cat(features_split2, dim=0)
    
def get_feat_norm(*features: torch.Tensor, order: int = 2) -> torch.Tensor | Sequence[torch.Tensor]:
    r = [torch.norm(feat, p=order, dim=1) for feat in features]
    return r if len(r) > 1 else r[0]  # Return a single tensor if only one feature tensor is provided

@torch.no_grad()
def get_correct_pred_feat(*loader: DataLoader,
                          model: nn.Module | CustomModel,
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    correct_features = []
    incorrect_features = []
    model.eval()
    assert hasattr(model, 'backbone'), 'Model must have a backbone attribute'
    assert hasattr(model, 'classifier'), 'Model must have a classifier attribute'
    assert hasattr(model.classifier, 'forward_test'), 'Model classifier must have a forward_test method'
    if hasattr(model, 'bottleneck') is False:
        setattr(model, 'bottleneck', nn.Identity())
    for l in loader:
        for data, labels in l:
            feat = model.backbone(data)
            pred: torch.Tensor = model.classifier.forward_test(model.bottleneck(feat))
            for i, p in enumerate(pred):
                if p.argmax() == labels[i]:
                    correct_features.append(feat[i])
                else:
                    incorrect_features.append(feat[i])
    return torch.stack(correct_features), torch.stack(incorrect_features)