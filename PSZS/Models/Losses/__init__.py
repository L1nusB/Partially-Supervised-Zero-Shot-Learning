from typing import Optional
from torch.nn import Module

from .BinomalLoss import *
from .ContrastiveLoss import *
from .LiftedStructureLoss import *
from .MarginLoss import *
from .MultiSimilarityLoss import *
from .TripletLoss import *

METRIC_LOSS_MAP = {'binom': BinomialLoss,
                   'contrastive': ContrastiveLoss,
                   'lifted': LiftedStructureLoss,
                   'margin': MarginLoss,
                   'ms': MultiSimilarityLoss,
                   'triplet': HardMineTripletLoss}

AVAIL_METRIC_LOSS = list(METRIC_LOSS_MAP.keys()) + [l.__name__ for l in METRIC_LOSS_MAP.values()]

def get_losses_names():
    return ",".join(sorted(name for name in __dict__
        if not name.startswith("__") and callable(__dict__[name])
    ))
    
def resolve_metric_loss(loss: str, **kwargs) -> Optional[Module]:
    if loss in AVAIL_METRIC_LOSS:
        if loss in METRIC_LOSS_MAP:
            loss = METRIC_LOSS_MAP[loss]
        else:
            loss = __dict__[loss]
            
        return loss(**loss.resolve_params(**kwargs))
    else:
        print(f"Loss {loss} is no metric loss.")
        return None