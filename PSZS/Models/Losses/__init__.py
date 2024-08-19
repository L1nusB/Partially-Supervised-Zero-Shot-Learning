from .BinomalLoss import *
from .ContrastiveLoss import *
from .LiftedStructureLoss import *
from .MarginLoss import *
from .MultiSimilarityLoss import *
from .TripletLoss import *

def get_losses_names():
    return ",".join(sorted(name for name in __dict__
        if not name.startswith("__") and callable(__dict__[name])
    ))