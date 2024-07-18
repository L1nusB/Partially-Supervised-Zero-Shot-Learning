from typing import Dict
from .funcs import *
from .CustomModel import CustomModel
from .Domain_Adv_Model import Domain_Adv_Model
from .ADDA_Model import ADDA_Model
from .JAN_Model import JAN_Model
from .MCD_Model import MCD_Model
from .MDD_Model import MDD_Model
from .UJDA_Model import UJDA_Model
from .PAN_Model import PAN_Model
# Put this here instead of in models.py to avoid circular imports
METHOD_MODEL_MAP : Dict[str, CustomModel] = {
    'erm': CustomModel,
    'adda': ADDA_Model,
    'base': CustomModel,
    'dann': Domain_Adv_Model,
    'jan': JAN_Model,
    'mcc': CustomModel,
    'mcd': MCD_Model,
    'mdd': MDD_Model,
    'ujda': UJDA_Model,
    'pan': PAN_Model,
    'val': CustomModel,
}

from .models import get_model_names, build_model, load_backbone, load_checkpoint, save_checkpoint

__all__ = ['get_model_names', 'build_model', 'load_backbone', 
           'load_checkpoint', 'save_checkpoint' ,'METHOD_MODEL_MAP', 'MDD_Model',
           'CustomModel', 'Domain_Adv_Model', 'UJDA_Model',
           'JAN_Model', 'ADDA_Model', 'MCD_Model', 'PAN_Model']
