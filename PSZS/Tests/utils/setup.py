from argparse import Namespace
from typing import Optional
import warnings
import random

import torch
import torch.backends.cudnn as cudnn
import numpy as np

def setup_device(args: Namespace, device_key: str = "device", device: Optional[str]=None) -> torch.device:
    if device is None:
       device = getattr(args, device_key, "cpu")
    
    if torch.cuda.is_available() == False:
        warnings.warn("CUDA is not available, using CPU.")
        device = torch.device('cpu')
    elif device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def setup_seed(args: Namespace, seed_key: str = "seed", seed: Optional[int] = None) -> None:
    if seed is None:
        seed = getattr(args, seed_key, None)
    if seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False