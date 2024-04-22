import random
import numpy as np
import torch

def global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)