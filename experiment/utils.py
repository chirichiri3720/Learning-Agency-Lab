import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

import os
import random
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)