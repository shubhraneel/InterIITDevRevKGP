import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import compute_f1

class Base_Model():
    def __init__(self):
        pass

    def __train__(self):
        raise NotImplementedError("No training method implemented")

    def __evaluate__(self):
        raise NotImplementedError("No evaluation method implemented")

    def calculate_metrics(self):
        # TODO: add logic here 