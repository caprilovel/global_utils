import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat

class Residual(nn.Module):
    def __init__(self, fn, res_mode='add') -> None:
        super().__init__()
        self.fn = fn
        self.mode = res_mode
        
    def forward(self, x, *args, **kwargs):
        if self.mode == "add":
            return self.fn(x, *args, **kwargs) + x
        elif self.mode == "cat":
            return torch.cat([x, self.fn(x, *args, **kwargs)], dim=0)
        
 

            