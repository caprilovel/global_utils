import torch 
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

