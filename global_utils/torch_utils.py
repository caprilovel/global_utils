import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat


class MyOnehot():
    ''' My one hot encoder 
    
    this class is designed to transform a numpy array of labels to a onehot matrix, in sklearn library, there is a OneHotEncoder class, but it may cause the label order changed, so I write this class to avoid this problem.
    '''
    def __init__(self, labels):
        self.labels = np.unique(labels)
        self.onehot_matrix = np.eye(len(self.labels))
        
    def transform(self, input):
        # print(self.labels.shape, input.shape)
        return (input==self.labels.reshape(-1,1)).astype(int).T
    
class DefaultArgs(argparse.ArgumentParser):
    ''' A class taht encapsulates commonly used arguments for torch training, inheriting from argparse.ArgumentParser.
    
    default arguments:
        seed: int, default 100, random seed
        batch_size: int, default 32, batch size
        lr: float, default 1e-4, learning rate
        epochs: int, default 100, epochs
    
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_argument('--seed', type=int, default=100, help="seed of the random")
        self.add_argument('--batch_size', type=int, default=32, help="batch size")
        self.add_argument('--lr', type=float, default=1e-4, help="learning rate")
        self.add_argument('--epochs', type=int, default=100, help="epochs")
        
        
def softmax_with_temperature(logits, temperature, dim=-1):
    """
    Perform softmax on logits divided by temperature.
    
    Args:
        logits: torch.Tensor, the logits of the model
        temperature: float, the temperature of the softmax
        
    Returns:
        torch.Tensor, the softmax result
        
    Examples:
        >>> logits = torch.randn(2, 3)
        >>> logits
        tensor([[ 0.5410, -0.2934, -0.8312],
                [-0.2477,  0.2062, -0.2197]])
        >>> softmax_with_temperature(logits, 1)
        tensor([[0.5003, 0.2152, 0.2845],
                [0.2656, 0.3980, 0.3364]])
    """
    logits = logits / temperature
    return torch.softmax(logits, dim=dim)


def ratio_or_exact(value, fuzzy_ratio):
    '''
    The function is used to generate the hidden dimension by taking an uncertain value. If the value is a float, it is considered as a ratio. If the value is an integer, it is considered as the value of the hidden dimension.
    
    Args:
        value: int, the input dim 
        fuzzy_ratio: float or int, the ratio of the hidden dimension, or the exact value of the hidden dimension.
    '''
    return int(value * fuzzy_ratio) if isinstance(fuzzy_ratio, float) else fuzzy_ratio

