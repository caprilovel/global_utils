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
        
        
def softmax_with_temperature(logits, temperature):
    """
    Perform softmax on logits divided by temperature.
    """
    logits = logits / temperature
    return torch.softmax(logits, dim=-1)
