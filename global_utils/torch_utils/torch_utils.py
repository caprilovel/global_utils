import argparse
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat
import os
import random


def random_seed(seed=2020):
    """to set the random seed for torch, numpy and random library

    Args:
        seed (int, optional): random seed. Defaults to 2020.
    """
    # determine the random seed
    random.seed(seed)
    # hash, save the random seed                   
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MyOnehot():
    ''' My one hot encoder 
    
    this class is designed to transform a numpy array of labels to a onehot matrix, in sklearn library, there is a OneHotEncoder class, but it may cause the label order changed, so I write this class to avoid this problem.
    '''
    def __init__(self, labels):
        """Myonehot init function

        To form the onehot matrix, we need to know the labels, so we need to input the labels to the init function.
        
        Args:
            labels (np.array): labels of the dataset
        """
        self.labels = np.unique(labels)
        self.onehot_matrix = np.eye(len(self.labels))
        
    def __call__(self, input) -> Any:
        return self.transform(input)
        
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
        """_summary_
        """
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


def alignment_add(tensor1, tensor2, alignment_opt='trunc'):
    '''add with auto-alignment
    
    Using for the transpose convolution. Transpose convolution will cause the size of the output uncertain. However, in the unet structure, the size of the output should be the same as the input. So, we need to align the size of the output with the input.
    
    Args:
        tensor1: the first tensor
        tensor2: the second tensor, only the last dim is not same as the first tensor
        alignment_opt: the alignment option, can be 'trunc' or 'padding'
    
    Examples:
        >>> tensor1 = torch,randn(1, 2, 3)
        >>> tensor2 = torch.randn(1, 2, 4)
        >>> tensor3 = alignment_add(tensor1, tensor2)
        >>> tensor3.shape 
        torch.Size([1, 2, 3])
    
    '''
    
    assert tensor1.shape[0:-1] == tensor2.shape[0:-1], 'the shape of the first tensor should be the same as the second tensor'
    short_tensor = tensor1 if tensor1.shape[-1] < tensor2.shape[-1] else tensor2
    long_tensor = tensor1 if tensor1.shape[-1] >= tensor2.shape[-1] else tensor2
    if alignment_opt == 'trunc':
        return short_tensor + long_tensor[..., :short_tensor.shape[-1]]
    elif alignment_opt == 'padding':
        return long_tensor + F.pad(short_tensor, (0, long_tensor.shape[-1] - short_tensor.shape[-1]))