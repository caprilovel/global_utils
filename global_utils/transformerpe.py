import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat


#################    ALiBi    ################
class AttnwithLinearBiases(nn.Module):
    """AttnwithLinearBiases adds linear biases to attention scores.

    Attnention scores are calculated as the dot product of the query and key
    
    Examples:
        >>> attn = AttnwithLinearBiases(8, 10)
        >>> attn_score = torch.randn(2, 8, 10, 10)
        >>> attn_score = attn(attn_score)
        
    """
    def __init__(self, heads, length) -> None:
        super().__init__()
        range_matrix = torch.arange(length)
        bias_matrix = repeat(range_matrix, 'i -> j i', j=length) - repeat(range_matrix, 'i -> i j', j=length)
        bias_matrix = torch.abs(bias_matrix)
        m = calculate_m(heads)
        m = m.unsqueeze(-1).unsqueeze(-1)
        bias_matrix = m * repeat(bias_matrix, 'i j -> h i j', h=heads)
        self.bias_matrix = nn.Parameter(bias_matrix, requires_grad=False)
        
    def forward(self, attn_score):
        
        return attn_score - self.bias_matrix.unsqueeze(0)
        


def calculate_m(heads):
    """calculate_m calculates the slopes of the linear bias, the size of the slopes is same as the number of heads, and the slopes are an isometric series associated with the number of heads.

    Args:
        heads (int): the number of heads
        
    Examples:
        >>> m = calculate_m(8)
        >>> print(m.shape)
        torch.Size([8])
        
    """
    
    m = [ 2 ** -(8*(i+1)/heads) for i in range(heads)]
    return torch.tensor(m)

#################    SinusoidalPE    ################

class SinusoidalPE(nn.Module):
    '''sinusoidal positional embedding
    
    '''
    def __init__(self, num_hiddens, dropout=0., max_len=1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

#################    RoPE    ################

class RoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
