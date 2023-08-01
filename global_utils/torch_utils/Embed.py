import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from einops import rearrange, reduce, repeat



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
        # self.bias_matrix = nn.Parameter(bias_matrix, requires_grad=False)
        
        self.register_buffer('bias_matrix', bias_matrix) 
    def forward(self, attn_score):
        return attn_score - self.bias_matrix.unsqueeze(0)

class RelativePositionEmbedding(nn.Module):
    def __init__(self, Length, num_heads) -> None:
        super().__init__()
        self.Length = Length
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Length - 1), num_heads))
        
        coords_l = torch.arange(Length)
        coords = torch.stack(torch.meshgrid([coords_l], index='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 1, Wl, Wl
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wl, Wl, 2
        relative_coords[:, :, 0] += Length - \
            1  # shift to start from 0
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wl, Wl
        self.register_buffer("relative_position_index",
                             relative_position_index)
        
    def forward(self, attn):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.Length, self.Length, -1)  # Wl, Wl, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wl, Wl
        return attn + relative_position_bias.unsqueeze(0)

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
# todo: RoPE
class RoPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
