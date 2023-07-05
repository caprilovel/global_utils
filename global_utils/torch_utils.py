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
    