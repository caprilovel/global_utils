import os
import wfdb
import numpy as np
from torch.utils.data import Dataset

ecg_names = ['100', '101', '102', '103', '104', '105', '106', '107', '108',
 '109', '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
 '122', '123', '124', '200', '201','202', '203', '205', '207', '208', '209',
 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223',
 '228', '230', '231', '232', '233', '234']

noise_names = ["em", "ma", "bw"]

exact_symbol = [
    "N", "L", "R", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "f", "x", "Q", "!", 
]

class mitbih_arryth_read():
    def __init__(self, num, path=None) -> None:
        if not path:
            if os.path.exists("../data/mit-bih-arrhythmia-database-1.0.0/"):
                path = "../data/mit-bih-arrhythmia-database-1.0.0/"
            elif os.path.exists("./data/mit-bih-arrhythmia-database-1.0.0/"):
                path = "./data/mit-bih-arrhythmia-database-1.0.0/"
        self.data = (wfdb.rdrecord(os.path.join(path, str(num)), physical=False)).d_signal
        self.data = np.array(self.data)
        self.ann = wfdb.rdann(os.path.join(path, str(num)), extension="atr").__dict__
        
    def ann(self,):
        return self.ann
    
    def data(self,):
        return self.data
    
class mitbih_noise_read():
    """mit-bih noise read
    """
    def __init__(self, noise_name, path=None) -> None:
        """_summary_

        Args:
            noise_name (_type_): _description_
            path (_type_, optional): _description_. Defaults to None.
        """
        if not path:
            if os.path.exists("../data/mit-bih-noise-stress-test-database-1.0.0/"):
                path = "../data/mit-bih-noise-stress-test-database-1.0.0/"
            elif os.path.exists("./data/mit-bih-noise-stress-test-database-1.0.0/"):
                path = "./data/mit-bih-noise-stress-test-database-1.0.0/"
        self.data = (wfdb.rdrecord(os.path.join(path, str(noise_name)), physical=False)).d_signal
        self.data = np.array(self.data)

    
    def data(self, ):
        return self.data
        

class Ecg_dataset(Dataset):
    '''
    ECG dataset for .npy files
    '''
    def __init__(self, class_names, dict_data_path) -> None:
        super().__init__()
        datas = []
        labels = []
        nums = []
        for cls_name in class_names:
            data = np.load(os.path.join(dict_data_path, cls_name + '.npy'))
            label = np.repeat(cls_name, data.shape[0])
            datas.append(data)
            labels.append(label)
            nums.append(data.shape[0])
        self.datas = np.concatenate(datas, axis=0)
        self.labels = np.concatenate(labels, axis=0)
        self.nums = np.array(nums)
        
    def __len__(self):
        return self.datas.shape[0]
    
    def __getitem__(self, index):
        return self.datas[index], self.labels[index]
    
    def split(self, split_factor=0.5, random_position=False):
        """split data set into train and test set 

        Args:
            split_factor (float): the factor of train set, default 0.5
            random_position (bool, optional): whether use random data, if true, then will choose the data in random position each class; else, will choose the front data. Defaults to False.

        Returns:
            tarin_data_slice, test_data_slice: the slice of train and test data
            
        Using Example:
            >>> from torch.utils.data import DataLoader
            >>> from torch.utils.data.sampler import SubsetRandomSampler
            >>> dataset = Ecg_dataset(['N', 'A', 'V', 'F', 'Q'])
            >>> train_data_slice, test_data_slice = dataset.split(split_factor=0.5, random_position=False)
            >>> train_sampler = SubsetRandomSampler(train_data_slice)
            >>> test_sampler = SubsetRandomSampler(test_data_slice)
            >>> train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
            >>> test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)
        """
        temp_num = 0
        train_nums = np.floor(self.nums * split_factor).astype(np.int64)
        train_index = np.full(self.__len__(), False, dtype=bool)
        for num, train_num in zip(self.nums, train_nums):
            if random_position:
                temp_whole_index = np.arange(num)
                temp_whole_index += temp_num
                temp_train_index = np.random.choice(temp_whole_index, train_num, replace=False)
                train_index[temp_train_index] = True
            else:
                train_index[temp_num:temp_num+train_num] = True
            temp_num += num
        test_index = np.logical_not(train_index)
        
        indice = np.arange(self.__len__())            
        
        return indice[train_index], indice[test_index]
    