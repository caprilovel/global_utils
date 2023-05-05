from __future__ import division
import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd
import random
# from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import torch 
from sklearn.metrics import accuracy_score, f1_score
import sys
import time 
#-----------------------------------------------------------#

#   label_select,用于样本不均衡的数据集，每一次的输入样本为每一类相同数目的样本

#-----------------------------------------------------------#


def label_select(labels, sampling):
    '''
    this function is used for selecting the same amount train samples in each class

    input:
    labels: List, the label list
    samling: the size of the samples in each class
    
    output:
     the list of initialize train dataset index
    
    '''
    if type(labels) is list:
        labels = np.array(labels)
    classes = np.unique(labels)
    # n_class = len(classes)
    class_dict = {}
    sample_list = []   
    for i in classes:
        class_dict[i] = [j for j,x in enumerate(labels) if x==i]
        np.random.shuffle(class_dict[i])
        sample_list.append(class_dict[i][0:sampling])
    return np.concatenate(np.array(sample_list))

def predict (output):
    '''
    输入为batch * 预测向量,返回batch * 预测结果
    '''
    return output.max(1)[1].cpu().numpy()

def save_confusion_matrix(cm, path, title=None, labels_name=None,   cmap=plt.cm.Blues):

    
    plt.rc('font',family='Times New Roman',size='8')   # 设置字体样式、大小
    # plt.colorbar()
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) == 0:
                cm[i, j]=0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels_name, yticklabels=labels_name,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
                        ha="center", va="center",
                        color="white"  if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(path + 'cm.jpg', dpi=300)


#-----------------------------------------------------------#

#   random_seed，用于torch训练随机种子

#-----------------------------------------------------------#


def random_seed(seed=2020):
    # determine the random seed
    random.seed(seed)
    # hash 
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def accuracy(output, labels):
    pred = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    return accuracy_score(labels, pred)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

#-----------------------------------------------------------#

#   将时间（秒）转换为 时间（天，小时，分钟，秒）

#   用法示例：day,hour,minute,second = second2time(30804.7)

#   print("time: {:02d}d{:02d}h{:02d}m{:.2f}s".format(day, hour, minute, second))

#   用于显示训练时间

#-----------------------------------------------------------#

def second2time(second):
    intsecond = int(second)
    day = int(second) // (24 * 60 * 60)
    intsecond -= day * (24 * 60 * 60)
    hour = intsecond // (60 * 60)
    intsecond -= hour * (60 * 60)
    minute = intsecond // 60
    intsecond -= 60 * minute
    return (day, hour, minute, second - int(second) + intsecond)


#-----------------------------------------------------------#

#   使用方法：在开始加入此行代码sys.stdout = Logger('log.log')

#-----------------------------------------------------------#


class Logger(object):
    def __init__(self, logFile='./Default.log'):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def loaddata(filename):
    df = pd.read_csv(filename, header=None, delimiter=",")
    a = np.array(df.as_matrix())
    return a


def load_raw_ts(dataset, tensor_format=True, path='data/'):
    path = "{}raw/{}/".format(path, dataset)
    # path = path + "Multivariate_ts" + "/"
    # if it is npy file 
    if os.path.exists(path+'X_train.npy'):
        x_train = np.load(path + 'X_train.npy')
        y_train = np.load(path + 'y_train.npy')
        x_test = np.load(path + 'X_test.npy')
        y_test = np.load(path + 'y_test.npy')
        ts = np.concatenate((x_train, x_test), axis=0)
        ts = np.transpose(ts, axes=(0, 2, 1))
        labels = np.concatenate((y_train, y_test), axis=0)
        nclass = int(np.amax(labels)) + 1


        train_size = y_train.shape[0]

        total_size = labels.shape[0]
        idx_train = range(train_size)
        idx_val = range(train_size, total_size)
        idx_test = range(train_size, total_size)
    # if it is ts file 
    elif os.path.exists(path + dataset + "_TEST.ts"):
        test_X, test_y = load_from_tsfile_to_dataframe(path+dataset+'_TEST.ts')
        train_X, train_y = load_from_tsfile_to_dataframe(path+dataset+'_TRAIN.ts')
        label_encoder = LabelEncoder()
        labels = np.concatenate((np.array(label_encoder.fit_transform(train_y)),np.array(label_encoder.fit_transform(test_y))),axis=0)
        nclass = int(np.amax(labels)) + 1 
        train_nd = series_to_nd(train_X.values)
        test_nd = series_to_nd(test_X.values)
        ts = np.concatenate((train_nd,test_nd), axis=0)
        np.transpose(ts, (0, 2, 1))
        train_size = train_y.shape[0]
        total_size = labels.shape[0]
        idx_train = range(train_size)
        idx_val = range(train_size, total_size)
        idx_test = range(train_size, total_size)
    if tensor_format:
        # features = torch.FloatTensor(np.array(features))
        ts = torch.FloatTensor(np.array(ts))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_val, idx_test, nclass


def normalize(mx):
    """Row-normalize sparse matrix"""
    row_sums = mx.sum(axis=1)
    mx = mx.astype('float32')
    row_sums_inverse = 1 / row_sums
    f = mx.multiply(row_sums_inverse)
    return sp.csr_matrix(f).astype('float32')


def accuracy(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score

def f1score(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    f1score = (sklearn.metrics.f1_score(labels, preds))

    return f1score

def sconfusion_matrix(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    f1score = (sklearn.metrics.confusion_matrix(labels, preds))

    return f1score
def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding=0):

    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output

def dump_embedding(proto_embed, sample_embed, labels, dump_file='./plot/embeddings.txt'):
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate((np.asarray([i for i in range(nclass)]),
                             labels.squeeze().cpu().detach().numpy()), axis=0)

    with open(dump_file, 'w') as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = label + "," + ",".join(["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')

def series_to_nd(m):
    series_len = len(m[0][0])
    train_nd = np.empty((len(m), len(m[0]), series_len))
    for i in range(len(m)):
        for j in range(len(m[0])):
            train_nd[i][j] = np.array(m[i][j])
    return train_nd


def data_iter(dataset_name, sample_size, fold=5):
    ts, labels, idx_train, idx_val, idx_test, nclass = load_raw_ts(dataset=dataset_name, tensor_format=False)
    classes = np.unique(labels)
    class_num = []
    sample_list = []
    class_dict = {}
    for i in classes:
        class_dict[i] = [j for j,x in enumerate(labels) if x==i]
        np.random.shuffle(class_dict[i])
        class_num.append(int(len(class_dict[i]) * (fold -1)/fold)-1)

    yield



def array_fulfill(init_array, length):
    '''fulfill an array
 
    '''
    init_array = np.array(init_array)
    if len(init_array) < length:
        ful_array = np.tile(init_array, length//len(init_array))
        tmp = init_array[0:length%(len(init_array))]
        ful_array = np.concatenate((ful_array, tmp), axis=0)
    else:
        ful_array = init_array
    return ful_array

    
#-----------------------------------------------------------#

#   Data iterator for UAE archive

#-----------------------------------------------------------#
# todo 增加噪声参数

class Data_iter():
    '''data iterator for the UAE Archive

    
    '''
    def __init__(self, dataset, division) -> None:
        self.ts, self.labels, _,_,_,_ = load_raw_ts(dataset, tensor_format=False)
        assert len(division)==3
        self.division = np.array(division)/np.sum(division)
        self.classes = np.unique(self.labels)
        self.class_dict = {}
        self.max_length = np.max(np.bincount(self.labels))
        self.min_length = np.min(np.bincount(self.labels))
        for i in self.classes:
            self.class_dict[i] = [j for j,x in enumerate(self.labels) if x==i]
        if self.min_length * division[1] < 1:
            raise
    

        
    def train_iter(self, batch_sample_size, use_last_data=True, tensor_format=True, use_noise=False):
        r'''Training Data Iterator

        This function was give a training data iterator which is divided by the division. The same size of every class train data are given in each batch.

        Args:
          batch_sample_size: interval, the amount of the single class data in a batch
          use_last_data: default True, whether use the last data which is not same size of the batch_sample_size
          tensor_format: default True, whether return the torch.Tensor or numpy.ndarray
          use_noise: default False, whether use the noise added to the time series 

        '''
        tmp_class_idx = {}
        tmp_maxlength = int(self.max_length*self.division[0])
        for i in self.classes:
            tmp_class_idx[i] = array_fulfill(self.class_dict[i][0:int(self.division[0]*len(self.class_dict[i]))], tmp_maxlength)    # 填充每一类的数量为最大值
        
        for i in range(tmp_maxlength//batch_sample_size):
            tmp_batch_idx = []
            for j in self.classes:
                tmp_batch_idx.append(tmp_class_idx[j][batch_sample_size*i: batch_sample_size*(i+1)])
            tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)

            ts = self.ts[tmp_batch_idx] 
            if use_noise:
                ts += np.random.normal(0, 0.01, [ts.shape[1], ts.shape[2]])
            
            if tensor_format:
                yield torch.FloatTensor(ts), torch.LongTensor(self.labels[tmp_batch_idx])
            else:
                yield ts, self.labels[tmp_batch_idx]  


        if use_last_data and (tmp_maxlength%batch_sample_size is not 0):
            tmp_batch_idx = []
            for j in self.classes:
                tmp_batch_idx.append(tmp_class_idx[j][-(tmp_maxlength%batch_sample_size)+1:])
            tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
            ts = self.ts[tmp_batch_idx]
            if use_noise:
                ts += np.random.normal(0, 0.01, [ts.shape[1], ts.shape[2]])
            if tensor_format:
                yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
            else:
                yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx]  

    def eval_iter(self, batch_sample_size, tensor_format=True): 
        tmp_class_idx = {}
        tmp_maxlength = int(self.max_length*self.division[1])
        for i in self.classes:
            tmp_class_idx[i] = array_fulfill(self.class_dict[i][int(self.division[0]*len(self.class_dict[i])):int((self.division[1]+self.division[0])*len(self.class_dict[i]))], tmp_maxlength)    # 填充每一类的数量为最大值
        
        for i in range(tmp_maxlength//batch_sample_size):
            tmp_batch_idx = []
            for j in self.classes:
                tmp_batch_idx.append(tmp_class_idx[j][batch_sample_size*i: batch_sample_size*(i+1)])
            tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
            if tensor_format:
                yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
            else:
                yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx]  


        tmp_batch_idx = []
        for j in self.classes:
            tmp_batch_idx.append(tmp_class_idx[j][-(self.max_length%batch_sample_size)+1:])
        tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
        if tensor_format:
            yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
        else:
            yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx] 

    def test_iter(self, batch_sample_size, tensor_format=True):
        tmp_class_idx = {}
        tmp_maxlength = int(self.max_length*self.division[2])
        for i in self.classes:
            tmp_class_idx[i] = array_fulfill(self.class_dict[i][int(self.division[1]*len(self.class_dict[i])):], tmp_maxlength)    # 填充每一类的数量为最大值
        
        for i in range(tmp_maxlength//batch_sample_size):
            tmp_batch_idx = []
            for j in self.classes:
                tmp_batch_idx.append(tmp_class_idx[j][batch_sample_size*i: batch_sample_size*(i+1)])
            tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
            if tensor_format:
                yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
            else:
                yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx]  


        tmp_batch_idx = []
        for j in self.classes:
            tmp_batch_idx.append(tmp_class_idx[j][-(self.max_length%batch_sample_size)+1:])
        tmp_batch_idx = np.concatenate(tmp_batch_idx, axis=0)
        if tensor_format:
            yield torch.FloatTensor(self.ts[tmp_batch_idx]), torch.LongTensor(self.labels[tmp_batch_idx])
        else:
            yield self.ts[tmp_batch_idx], self.labels[tmp_batch_idx] 


    def shuffle(self, ) -> None:
        for i in self.classes:
            np.random.shufffle(self.class_dict[i])
    
    def data_shape(self):
        return self.ts.shape

    def label_num(self):
        return len(self.classes)


def ts_equal_length(path):
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().lower()
            if line:
                if line.startswith("@equallength") :
                    tokens = line.split(" ")
                    if tokens[1] == "false":
                        return False
                    elif tokens[1] == "true":
                        return True
    return True
    raise Exception("No equallength")
     
#-----------------------------------------------------------#

#   用于获取当前时间，保存数据时可以作为标签

#-----------------------------------------------------------#

def get_time_str(style='Nonetype'):
    t = time.localtime()
    if style is 'Nonetype':
        return ("{}{}{}{}{}{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    elif style is 'underline':
        return ("{}_{}_{}_{}_{}_{}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec))
    

