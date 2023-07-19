import torch 
import numpy as np
import matplotlib.pyplot as plt

def tensor_predict(output):
    """return the predict of output. tensor version.

    Args:
        output (torch.tensor): the output of model, size is (batch, num_classes)

    Returns:
        torch.Tensor: the predict of output, size is (batch, )
        
    Examples:
        >>> output = torch.tensor([[0.1, 0.2, 0.3, 0.4],
        >>>                        [0.4, 0.3, 0.2, 0.1],
        >>>                      [0.1, 0.2, 0.3, 0.4]])
        >>> predict = tensor_predict(output)
        >>> print(predict)
        tensor([3, 0, 3]) 
    """
    return output.max(1)[1].cpu().numpy()

def numpy_predict(output_ndarray):
    """return the predict of output. numpy version.

    Args:
        output_ndarray (numpy.ndarray): the output of model, size is (batch, num_classes)

    Returns:
        np.array: the predict of output, size is (batch, )
        
    Examples:
        >>> output = np.array([[0.1, 0.2, 0.3, 0.4],
        >>>                        [0.4, 0.3, 0.2, 0.1],
        >>>                      [0.1, 0.2, 0.3, 0.4]])
        >>> predict = numpy_predict(output)
        >>> print(predict)
        array([3, 0, 3])
    """
    return np.argmax(output_ndarray, axis=1)

def accuracy(output, target, topk=(1,)):
    """calculate accuracy of output and target with topk. tensor version.

    Args:
        output (torch.tensor): output of model
        target (torch.tensor): target of model
        topk (tuple, optional): topk. Defaults to (1,).

    Returns:
        list: accuracy of output and target with topk
        
    Examples:
        >>> output = torch.tensor([[0.1, 0.2, 0.3, 0.4],
        >>>                        [0.4, 0.3, 0.2, 0.1],
        >>>                       [0.1, 0.2, 0.3, 0.4],
        >>>                       [0.4, 0.3, 0.2, 0.1]])
        >>> target = torch.tensor([3, 2, 1, 0])
        >>> accuracy(output, target, topk=(1, 2))
        [50.0, 75.0]
    """
    output = torch.nn.functional.softmax(output, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True) # (batch_size, maxk)
    pred = pred.t() # (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # (maxk, batch_size)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def accuracy_np(output, target, topk=(1,)):
    """calculate accuracy of output and target with topk. numpy version.

    Args:
        output (numpy.ndarray): output of model
        target (numpy.ndarray): target of model
        topk (tuple, optional): topk. Defaults to (1,).

    Returns:
        list: accuracy of output and target with topk
        
    Examples:
        >>> output = np.array([[0.1, 0.2, 0.3, 0.4],
        >>>                        [0.4, 0.3, 0.2, 0.1],
        >>>                       [0.1, 0.2, 0.3, 0.4],
        >>>                       [0.4, 0.3, 0.2, 0.1]])
        >>> target = np.array([3, 2, 1, 0])
        >>> accuracy(output, target, topk=(1, 2))
        [50.0, 75.0]
    """
    output = torch.nn.functional.softmax(torch.from_numpy(output), dim=1)
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True) # (batch_size, maxk)
    pred = pred.t() # (maxk, batch_size)
    correct = pred.eq(torch.from_numpy(target).view(1, -1).expand_as(pred)) # (maxk, batch_size)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def precision(output, target, topk=(1,)):
    output = torch.nn.functional.softmax(output, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True) # (batch_size, maxk)
    pred = pred.t() # (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # (maxk, batch_size)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        pred_k = pred[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/pred_k))
    return res

def f1score(output, target):
    output = torch.nn.functional.softmax(output, dim=1)
    _, pred = output.topk(1, dim=1, largest=True, sorted=True) # (batch_size, maxk)
    pred = pred.t() # (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # (maxk, batch_size)
    correct_k = correct[:1].view(-1).float().sum(0)
    pred_k = pred[:1].view(-1).float().sum(0)
    return correct_k.mul_(2.0/pred_k)

def f1score_np(output, target):
    output = torch.nn.functional.softmax(torch.from_numpy(output), dim=1)
    _, pred = output.topk(1, dim=1, largest=True, sorted=True) # (batch_size, maxk)
    pred = pred.t() # (maxk, batch_size)
    correct = pred.eq(torch.from_numpy(target).view(1, -1).expand_as(pred)) # (maxk, batch_size)
    correct_k = correct[:1].view(-1).float().sum(0)
    pred_k = pred[:1].view(-1).float().sum(0)
    return correct_k.mul_(2.0/pred_k)

def recall(output, target):
    output = torch.nn.functional.softmax(output, dim=1)
    _, pred = output.topk(1, dim=1, largest=True, sorted=True) # (batch_size, maxk)
    pred = pred.t() # (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # (maxk, batch_size)
    correct_k = correct[:1].view(-1).float().sum(0)
    target_k = target.view(-1).float().sum(0)
    return correct_k.mul_(100.0/target_k)


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
