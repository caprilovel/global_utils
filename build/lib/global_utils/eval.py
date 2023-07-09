import torch 
import numpy as np

def accuracy(output, target, topk=(1,)):
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