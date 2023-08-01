import torch 
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    """swich activate func

    Args:
        x (tensor): input

    Returns:
        tensor: the result of swich activate func
    """
    return x * torch.sigmoid(x)


def dice_loss(target, predictive, ep=1e-8):
    """dice loss for autogress.

    From paper 'V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation'. In fact, dice loss is equal to the f1 score in terms of results(both are 2TP/(2TP+FP+FN)). Referd by the article 'https://zhuanlan.zhihu.com/p/269592183'
    来自于论文'V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation', dice loss结果上等同于f1-score, 为2TP/(2TP+FP+FN).可以参考文章'https://zhuanlan.zhihu.com/p/269592183'

    Args:
      target:
        forecast traget, the groudtruth.
        预测目标
      predictive:
        forecast result
        预测结果
      ep:
        a small number, used to prevent division by zero.
        小参数,用于防止除0
    
    Returns:
      loss
    """
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss



class Tripletloss(nn.Module):
    """Triplet Loss 

    From paper 'FaceNet: A Unified Embedding for Face Recognition and Clustering
'

    Attributes:


    """
    def __init__(self, margin=None):
        super(Tripletloss, self).__init__()
        self.margin = margin
        if self.margin is None:
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss 




def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'