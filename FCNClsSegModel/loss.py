
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class LossMulti(nn.Module):
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            self.nll_weight = class_weights#Variable(class_weights.float()).cuda()
        else:
            self.nll_weight = None

        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):

        loss = (1 - self.jaccard_weight) * F.cross_entropy(outputs, targets, weight=self.nll_weight)

        if self.jaccard_weight:
            eps = 1e-15
            outputs = F.softmax(outputs, dim=1)
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls]#.exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                # print(union)
                # print(intersection)
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss



def Weighted_Jaccard_loss(label, pred, class_weights=None, gpu=0, num_classes=3):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    label = Variable(label.long()).cuda(gpu)
    if class_weights is not None  and class_weights != 0:
        class_weights = torch.Tensor(class_weights)
        class_weights = Variable(class_weights).cuda(gpu)
        criterion = LossMulti(jaccard_weight=0.5, class_weights=class_weights,num_classes=num_classes)#.cuda(gpu)
    else:
        criterion = LossMulti(jaccard_weight=0.5, num_classes=num_classes)  # .cuda(gpu)
    return criterion(pred, label)


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
        https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)




