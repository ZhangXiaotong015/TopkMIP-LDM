import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Dice, Precision, Recall, F1Score, Accuracy
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
import nibabel
import numpy as np

def save_nifti(img, img_path):
    pair_img = nibabel.Nifti1Pair(img,np.eye(4))
    nibabel.save(pair_img,img_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val==-1:
            return 
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
def dice(preds,target):
    """2TP / (2TP + FP + FN)"""
    dice_m = Dice(average='micro',ignore_index=0).cuda() # exclude the background
    return dice_m(preds,target)

def precision(preds,target,task='binary'):
    """TP / (TP + FP)"""
    precision_m = Precision(task=task).cuda()
    return precision_m(preds,target)

def recall(preds,target,task='binary'):
    """TP / (TP + FN)"""
    recall_m = Recall(task=task).cuda()
    return recall_m(preds,target)

def fscore(preds,target,task='binary'):
    f1_m = F1Score(task=task).cuda()
    return f1_m(preds,target)

def accuracy(preds,target,task='binary'):
    acc_m = Accuracy(task=task).cuda()
    return acc_m(preds,target)

def mcprecision(preds,target,task='multiclass',num_classes=3):
    """TP / (TP + FP)"""
    precision_m = MulticlassPrecision(num_classes=num_classes, average=None, ignore_index=None).cuda()
    return precision_m(preds,target)

def mcrecall(preds,target,task='multiclass',num_classes=3):
    """TP / (TP + FN)"""
    recall_m = MulticlassRecall(num_classes=num_classes, average=None, ignore_index=None).cuda()
    return recall_m(preds,target)

def mcfscore(preds,target,task='multiclass',num_classes=3):
    f1_m = MulticlassF1Score(num_classes=num_classes, average=None, ignore_index=None).cuda()
    return f1_m(preds,target)

def mcaccuracy(preds,target,task='multiclass',num_classes=3):
    acc_m = MulticlassAccuracy(num_classes=num_classes, average=None, ignore_index=None).cuda()
    return acc_m(preds,target)