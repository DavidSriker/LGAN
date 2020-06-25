import numpy as np


def IoU(gt, pred):
    """ returns Intersection over Union (IoU) score for ground truth and predicted masks """
    assert gt.dtype == bool and pred.dtype == bool
    gt_f = gt.flatten()
    pred_f = pred.flatten()
    intersection = np.logical_and(gt_f, pred_f).sum()
    union = np.logical_or(gt_f, pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)


def Dice(gt, pred):
    """ returns Dice Similarity (Dice) for ground truth and predicted masks """
    assert gt.dtype == bool and pred.dtype == bool
    gt_f = gt.flatten()
    pred_f = pred.flatten()
    intersection = np.logical_and(gt_f, pred_f).sum()
    return (2. * intersection + 1.) / (gt.sum() + pred.sum() + 1.)