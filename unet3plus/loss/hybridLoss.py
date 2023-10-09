from focalLoss import FocalLoss
from iouLoss import IOUloss
from msssimLoss import MSSSIMloss
from torch import nn


class HybridLoss(nn.Module):
    def __init__(self, size_average=True):
        super(HybridLoss, self).__init__()
        self.focal_loss = FocalLoss(size_average=size_average)
        self.iou_loss = IOUloss(size_average=size_average)
        self.msssim_loss = MSSSIMloss(size_average=size_average)

    def forward(self, pred, target):
        return self.focal_loss(pred, target) + self.iou_loss(pred, target) + self.msssim_loss(pred, target)
