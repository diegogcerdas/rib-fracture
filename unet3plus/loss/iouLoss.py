import torch


def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU_loss = 0.0
    for i in range(b):
        # compute the IoU of the foreground
        intersection = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        union = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - intersection
        IoU = intersection / union

        IoU_loss += (1 - IoU)

    if size_average:
        return IoU_loss / b
    else:
        return IoU_loss


class IOUloss(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOUloss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)
    

def _iou_metric(pred, target, size_average=True):
    b = pred.shape[0]
    IoU_metric = 0.0
    for i in range(b):
        # compute the IoU of the foreground
        intersection = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        union = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - intersection
        IoU_metric += intersection / union

    if size_average:
        return IoU_metric / b
    else:
        return IoU_metric


class IOUmetric(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOUmetric, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou_metric(pred, target, self.size_average)
