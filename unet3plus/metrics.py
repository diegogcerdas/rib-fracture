import torch

class IOUmetric(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOUmetric, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return self._iou_metric(pred, target)
    
    def _iou_metric(self, pred, target):
        b = pred.shape[0]
        IoU_metric = 0.0
        for i in range(b):
            # compute the IoU of the foreground
            intersection = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            union = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - intersection
            IoU_metric += intersection / union

        if self.size_average:
            return IoU_metric / b
        else:
            return IoU_metric
        
class FPRmetric(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred, target):
        return self._fpr_metric(pred, target)
    
    def _fpr_metric(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        TN = torch.sum((1 - pred_flat) * (1 - target_flat))
        FP = torch.sum(pred_flat * (1 - target_flat))
        return FP / (TN + FP)
    

