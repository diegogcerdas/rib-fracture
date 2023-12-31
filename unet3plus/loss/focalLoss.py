"""
Translated to pytorch from https://github.com/hamidriasat/UNet-3-Plus/blob/unet3p_lits/losses/loss.py
"""
import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _focal(pred, target, size_average=self.size_average)


def _focal(y_true, y_pred, size_average=True, gamma=2.0, alpha=4.0, epsilon=1.0e-9):
    y_true_c = y_true.float()
    y_pred_c = y_pred.float()

    model_out = y_pred_c + epsilon
    ce = -y_true_c * torch.log(model_out)
    weight = y_true_c * torch.pow(1 - model_out, gamma)
    fl = alpha * weight * ce
    reduced_fl = torch.max(fl, dim=-1).values
    if size_average:
        return reduced_fl.mean()
    else:
        return reduced_fl.sum()
