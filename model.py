import abc
import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from tqdm import tqdm
import pandas as pd
from numpy.lib.format import open_memmap

from unet3plus.loss import FocalLoss, IOUloss, MSSSIMloss
from unet3plus.metrics import FPRmetric, IOUmetric
from unet3plus.models.UNet import Unet
from unet3plus.models.UNet_3Plus import (UNet_3Plus, UNet_3Plus_DeepSup,
                                         UNet_3Plus_DeepSup_CGM)


class BaseUnetModule(pl.LightningModule, abc.ABC):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        cutoff_height: int,
        data_root: str,
        use_focal_loss: bool = True,
        use_msssim_loss: bool = True,
        log_every_step: bool = False,
    ):
        super(BaseUnetModule, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_root = data_root
        self.use_focal_loss = use_focal_loss
        self.use_msssim_loss = use_msssim_loss
        self.cutoff_height = cutoff_height
        self.log_every_step = log_every_step

        self.network = ...  # UNet_3Plus(n_channels)
        self.loss = ...  # F.binary_cross_entropy_with_logits

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = optim.RMSprop(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    @abc.abstractmethod
    def compute_loss(self, batch, mode):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_stat(f"{mode}_bce_loss", loss)
        return loss

    @abc.abstractmethod
    def predict_mask(self, x):
        y_hat = self(x)
        y_hat = F.sigmoid(y_hat)
        return y_hat

    def update_pred_masks(self, batch):
        patches, coords, filenames, pts, ctx, shapes = batch
        coords = torch.stack(coords).transpose(0, 1)
        patch_original_size = pts[0].item()
        p = patch_original_size // 2
        self.p = p
        context_size = ctx[0].item()
        resize = transforms.Resize(patch_original_size, antialias=True)

        pred_patches = (
                resize(self.predict_mask(patches))
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )

        open_files = {}
        for pred, patch, coord, filename, shape in zip(
            pred_patches, patches, coords, filenames, shapes
        ):
            if torch.all(patch[context_size] < 0.05):
                continue
            if np.all(pred < 0.05):
                continue
            if coord[0] > self.cutoff_height:
                continue
            
            ix, iy = coord
            side = shape[-1] + 2 * p
            shape = (2, side, side)

            if filename not in open_files:
                open_files[filename] = np.memmap(
                    filename,
                    dtype=np.float16,
                    mode="w+",
                    shape=shape,
                )

            open_files[filename][
                0,
                ix - p : ix + p,
                iy - p : iy + p,
            ] += pred
            open_files[filename][
                1,
                ix - p : ix + p,
                iy - p : iy + p,
            ] += 1

        for filename in open_files.keys():
            open_files[filename].flush()

    def postprocessing(self, mode):
        pred_dir = os.path.join(self.data_root, f"{mode}-pred-masks")
        shape = (2, 576, 576) # TODO: avoid hardcode, but it's ok for now
        
        for filename in os.listdir(pred_dir):
            filename = os.path.join(pred_dir, filename)
            arr = np.memmap(filename, dtype=np.float16, mode='r', shape=shape)
            arr_tmp = open_memmap('tmp.npy', mode='w+', dtype=np.float16, shape=shape)
            arr_tmp[:] = arr[:]
            np.save(filename, arr_tmp)
            os.remove('tmp.npy')

        pred_dir = os.path.join(self.data_root, f"{mode}-pred-masks-final")
        os.mkdir(pred_dir) if not os.path.exists(pred_dir) else None

        df = pd.read_csv(os.path.join(self.data_root, f"{mode}_data_info.csv"))
        df = df.drop_duplicates(subset=['img_filename'])[['img_filename', 'scan_shape', 'slice_idx']]

        for filename, shape in tqdm(df[['img_filename', 'scan_shape']].values, desc="Postprocessing"):
            filename = os.basename(filename)
            shape = tuple(map(int, shape[1:-1].split(', ')))
            pred_filename = os.path.join(pred_dir, filename.replace("image.nii", "pred_mask.npy").replace(".gz", ""))
            prediction = np.empty(shape)
            df_sub = df[df.img_filename == filename]
            for slice in df_sub['slice_idx'].values:
                f = filename.replace("image.nii", f"pred_mask_{slice:03d}.npy").replace(".gz", "")
                f = os.path.join(self.data_root, f"{mode}-pred-masks", f)
                arr = np.load(f)
                p = self.p
                arr[1][arr[1] == 0] = 1
                arr = arr[0][p : - p, p : - p] / (arr[1][p : - p, p : - p])
                arr = arr.T
                prediction[slice] = arr
            np.save(pred_filename, prediction.astype(np.float16))


    def compute_eval(self, mode):
        thresholds = np.linspace(0, 1, 11)
        smooth = 1e-6
        dice_scores = {}

        for filename in tqdm(
            os.listdir(os.path.join(self.data_root, f"ribfrac-{mode}-labels")),
            desc="Computing dice scores",
        ):
            f = os.path.join(self.data_root, f"ribfrac-{mode}-labels", filename)
            masks = nib.load(f).get_fdata().T.astype(int)
            f = os.path.join(
                self.data_root,
                f"{mode}-pred-masks",
                filename.replace("label.nii", "pred_mask.npy").replace(".gz", ""),
            )
            recon = np.load(f)
            recon_original_shape = recon.shape
            recon = recon[0] / recon[1]
            recon_side = recon.shape[-1]
            p = (recon_side - masks.shape[-1]) // 2
            recon = recon[:, p : recon_side - p, p : recon_side - p]
            recon[self.cutoff_height :] = 0
            for threshold in thresholds:
                dice_scores.setdefault(threshold, [])
                pred = (recon > threshold).astype(float)
                dice = (2 * (pred * masks).sum() + smooth) / (
                    pred.sum() + masks.sum() + smooth
                )
                dice_scores[threshold].append(dice)
            if mode == "val":
                os.remove(f)
                np.save(f, np.zeros(recon_original_shape).astype(np.float16))

        for threshold in thresholds:
            dice_scores[threshold] = np.mean(dice_scores[threshold])
            self.log_stat(
                f"{mode}_dice_coeff_{np.round(threshold, 1)}",
                dice_scores[threshold],
                on_step=False,
            )

    def log_stat(self, name, stat, on_step=True, prog_bar=False):
        on_step = on_step and self.log_every_step
        self.log(
            name,
            stat,
            on_step=on_step,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        loss, _ = self.compute_loss(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, pred = self.compute_loss(batch, "val")
        return pred

    def test_step(self, batch, batch_idx):
        self.update_pred_masks(batch)

    def on_test_end(self) -> None:
        self.postprocessing('test')


class UNetModule(BaseUnetModule):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = Unet(n_channels)
        self.loss = F.binary_cross_entropy_with_logits
        self.iou_metric = IOUmetric()
        self.fpr_metric = FPRmetric()

    def compute_loss(self, batch, mode):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = F.sigmoid(y_hat)
        with torch.no_grad():
            cls_true = (y.sum(dim=(1, 2, 3)) > 0).long()
            iou_metric = self.iou_metric(y_hat[cls_true == 1], y[cls_true == 1])
            fpr_metric = self.fpr_metric(y_hat, y)
        self.log_stat(f"{mode}_iou_metric", iou_metric, prog_bar=True)
        self.log_stat(f"{mode}_fpr_metric", fpr_metric, prog_bar=True)
        self.log_stat(f"{mode}_metric", (iou_metric + (1 - fpr_metric))/2, prog_bar=True)
        self.log_stat(f"{mode}_loss", loss, prog_bar=True)
        return loss

    def predict_mask(self, x):
        y_hat = self(x)
        y_hat = F.sigmoid(y_hat)
        return y_hat


class UNet3plusModule(BaseUnetModule):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = UNet_3Plus(n_channels)
        self.loss = F.binary_cross_entropy_with_logits
        self.iou_metric = IOUmetric()
        self.fpr_metric = FPRmetric()

    def compute_loss(self, batch, mode):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        y_hat = F.sigmoid(y_hat)
        with torch.no_grad():
            cls_true = (y.sum(dim=(1, 2, 3)) > 0).long()
            iou_metric = self.iou_metric(y_hat[cls_true == 1], y[cls_true == 1])
            fpr_metric = self.fpr_metric(y_hat, y)
        self.log_stat(f"{mode}_iou_metric", iou_metric, prog_bar=True)
        self.log_stat(f"{mode}_fpr_metric", fpr_metric, prog_bar=True)
        self.log_stat(f"{mode}_metric", (iou_metric + (1 - fpr_metric))/2, prog_bar=True)
        self.log_stat(f"{mode}_loss", loss, prog_bar=True)
        return loss, y_hat

    def predict_mask(self, x):
        y_hat = self(x)
        y_hat = F.sigmoid(y_hat)
        return y_hat


class UNet3plusDsModule(BaseUnetModule):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = UNet_3Plus_DeepSup(n_channels)
        self.iou_loss = IOUloss()
        self.focal_loss = FocalLoss()
        self.msssim_loss = MSSSIMloss()
        self.iou_metric = IOUmetric()
        self.fpr_metric = FPRmetric()

    def compute_loss(self, batch, mode):
        x, y = batch
        d1_hat, d2_hat, d3_hat, d4_hat, d5_hat = self(x)
        loss_seg_focal = 0
        loss_seg_iou = 0
        loss_seg_msssim = 0
        for i, d_hat in enumerate([d1_hat, d2_hat, d3_hat, d4_hat, d5_hat]):
            loss_d_iou = self.iou_loss(d_hat, y)
            loss_seg_iou = loss_seg_iou + loss_d_iou
            self.log_stat(f"{mode}_iou_loss_d{i+1}", loss_d_iou)
            if self.use_focal_loss:
                loss_d_focal = self.focal_loss(d_hat, y)
                loss_seg_focal = loss_seg_focal + loss_d_focal
                self.log_stat(f"{mode}_focal_loss_d{i+1}", loss_d_focal)
            if self.use_msssim_loss:
                loss_d_msssim = self.msssim_loss(d_hat, y)
                loss_seg_msssim = loss_seg_msssim + loss_d_msssim
                self.log_stat(f"{mode}_msssim_loss_d{i+1}", loss_d_msssim)
        loss = loss_seg_iou
        self.log_stat(f"{mode}_iou_loss", loss_seg_iou)
        if self.use_focal_loss:
            loss = loss + loss_seg_focal
            self.log_stat(f"{mode}_focal_loss", loss_seg_focal)
        if self.use_msssim_loss:
            loss = loss + loss_seg_msssim
            self.log_stat(f"{mode}_msssim_loss", loss_seg_msssim)
        with torch.no_grad():
            cls_true = (y.sum(dim=(1, 2, 3)) > 0).long()
            iou_metric = self.iou_metric(d1_hat[cls_true == 1], y[cls_true == 1])
            fpr_metric = self.fpr_metric(d1_hat, y)
        self.log_stat(f"{mode}_iou_metric", iou_metric, prog_bar=True)
        self.log_stat(f"{mode}_fpr_metric", fpr_metric, prog_bar=True)
        self.log_stat(f"{mode}_metric", (iou_metric + (1 - fpr_metric))/2, prog_bar=True)
        self.log_stat(f"{mode}_segmentation_loss", loss)
        self.log_stat(f"{mode}_loss", loss, prog_bar=True)
        return loss, d1_hat

    def predict_mask(self, x):
        d1_hat, _, _, _, _ = self(x)
        return d1_hat


class UNet3plusDsCgmModule(BaseUnetModule):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = UNet_3Plus_DeepSup_CGM(n_channels)
        self.iou_loss = IOUloss()
        self.focal_loss = FocalLoss()
        self.msssim_loss = MSSSIMloss()
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.iou_metric = IOUmetric()
        self.fpr_metric = FPRmetric()

    def compute_loss(self, batch, mode):
        x, y = batch
        d1_hat, d2_hat, d3_hat, d4_hat, d5_hat, cls_hat = self(x)

        loss_seg_focal = 0
        loss_seg_iou = 0
        loss_seg_msssim = 0
        for i, d_hat in enumerate([d1_hat, d2_hat, d3_hat, d4_hat, d5_hat]):
            loss_d_iou = self.iou_loss(d_hat, y)
            loss_seg_iou = loss_seg_iou + loss_d_iou
            self.log_stat(f"{mode}_iou_loss_d{i+1}", loss_d_iou)
            if self.use_focal_loss:
                loss_d_focal = self.focal_loss(d_hat, y)
                loss_seg_focal = loss_seg_focal + loss_d_focal
                self.log_stat(f"{mode}_focal_loss_d{i+1}", loss_d_focal)
            if self.use_msssim_loss:
                loss_d_msssim = self.msssim_loss(d_hat, y)
                loss_seg_msssim = loss_seg_msssim + loss_d_msssim
                self.log_stat(f"{mode}_msssim_loss_d{i+1}", loss_d_msssim)

        cls_true = (y.sum(dim=(1, 2, 3)) > 0).long()  # cls=0/1
        cls_true = F.one_hot(
            cls_true, num_classes=cls_hat.shape[1]
        ).float()  # one-hot encoded, same shape as cls_hat
        loss_cls = self.bce_loss(cls_hat, cls_true)

        loss_seg = loss_seg_iou
        self.log_stat(f"{mode}_iou_loss", loss_seg_iou)
        if self.use_focal_loss:
            loss_seg = loss_seg + loss_seg_focal
            self.log_stat(f"{mode}_focal_loss", loss_seg_focal)
        if self.use_msssim_loss:
            loss_seg = loss_seg + loss_seg_msssim
            self.log_stat(f"{mode}_msssim_loss", loss_seg_msssim)
        loss = loss_seg + loss_cls  # TODO lambda weight
        with torch.no_grad():
            cls_true = (y.sum(dim=(1, 2, 3)) > 0).long()
            iou_metric = self.iou_metric(d1_hat[cls_true == 1], y[cls_true == 1])
            fpr_metric = self.fpr_metric(d1_hat, y)
        self.log_stat(f"{mode}_iou_metric", iou_metric, prog_bar=True)
        self.log_stat(f"{mode}_fpr_metric", fpr_metric, prog_bar=True)
        self.log_stat(f"{mode}_metric", (iou_metric + (1 - fpr_metric))/2, prog_bar=True)
        self.log_stat(f"{mode}_segmentation_loss", loss_seg)
        self.log_stat(f"{mode}_classification_loss", loss_cls)
        self.log_stat(f"{mode}_loss", loss, prog_bar=True)
        return loss, d1_hat

    def predict_mask(self, x):
        d1_hat, _, _, _, _, cls_hat = self(x)
        # cls is a tensor (B,2) with binary class probs
        cls = torch.argmax(cls_hat, dim=1)
        y_hat = d1_hat * cls
        return y_hat
