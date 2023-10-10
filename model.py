import abc
import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim, nn
from tqdm import tqdm

from unet3plus.loss.hybridLoss import HybridLoss
from unet3plus.models.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup, UNet_3Plus_DeepSup_CGM


class BaseUnetModule(pl.LightningModule, abc.ABC):
    """
    Based on https://github.com/hiepph/unet-lightning/blob/master/Unet.py
    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        cutoff_height: int,
        data_root: str,
        log_every_step: bool = False,
    ):
        super(BaseUnetModule, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_root = data_root
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
        patches, coords, filenames, slice_idx, pts = batch
        coords = torch.stack(coords).transpose(0, 1)
        patch_original_size = pts[0].item()
        p = patch_original_size // 2
        resize = transforms.Resize(patch_original_size, antialias=True)

        open_files = {}
        for patch, coord, filename, slice_i in zip(
            patches, coords, filenames, slice_idx
        ):
            if torch.all(patch == 0):
                continue
            if coord[0] > self.cutoff_height:
                continue
            output = (
                resize(self.predict_mask(patch.unsqueeze(0)))
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            ix, iy = coord

            if filename not in open_files:
                open_files[filename] = np.load(filename)
            open_files[filename][
                0,
                slice_i,
                ix - p : ix + p,
                iy - p : iy + p,
            ] += output
            open_files[filename][
                1,
                slice_i,
                ix - p : ix + p,
                iy - p : iy + p,
            ] += 1

        for filename in open_files.keys():
            np.save(filename, open_files[filename])

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
            os.remove(f)
            np.save(f, np.zeros(recon_original_shape).astype(np.float16))

        for threshold in thresholds:
            dice_scores[threshold] = np.mean(dice_scores[threshold])
            self.log_stat(
                f"{mode}_dice_coeff_{np.round(threshold, 1)}", dice_scores[threshold], on_step=False
            )

    def log_stat(self, name, stat, on_step=True):
        on_step = on_step and self.log_every_step
        self.log(
            name,
            stat,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.update_pred_masks(batch)

    def on_validation_epoch_end(self) -> None:
        self.compute_eval("val")


class UNet3plusModule(BaseUnetModule):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = UNet_3Plus(n_channels)
        self.loss = F.binary_cross_entropy_with_logits

    def compute_loss(self, batch, mode):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_stat(f"{mode}_bce_loss", loss)
        return loss

    def predict_mask(self, x):
        y_hat = self(x)
        y_hat = F.sigmoid(y_hat)
        return y_hat


class UNet3plusDsModule(BaseUnetModule):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = UNet_3Plus_DeepSup(n_channels)
        self.loss = HybridLoss()

    def compute_loss(self, batch, mode):
        x, y = batch
        d1_hat, d2_hat, d3_hat, d4_hat, d5_hat = self(x)
        loss = 0
        # TODO tmp: direct supervision on each level
        for i, d_hat in enumerate([d1_hat, d2_hat, d3_hat, d4_hat, d5_hat]):
            loss_d = self.loss(d_hat, y)
            loss += loss_d
            self.log_stat(f"{mode}_hybrid_loss_d{i+1}", loss_d)
        self.log_stat(f"{mode}_hybrid_loss", loss)
        return loss

    def predict_mask(self, x):
        d1_hat, _, _, _, _ = self(x)
        return d1_hat


class UNet3plusDsCgmModule(BaseUnetModule):
    def __init__(self, n_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = UNet_3Plus_DeepSup_CGM(n_channels)
        self.seg_loss = HybridLoss()
        self.bce_loss = F.binary_cross_entropy_with_logits

    def compute_loss(self, batch, mode):
        x, y = batch
        d1_hat, d2_hat, d3_hat, d4_hat, d5_hat, cls_hat = self(x)

        loss_seg = 0
        # TODO tmp: direct supervision on each level
        for i, d_hat in enumerate([d1_hat, d2_hat, d3_hat, d4_hat, d5_hat]):
            loss_d = self.seg_loss(d_hat, y)
            loss_seg += loss_d
            self.log_stat(f"{mode}_segmentation_loss_d{i+1}", loss_d)

        cls_true = (y.sum(dim=(1, 2, 3)) > 0).long()  # cls=0/1
        cls_true = F.one_hot(cls_true, num_classes=cls_hat.shape[1]).float()  # one-hot encoded, same shape as cls_hat
        loss_cls = self.bce_loss(cls_hat, cls_true)

        loss = loss_seg + loss_cls  # TODO lambda weight
        self.log_stat(f"{mode}_segmentation_loss", loss_seg)
        self.log_stat(f"{mode}_classification_loss", loss_cls)
        self.log_stat(f"{mode}_loss", loss)
        return loss

    def predict_mask(self, x):
        d1_hat, _, _, _, _, cls_hat = self(x)
        # cls is a tensor (B,2) with binary class probs
        cls = np.argmax(cls_hat.detach().cpu().numpy(), axis=1)
        y_hat = d1_hat * cls  # TODO test ok
        return y_hat
