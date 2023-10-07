import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from tqdm import tqdm

from unet3plus.model_unet3plus import Unet3Plus
from unet3plus.model_unet3plus_deep_supervision_cgm import UNet_3Plus_DeepSup_CGM

from loss import MSSSIM, IOU, FocalLoss

class UnetModule(pl.LightningModule):
    """
    Based on https://github.com/hiepph/unet-lightning/blob/master/Unet.py
    """

    def __init__(
        self,
        model_name: str,
        n_channels: int,
        learning_rate: float,
        weight_decay: float,
        cutoff_height: int,
        data_root: str,
    ):
        super(UnetModule, self).__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_root = data_root
        self.cutoff_height = cutoff_height

        if model_name == "unet3plus":
            self.network = Unet3Plus(n_channels)
        # elif model_name == "unet3plus_deepsup":
        #     self.network = UNet_3Plus_DeepSup(n_channels)
        elif model_name == "unet3plus_deep_sup_cgm":
            self.network = UNet_3Plus_DeepSup_CGM(n_channels)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = optim.RMSprop(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def compute_loss(self, batch, mode):
        x, y = batch

        # Without Classification guided model (CGM)
        if self.model_name == "unet3plus":
            y_hat = self(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, y)
            self.log_stat(f"{mode}_bce_loss", loss)
            return loss
        # With CGM
        elif self.model_name == "unet3plus_deepsup" or self.model_name == "unet3plus_deepsup_cgm":
            is_frac = y.reshape(y.shape[0], -1).sum(dim=1) > 0 # is there a fracture in this patch?
            is_frac = is_frac.float()
            cls_branch, (d1), (d2), d3, (d4),(d5) = self(x)
 
            loss_BCE = torch.binary_cross_entropy_with_logits(cls_branch[:,1], is_frac)
            
            # segmentation loss
            img1 = torch.concatenate([d1, d2, d3, d4, d5])    
            img2 = torch.concatenate([y, y, y, y, y])

            ## Hybrid loss: MSSSIM + IOU + FocalLoss
            loss_seg = MSSSIM()(img1, img2) + IOU()(img1, img2) + FocalLoss()(img1, img2)
             
            loss = loss_BCE + loss_seg  
            loss = loss.mean()
            
            # self.log_stat(f"{mode}_bce_loss", loss)
            # self.log_stat(f"{mode}_bce_loss", loss)
            # self.log_stat(f"{mode}_bce_loss", loss)
            # self.log_stat(f"{mode}_bce_loss", loss)
            
            return loss
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

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

                resize(F.sigmoid(self(patch.unsqueeze(0))))
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
                f"{mode}_dice_coeff_{np.round(threshold, 1)}", dice_scores[threshold]
            )

    def log_stat(self, name, stat):
        self.log(
            name,
            stat,
            on_step=False,
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
