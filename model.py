import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn, optim
from tqdm import tqdm


class UnetModule(pl.LightningModule):
    """
    Based on https://github.com/hiepph/unet-lightning/blob/master/Unet.py
    """

    def __init__(
        self, n_channels: int, learning_rate: float, weight_decay: float, data_root: str
    ):
        super(UnetModule, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_root = data_root

        self.network = Unet(n_channels)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = optim.RMSprop(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def compute_loss(self, batch, mode):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log_stat(f"{mode}_bce_loss", loss)
        return loss

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
                filename.replace("label.nii", "pred_mask.npy"),
            )
            recon = np.load(f)
            recon = recon[0] / recon[1]
            recon_side = recon.shape[-1]
            p = (recon_side - masks.shape[-1]) // 2
            recon = recon[:, p : recon_side - p, p : recon_side - p]
            for threshold in thresholds:
                dice_scores.setdefault(threshold, [])
                pred = (recon > threshold).astype(float)
                dice = (2 * (pred * masks).sum() + smooth) / (
                    pred.sum() + masks.sum() + smooth
                )
                dice_scores[threshold].append(dice)

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


class Unet(nn.Module):
    def __init__(self, n_channels):
        super(Unet, self).__init__()
        self.n_channels = n_channels

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2), double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    )
                else:
                    self.up = nn.ConvTranpose2d(
                        in_channels // 2, in_channels // 2, kernel_size=2, stride=2
                    )

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(
                    x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                )
                x = torch.cat([x2, x1], dim=1)  ## why 1?
                return self.conv(x)

        self.inc = double_conv(self.n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)
