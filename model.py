import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn, optim


class NetworkModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.network = Network()

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def compute_loss(self, batch, mode):
        x, _ = batch
        recon = self(x)
        loss = F.mse_loss(x, recon)
        self.log_stat(f"{mode}_mse_loss", loss)
        return loss

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
        _ = self.compute_loss(batch, "val")


class Network(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x
