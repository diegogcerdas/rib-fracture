import argparse

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from pytorch_lightning.loggers import WandbLogger

import wandb
from dataset import RibFracDataset
from model import UnetModule
from utils import config_from_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument("--data-root", type=str, default="./data/")

    # Model Parameters
    parser.add_argument("--context-size", type=int, default=0)

    # Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # WandB Parameters
    parser.add_argument("--wandb-name", type=str, default="test-run")
    parser.add_argument("--wandb-project", type=str, default="rib-frac")
    parser.add_argument("--wandb-entity", type=str, default="diego-gcerdas")
    parser.add_argument("--wandb-mode", type=str, default="online")
    args = parser.parse_args()
    cfg = config_from_args(args, mode="train")

    wandb.init(
        name=cfg.wandb_name,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
    )
    wandb_logger = WandbLogger()
    logger = [wandb_logger]
    callbacks = []

    pl.seed_everything(cfg.seed)

    train_set = RibFracDataset(
        cfg.data_root, set="train", context_size=cfg.context_size
    )
    train_loader = data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    val_set = RibFracDataset(cfg.data_root, set="val", context_size=cfg.context_size)
    val_loader = data.DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.num_workers,
    )

    model = UnetModule(
        n_channels=1 + 2 * cfg.context_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    trainer = pl.Trainer(
        accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=2,
    )

    trainer.fit(model, train_loader, val_loader)
