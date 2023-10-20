import pytorch_lightning as pl
import argparse
from model import (UNet3plusDsCgmModule, UNet3plusDsModule, UNet3plusModule,
                   UNetModule)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb
from dataset import RibFracDataset
from argparse import BooleanOptionalAction
from utils import config_from_args

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root", 
        type=str, 
        default="./data/", 
        help="Root directory for data."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint to test.",
    )
    parser.add_argument(
        "--use-model",
        type=str,
        choices=["unet3plus", "unet3plus-ds", "unet3plus-ds-cgm", "unet"],
        required=True,
        help="Unet3plus model used in text ckpt. Valid options: unet3plus, unet3plus-ds, unet3plus-ds-cgm",
    )

    # WandB Parameters
    parser.add_argument("--do-wandb", action=BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="ribfrac_experiments")
    parser.add_argument("--wandb-entity", type=str, default="ribfrac_team7")
    parser.add_argument("--wandb-mode", type=str, default="online")
    args = parser.parse_args()
    cfg = config_from_args(args, mode="test")

