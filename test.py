import os

import pytorch_lightning as pl
import argparse
from model import (UNet3plusDsCgmModule, UNet3plusDsModule, UNet3plusModule,
                   UNetModule)
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb
from dataset import RibFracDataset
from argparse import BooleanOptionalAction
from utils import load_config, config_from_args
import torch.utils.data as data

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

    args = parser.parse_args()
    cfg_test = config_from_args(args, mode="test")
    cfg_train = load_config(os.path.join(os.path.dirname(args.ckpt), "config_train.json"), mode="train")
    # TODO configs


    test_set = RibFracDataset(
        root_dir=cfg.data_root,
        partition="test",
        context_size=cfg.context_size,
        patch_original_size=cfg.patch_original_size,
        patch_final_size=cfg.patch_final_size,
        proportion_fracture_in_patch=cfg.proportion_fracture_in_patch,
        cutoff_height=cfg.cutoff_height,
        clip_min_val=cfg.clip_min_val,
        clip_max_val=cfg.clip_max_val,
        data_mean=cfg.data_mean,
        data_std=cfg.data_std,
        test_stride=cfg.test_stride,
        force_data_info=cfg.force_data_info,
        use_positional_encoding=cfg.use_positional_encoding,
    )
    test_sampler = test_set.get_test_sampler()
    test_loader = data.DataLoader(
        test_set,
        sampler=test_sampler,
        batch_size=cfg.batch_size_test,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    if cfg.use_model == "unet3plus":
        model_module = UNet3plusModule
    elif cfg.use_model == "unet3plus-ds":
        model_module = UNet3plusDsModule
    elif cfg.use_model == "unet3plus-ds-cgm":
        model_module = UNet3plusDsCgmModule
    elif cfg.use_model == "unet":
        model_module = UNetModule
    else:
        raise NotImplementedError(f"Unknown model {cfg.use_model}")

    model = model_module.load_from_checkpoint(cfg.ckpt)

    # TODO wandb

    # pytorch lightning inference
    trainer = pl.Trainer(
        accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",
        devices=1,
        logger=None,
    )
    trainer.test(model, test_loader)  # TODO log results img&loss&metrics
