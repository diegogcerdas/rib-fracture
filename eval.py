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
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint to test.",
    )
    parser.add_argument(
        "--data-root", 
        type=str, 
        default="./data/", 
        help="Root directory for data."
    )
    parser.add_argument(
        "--download-data",
        action=BooleanOptionalAction,
        default=False,
        help="Download data in specified data directory if not present.",
    )

    # Run Parameters
    parser.add_argument(
        "--test-stride",
        type=int,
        default=32,
        help="Stride for test/val patches.",
    )
    parser.add_argument(
        "--batch-size-test",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # Wandb Parameters
    parser.add_argument("--do-wandb", action=BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-key", type=str, default=None)
    # use rest of config from train, if applicable

    args = parser.parse_args()
    cfg_test = config_from_args(args, mode="test")
    cfg_train = load_config(os.path.join(os.path.dirname(args.ckpt), "config_train.json"), mode="train")
    
    exp_name = "eval_" + cfg_train.exp_name

    # dataloader
    test_set = RibFracDataset(
        root_dir=cfg_test.data_root,
        partition="test",
        context_size=cfg_train.context_size,
        patch_original_size=cfg_train.patch_original_size,
        patch_final_size=cfg_train.patch_final_size,
        proportion_fracture_in_patch=cfg_train.proportion_fracture_in_patch,
        cutoff_height=cfg_train.cutoff_height,
        clip_min_val=cfg_train.clip_min_val,
        clip_max_val=cfg_train.clip_max_val,
        data_mean=cfg_train.data_mean,
        data_std=cfg_train.data_std,
        test_stride=cfg_test.test_stride,
        force_data_info=cfg_train.force_data_info,
        use_positional_encoding=cfg_train.use_positional_encoding,
    )
    test_sampler = test_set.get_test_sampler()
    test_loader = data.DataLoader(
        test_set,
        sampler=test_sampler,
        batch_size=cfg_test.batch_size_test,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg_test.num_workers,
    )

    # model
    if cfg_train.use_model == "unet3plus":
        model_module = UNet3plusModule
    elif cfg_train.use_model == "unet3plus-ds":
        model_module = UNet3plusDsModule
    elif cfg_train.use_model == "unet3plus-ds-cgm":
        model_module = UNet3plusDsCgmModule
    elif cfg_train.use_model == "unet":
        model_module = UNetModule
    else:
        raise NotImplementedError(f"Unknown model {cfg_train.use_model}")

    model = model_module.load_from_checkpoint(cfg_test.ckpt)

    # logger
    logger = []
    if cfg_test.do_wandb:
        wandb.login(key=cfg_test.wandb_key)
        wandb.init(
            name=exp_name,
            project=cfg_train.wandb_project,
            entity=cfg_train.wandb_entity,
            mode=cfg_train.wandb_mode,
        )
        wandb_logger = WandbLogger()
        logger.append(wandb_logger)
    logger.append(CSVLogger("./logs/", name=exp_name, flush_logs_every_n_steps=1))

    # pytorch lightning inference
    trainer = pl.Trainer(
        accelerator="gpu" if str(cfg_test.device).startswith("cuda") else "cpu",
        devices=1,
        logger=None,
    )
    trainer.test(model, test_loader)
    #trainer.test(model, test_loader, ckpt_path=cfg_test.ckpt)

    # TODO VizCallback(cfg_train.context_size)
