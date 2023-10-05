import argparse
from argparse import BooleanOptionalAction

import pytorch_lightning as pl
import torch
import torch.utils.data as data
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import RibFracDataset
from model import UnetModule
from utils import SetEpochCallback, config_from_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data Parameters
    parser.add_argument(
        "--data-root", type=str, default="./data/", help="Root directory for data."
    )
    parser.add_argument(
        "--ckpt-root", type=str, default="./checkpoints/", help="Root directory for checkpoints."
    )
    parser.add_argument(
        "--patch-original-size",
        type=int,
        default=64,
        help="Size of the patches extracted from original images.",
    )
    parser.add_argument(
        "--patch-final-size",
        type=int,
        default=256,
        help="Size of the patches after resizing.",
    )
    parser.add_argument(
        "--proportion-fracture-in-patch",
        type=float,
        default=0.05,
        help="Proportion of fracture pixels in a patch.",
    )
    parser.add_argument(
        "--clip-min-val",
        type=int,
        default=100,
        help="Lower threshold to clip intensity values",
    )
    parser.add_argument(
        "--clip-max-val",
        type=int,
        default=8000,
        help="Upper threshold to clip intensity values.",
    )
    parser.add_argument(
        "--test-stride", type=int, default=32, help="Stride for test/val patches."
    )
    parser.add_argument(
        "--force-data-info",
        action=BooleanOptionalAction,
        default=False,
        help="Force data info generation.",
    )

    # Model Parameters
    parser.add_argument(
        "--context-size",
        type=int,
        default=32,
        help="Number of slices above and below the middle slice.",
    )

    # Training Parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size-train", type=int, default=64)
    parser.add_argument("--batch-size-test", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--exp-name", type=str, default="test-run")
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # WandB Parameters
    parser.add_argument("--do-wandb", action=BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="rib-frac")
    parser.add_argument("--wandb-entity", type=str, default="diego-gcerdas")
    parser.add_argument("--wandb-mode", type=str, default="online")
    args = parser.parse_args()
    cfg = config_from_args(args, mode="train")

    pl.seed_everything(cfg.seed)

    train_set = RibFracDataset(
        root_dir=cfg.data_root,
        partition="train",
        context_size=cfg.context_size,
        patch_original_size=cfg.patch_original_size,
        patch_final_size=cfg.patch_final_size,
        proportion_fracture_in_patch=cfg.proportion_fracture_in_patch,
        clip_min_val=cfg.clip_min_val,
        clip_max_val=cfg.clip_max_val,
        test_stride=cfg.test_stride,
        force_data_info=cfg.force_data_info,
    )
    train_sampler = train_set.get_train_sampler(seed=cfg.seed)
    train_loader = data.DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=cfg.batch_size_train,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    val_set = RibFracDataset(
        root_dir=cfg.data_root,
        partition="val",
        context_size=cfg.context_size,
        patch_original_size=cfg.patch_original_size,
        patch_final_size=cfg.patch_final_size,
        proportion_fracture_in_patch=cfg.proportion_fracture_in_patch,
        clip_min_val=cfg.clip_min_val,
        clip_max_val=cfg.clip_max_val,
        test_stride=cfg.test_stride,
        force_data_info=cfg.force_data_info,
    )
    val_sampler = val_set.get_test_sampler()
    val_loader = data.DataLoader(
        val_set,
        sampler=val_sampler,
        batch_size=cfg.batch_size_test,
        drop_last=False,
        num_workers=cfg.num_workers,
    )

    model = UnetModule(
        n_channels=1 + 2 * cfg.context_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        data_root=cfg.data_root,
    )

    logger = []
    checkpoint_callback = ModelCheckpoint(dirpath=f'{cfg.ckpt_root}/{cfg.exp_name}', save_top_k=1, monitor="val_dice_coeff_0.5")
    callbacks = [SetEpochCallback(train_sampler), checkpoint_callback]

    if cfg.do_wandb:
        wandb.init(
            name=cfg.exp_name,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            mode=cfg.wandb_mode,
        )
        wandb_logger = WandbLogger()
        logger.append(wandb_logger)

    trainer = pl.Trainer(
        accelerator="gpu" if str(cfg.device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=2,
    )

    trainer.fit(model, train_loader, val_loader)
