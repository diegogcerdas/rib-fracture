import dataclasses
import json
from dataclasses import dataclass
import torch

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl

from dataset import BalancedFractureSampler


@dataclass
class ConfigTrain:
    data_root: str
    ckpt_root: str
    resume_ckpt: str
    patch_original_size: int
    patch_final_size: int
    proportion_fracture_in_patch: float
    cutoff_height: int
    clip_min_val: int
    clip_max_val: int
    data_mean: float
    data_std: float
    test_stride: int  # unused
    force_data_info: bool
    download_data: bool
    use_focal_loss: bool
    use_msssim_loss: bool
    use_model: str
    context_size: int
    use_positional_encoding: bool
    seed: int
    learning_rate: float
    weight_decay: float
    batch_size_train: int
    batch_size_test: int  # for val
    num_workers: int
    max_epochs: int
    log_every_step: bool
    device: str
    do_wandb: bool
    wandb_key: str
    exp_name: str
    wandb_project: str
    wandb_entity: str
    wandb_mode: str


class ConfigTest:
    # data
    ckpt: str
    data_root: str
    download_data: bool

    # run
    test_stride: int
    batch_size_test: int
    num_workers: int
    device: str

    # wandb
    do_wandb: bool
    wandb_key: str


def config_from_args(args, mode="train"):
    """creates config from args"""
    if mode == "train":
        class_name = ConfigTrain
    elif mode == "test":
        class_name = ConfigTest
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
    return class_name(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(class_name)}
    )


def save_config(config, path):
    """saves config to json"""
    dic = dataclasses.asdict(config)
    dic['device'] = str(dic['device'])
    dic['wandb_key'] = None  # security
    with open(path, "w") as f:
        json.dump(dic, f, indent=4)


def load_config(path, mode="test"):
    """loads config from json. args overwrites config values"""
    with open(path, "r") as f:
        lines = json.load(f)
    if mode == "train":
        class_name = ConfigTrain
    elif mode == "test":
        class_name = ConfigTest
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
    lines = {f.name: lines[f.name] for f in dataclasses.fields(class_name)}
    lines['device'] = torch.device(lines['device'])  # TODO: check if this works
    return class_name(**lines)


class SetEpochCallback(pl.Callback):
    def __init__(self, sampler: BalancedFractureSampler):
        super().__init__()
        self.sampler = sampler

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self.sampler.set_epoch(trainer.current_epoch)


class VizCallback(pl.Callback):
    def __init__(self, context_size: int, size: int = 8):
        super().__init__()
        self.context_size = context_size
        self.size = size

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if batch_idx == 0:
            x, y = batch
            y_hat = outputs
            batch_size = x.shape[0]

            choice = np.random.choice(range(batch_size), size=self.size, replace=True)
            x = x[:, self.context_size].detach().cpu().numpy().squeeze()[choice]
            y = y.detach().cpu().numpy().squeeze()[choice]
            y_hat = y_hat.detach().cpu().numpy().squeeze()[choice]

            f, axes = plt.subplots(self.size, 3, figsize=(12, self.size * 4))
            for i in range(self.size):
                axes[i, 0].imshow(x[i], cmap="gray")
                axes[i, 1].imshow(y[i], cmap="gray")
                axes[i, 2].imshow(y_hat[i], cmap="gray")
                axes[i, 0].axis("off")
                axes[i, 1].axis("off")
                axes[i, 2].axis("off")
            axes[0, 0].set_title("Input")
            axes[0, 1].set_title("Ground Truth")
            axes[0, 2].set_title("Prediction")
            plt.tight_layout()
            if isinstance(trainer.logger, pl.loggers.wandb.WandbLogger):
                trainer.logger.log_image(key="Visualization", images=[f])
            plt.close()
