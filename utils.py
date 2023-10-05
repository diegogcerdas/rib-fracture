import dataclasses
from dataclasses import dataclass

import pytorch_lightning as pl

from dataset import BalancedFractureSampler


@dataclass
class ConfigTrain:
    data_root: str
    ckpt_root: str
    patch_original_size: int
    patch_final_size: int
    proportion_fracture_in_patch: float
    cutoff_height: int
    clip_min_val: int
    clip_max_val: int
    test_stride: int
    force_data_info: bool
    context_size: int
    seed: int
    learning_rate: float
    weight_decay: float
    batch_size_train: int
    batch_size_test: int
    num_workers: int
    max_epochs: int
    device: str
    do_wandb: bool
    exp_name: str
    wandb_project: str
    wandb_entity: str
    wandb_mode: str


# TODO: ConfigTest


def config_from_args(args, mode="train"):
    if mode == "train":
        class_name = ConfigTrain
    else:
        raise ValueError("Mode must be either 'train' or 'test'")
    return class_name(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(class_name)}
    )


class SetEpochCallback(pl.Callback):
    def __init__(self, sampler: BalancedFractureSampler):
        super().__init__()
        self.sampler = sampler

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self.sampler.set_epoch(trainer.current_epoch)
