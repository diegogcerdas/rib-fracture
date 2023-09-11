import dataclasses
from dataclasses import dataclass


@dataclass
class ConfigTrain:
    seed: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    num_workers: int
    max_epochs: int
    device: str
    wandb_name: str
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


# TODO: Callbacks
