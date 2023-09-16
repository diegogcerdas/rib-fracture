import os

import nibabel as nib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def create_data_info_csv(data_root: str, csv_filename: str = "data_info.csv"):
    def process_folder(folder_name):
        data = []
        for filename in tqdm(
            os.listdir(os.path.join(data_root, folder_name)), desc=folder_name
        ):
            f = os.path.join(data_root, folder_name, filename)
            if filename.endswith(".nii"):
                scan = nib.load(f).get_fdata().T.astype(float)
                label_f = f.replace("image", "label")
                labels = nib.load(label_f).get_fdata().T.astype(int)
                for i, slice in enumerate(scan):
                    data.append(
                        [
                            os.path.join(folder_name, filename),
                            i,
                            "train" if "train" in folder_name else "val",
                            np.unique(labels[i]).tolist(),
                            slice.min(),
                            slice.max(),
                            slice.mean(),
                            slice.std(),
                        ]
                    )
        df = pd.DataFrame(
            data,
            columns=[
                "img_filename",
                "slice_idx",
                "set",
                "labels",
                "min",
                "max",
                "mean",
                "std",
            ],
        )
        return df

    dfs = []
    for folder in ["ribfrac-val-images"]:  # TODO: add train folders
        dfs.append(process_folder(folder))
    df = pd.concat(dfs)
    df = df.sort_values(by=["set", "img_filename", "slice_idx"])
    df.to_csv(os.path.join(data_root, csv_filename), index=False)


class RibFracDataset(Dataset):
    def __init__(self, data_root: str, train: bool = True):
        super().__init__()
        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(data_root, "data_info.csv"))
        self.df = self.df[self.df["set"] == ("train" if train else "val")]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        proxy_img = nib.load(os.path.join(self.data_root, row["img_filename"]))
        return proxy_img.dataobj[..., row["slice_idx"]].copy()
