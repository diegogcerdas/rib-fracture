import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def create_data_info_csv(data_root: str, csv_filename: str = "data_info.csv"):
    def process_folder(folder_name):
        data = []
        set_ = (
            "train"
            if "train" in folder_name
            else "val"
            if "val" in folder_name
            else "test"
        )
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
                            set_,
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
    for folder in ["ribfrac-train-images", "ribfrac-val-images", "ribfrac-test-images"]:
        dfs.append(process_folder(folder))
    df = pd.concat(dfs)
    df = df.sort_values(by=["set", "img_filename", "slice_idx"])
    df.to_csv(os.path.join(data_root, csv_filename), index=False)
    return df


class RibFracDataset(Dataset):
    def __init__(self, data_root: str, set: str, context_size: int = 0):
        super().__init__()
        assert set in ["train", "val", "test"]
        self.data_root = data_root
        self.set = set
        self.context_size = context_size

        data_info_path = os.path.join(data_root, "data_info.csv")
        if os.path.exists(data_info_path):
            self.df = pd.read_csv(data_info_path)
        else:
            self.df = create_data_info_csv(data_root)
        self.df = self.df[self.df["set"] == set]
        self.drop_slices_without_context()

        self.transform = transforms.Compose(
            [
                # TODO: Define transforms / preprocessing
                # Beware: flips must be applied on both image and mask
            ]
        )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        proxy_img = nib.load(os.path.join(self.data_root, row["img_filename"]))
        img = torch.from_numpy(
            proxy_img.dataobj[
                ...,
                row["slice_idx"]
                - self.context_size : row["slice_idx"]
                + self.context_size
                + 1,
            ]
            .copy()
            .T
        ).float()
        proxy_mask = nib.load(
            os.path.join(self.data_root, row["img_filename"]).replace("image", "label")
        )
        mask = (
            torch.from_numpy(proxy_mask.dataobj[..., row["slice_idx"]].copy().T != 0)
            .float()
            .unsqueeze(0)
        )
        return img, mask

    def drop_slices_without_context(self):
        sizes = self.df.sort_values(by=["img_filename", "slice_idx"])
        sizes = sizes.drop_duplicates(subset=["img_filename"], keep="last")
        sizes = sizes[["img_filename", "slice_idx"]].values

        for img_filename, size in sizes:
            self.df.drop(
                self.df[
                    (self.df.img_filename == img_filename)
                    & (self.df.slice_idx > (size - self.context_size))
                ].index,
                inplace=True,
            )
            self.df.drop(
                self.df[
                    (self.df.img_filename == img_filename)
                    & (self.df.slice_idx < self.context_size)
                ].index,
                inplace=True,
            )
