import ast
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from tqdm import tqdm


class RibFracDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        partition: str,
        context_size: int = 0,
        patch_original_size: int = 24,
        patch_final_size: int = 256,
        proportion_fracture_in_patch: float = 0.25,
        level: int = 400,
        window: int = 1800,
        threshold: float = 0.35,
        test_stride: int = 1,
        force_data_info: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        assert partition in ["train", "val", "test"]
        self.root_dir = root_dir
        self.partition = partition
        self.context_size = context_size
        self.patch_original_size = patch_original_size
        self.proportion_fracture_in_patch = proportion_fracture_in_patch
        self.level = level
        self.window = window
        self.threshold = threshold
        self.test_stride = test_stride
        self.debug = debug

        # Compute a DataFrame of all available slices
        self.data_info_path = os.path.join(root_dir, f"{partition}_data_info.csv")
        if not force_data_info and os.path.exists(self.data_info_path):
            self.df = pd.read_csv(self.data_info_path)
        else:
            self.df = self.create_data_info_csv()
        self.drop_slices_without_context()

        if partition == "train":
            self.drop_slices_without_ribs()
            self.repeat_slices_with_fracture()
            self.add_df_index()

        if partition == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(patch_final_size, antialias=True),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(patch_final_size, antialias=True)]
            )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image slice
        proxy_img = nib.load(os.path.join(self.root_dir, row["img_filename"]))
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
        img_channels = img.shape[0]
        img = self.preprocess(img)
        p = self.patch_original_size // 2
        img = torch.nn.functional.pad(img, (p, p, p, p), mode="constant", value=0)

        # Load mask slice
        proxy_mask = nib.load(
            os.path.join(self.root_dir, row["img_filename"]).replace("image", "label")
        )
        mask = (
            torch.from_numpy(proxy_mask.dataobj[..., row["slice_idx"]].copy().T != 0)
            .float()
            .unsqueeze(0)
        )

        if self.partition == "train":
            is_fracture_slice = row["is_fracture_slice"]
            mask = torch.nn.functional.pad(mask, (p, p, p, p), mode="constant", value=0)
            img_patch, mask_patch, random_coord = self.create_patch(
                img, mask, is_fracture_slice
            )
            patch = self.transform(torch.cat([img_patch, mask_patch], dim=0))
            img_patch, mask_patch = patch[:img_channels], patch[-1:]

            if self.debug:
                return (
                    img_patch,
                    mask_patch,
                    random_coord,
                    img,
                    mask,
                    row["img_filename"],
                    row["slice_idx"],
                    is_fracture_slice,
                )

            return img_patch, mask_patch

        else:
            patches = self.create_patch(img, mask)
            patches = self.transform(patches)
            return patches, mask, self.patch_original_size, self.test_stride

    # TODO: Remove backplate of CT scan
    def create_patch(self, img, mask, is_fracture_slice=None):
        """Create a patch from the image and mask slices."""

        def crop(center_coord):
            img_patch = img[
                :,
                center_coord[0]
                - self.patch_original_size // 2 : center_coord[0]
                + self.patch_original_size // 2,
                center_coord[1]
                - self.patch_original_size // 2 : center_coord[1]
                + self.patch_original_size // 2,
            ]
            mask_patch = mask[
                :,
                center_coord[0]
                - self.patch_original_size // 2 : center_coord[0]
                + self.patch_original_size // 2,
                center_coord[1]
                - self.patch_original_size // 2 : center_coord[1]
                + self.patch_original_size // 2,
            ]
            return img_patch, mask_patch

        if self.partition == "train":
            # Get middle slice index
            middle = self.context_size
            # Get all bone pixel locations
            coords = torch.stack(torch.where(img[middle] > 0), dim=1)

            if is_fracture_slice:
                # Look for patch with sufficient fracture pixels
                for random_coord in np.random.permutation(coords):
                    img_patch, mask_patch = crop(random_coord)
                    if (
                        torch.sum(mask_patch) / mask_patch.numel()
                        > self.proportion_fracture_in_patch
                    ):
                        break
            else:
                # Look for patch with no fracture pixels
                for random_coord in np.random.permutation(coords):
                    img_patch, mask_patch = crop(random_coord)
                    if torch.sum(mask_patch) == 0:
                        break

            return img_patch, mask_patch, random_coord

        else:
            chs = img.shape[0]
            ks = self.patch_original_size + (
                1 if self.patch_original_size % 2 == 0 else 0
            )
            patches = F.unfold(img, kernel_size=ks, stride=self.test_stride)
            patches = patches.reshape(chs, ks, ks, -1)
            patches = patches.permute(3, 0, 1, 2)
            return patches

    # TODO: Integrate with drop_slices_without_ribs
    def drop_slices_without_context(self):
        """Drop slices without sufficient context from the DataFrame."""
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

    # TODO: Remove slices without ribs for better data quality
    def drop_slices_without_ribs(self):
        """Drop slices without ribs from the DataFrame."""
        None

    def repeat_slices_with_fracture(self):
        """Repeat slices with fractures in the DataFrame to aid the BalancedFractureSampler."""
        self.df["is_fracture_slice"] = False
        slices_with_fracture = self.df[
            self.df.apply(
                lambda x: sum(ast.literal_eval(str(x["labels"]))) != 0, axis=1
            )
        ].copy()
        slices_with_fracture["is_fracture_slice"] = True
        self.df = pd.concat([self.df, slices_with_fracture])

    def add_df_index(self):
        """Add a column with the index of the slice in the DataFrame."""
        self.df = self.df.reset_index(drop=True)
        self.df["df_index"] = np.arange(len(self.df))

    def preprocess(self, img):
        """Preprocess image slice."""
        max_val = self.level + self.window / 2
        min_val = self.level - self.window / 2
        img = img.clip(min_val, max_val)
        img = (img - img.min()) / (img.max() - img.min())
        img[img <= self.threshold] = 0
        return img

    def create_data_info_csv(self):
        """Create a DataFrame with all available slices for this specific partition."""

        def process_folder(folder_name):
            data = []
            for filename in tqdm(
                os.listdir(os.path.join(self.root_dir, folder_name)), desc=folder_name
            ):
                f = os.path.join(self.root_dir, folder_name, filename)
                if filename.endswith(".nii"):
                    scan = nib.load(f).get_fdata().T.astype(float)
                    label_f = f.replace("image", "label")
                    labels = nib.load(label_f).dataobj
                    for i, slice in enumerate(scan):
                        data.append(
                            [
                                os.path.join(folder_name, filename),
                                i,
                                np.unique(labels[..., i]).astype(int).tolist(),
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
                    "labels",
                    "min",
                    "max",
                    "mean",
                    "std",
                ],
            )
            return df

        print(f"Collecting {self.partition} dataset info. This may take a while...")
        df = process_folder(f"ribfrac-{self.partition}-images")
        df = df.sort_values(by=["img_filename", "slice_idx"])
        df.to_csv(self.data_info_path, index=False)
        print("Done!")
        return df

    def get_sampler(self, seed):
        return BalancedFractureSampler(self.df, seed)


class BalancedFractureSampler(Sampler):
    """Samples one fracture slice for each non-fracture slice."""

    def __init__(self, data_info: pd.DataFrame, seed: int):
        self.data_info = data_info
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.data_info.shape[0]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        idx_list = []
        g = torch.Generator()
        g.manual_seed(self.seed * 10000 + self.epoch)

        fracture_slice_idx = torch.tensor(
            self.data_info[self.data_info.is_fracture_slice == True].df_index.values
        )
        choice = torch.randperm(len(fracture_slice_idx), generator=g).tolist()
        fracture_slice_idx = fracture_slice_idx[choice].tolist()
        print(f"Fracture slices: {len(fracture_slice_idx)}")

        non_fracture_slice_idx = torch.tensor(
            self.data_info[self.data_info.is_fracture_slice == False].df_index.values
        )
        choice = torch.randperm(len(fracture_slice_idx), generator=g).tolist()
        non_fracture_slice_idx = non_fracture_slice_idx[choice].tolist()
        print(f"Non-fracture slices: {len(non_fracture_slice_idx)}")

        for i in range(len(fracture_slice_idx)):
            idx_list.append(fracture_slice_idx[i])
            idx_list.append(non_fracture_slice_idx[i])

        return iter(idx_list)
