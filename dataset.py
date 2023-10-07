import ast
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from tqdm import tqdm


class RibFracDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        partition: str,
        context_size: int,
        patch_original_size: int,
        patch_final_size: int,
        proportion_fracture_in_patch: float,
        cutoff_height: int,
        clip_min_val: int,
        clip_max_val: int,
        test_stride: int,
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
        self.cutoff_height = cutoff_height
        self.clip_min_val = clip_min_val
        self.clip_max_val = clip_max_val
        self.test_stride = test_stride
        self.debug = debug

        # Compute a DataFrame of all available slices
        self.data_info_path = os.path.join(root_dir, f"{partition}_data_info.csv")
        if not force_data_info and os.path.exists(self.data_info_path):
            self.df = pd.read_csv(self.data_info_path)
        else:
            self.df = self.create_data_info_csv()

        # Clean the DataFrame and prepare for sampling
        if partition == "train":
            self.drop_slices_without_context()
            self.repeat_slices_with_fracture()
            self.add_df_index()
        else:
            self.img_size, self.num_patches = self.compute_img_size_and_num_patches()
            self.create_local_pred_masks()
            self.drop_slices_without_context()

        # Set up transforms
        self.normalize = transforms.Normalize(mean=0.5, std=0.5)

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
        # If training, create a random patch from the image and mask slices
        if self.partition == "train":
            row = self.df.iloc[idx]
            # Load image and mask slices
            filename = os.path.join(self.root_dir, row["img_filename"])
            img = self.load_img(filename, row["slice_idx"])
            mask = self.load_mask(filename.replace("image", "label"), row["slice_idx"])
            is_fracture_slice = row["is_fracture_slice"]
            # Pad mask to allow for patches on the edge
            p = self.patch_original_size // 2
            mask = torch.nn.functional.pad(mask, (p, p, p, p), mode="constant", value=0)
            # Create random patch
            img_patch, mask_patch, random_coord = self.create_train_patch(
                img, mask, is_fracture_slice
            )
            # Transform image patch and mask patch together
            patch = self.transform(torch.cat([img_patch, mask_patch], dim=0))
            # Split image patch and mask patch
            img_patch, mask_patch = patch[:-1], patch[-1:]

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

        # If validation or test, create patches from a predetermined coordinate
        else:
            # Divide index into row and patch indices
            row_idx = int(idx // self.num_patches)
            patch_idx = int(idx - (row_idx * self.num_patches))
            # Compute patch coordinates
            row = self.df.iloc[row_idx]
            p = self.patch_original_size // 2
            num_patches_sqrt = int(np.sqrt(self.num_patches))
            ix, iy = np.unravel_index(patch_idx, (num_patches_sqrt, num_patches_sqrt))
            ix = (ix * self.test_stride) + p
            iy = (iy * self.test_stride) + p
            coord = (ix, iy)
            # Load image slice
            filename = os.path.join(self.root_dir, row["img_filename"])
            img = self.load_img(filename, row["slice_idx"])
            # Create patch
            img_patch = crop_patch(img, coord, self.patch_original_size)
            # Normalize and transform image patch
            img_patch = self.normalize(img_patch)
            img_patch = self.transform(img_patch)

            npy_filename = os.path.join(
                self.root_dir,
                f"{self.partition}-pred-masks",
                os.path.basename(filename)
                .replace("image", "pred_mask")
                .replace(".nii", ".npy")
                .replace(".gz", ""),
            )

            if self.debug:
                return (
                    img_patch,
                    coord,
                    npy_filename,
                    row["slice_idx"],
                    self.patch_original_size,
                    img,
                )

            return (
                img_patch,
                coord,
                npy_filename,
                row["slice_idx"],
                self.patch_original_size,
            )

    def load_img(self, img_filename, slice_idx):
        # Load whole image slice
        proxy_img = nib.load(img_filename)
        img = torch.from_numpy(
            proxy_img.dataobj[
                ...,
                slice_idx - self.context_size : slice_idx + self.context_size + 1,
            ]
            .copy()
            .T
        ).float()
        img = self.preprocess(img)
        return img

    def load_mask(self, mask_filename, slice_idx):
        # Load whole mask slice
        proxy_mask = nib.load(mask_filename)
        mask = (
            torch.from_numpy(proxy_mask.dataobj[..., slice_idx].copy().T != 0)
            .float()
            .unsqueeze(0)
        )
        return mask

    def preprocess(self, img):
        """Preprocess image slice."""

        # Clip values
        img = img.clip(self.clip_min_val, self.clip_max_val)

        # Rescale values (minmax normalization)
        img = (img - self.clip_min_val) / (self.clip_max_val - self.clip_min_val)

        # Pad image to allow for patches on the edge
        p = self.patch_original_size // 2
        img = torch.nn.functional.pad(img, (p, p, p, p), mode="constant", value=0)

        return img

    def create_train_patch(self, img, mask, is_fracture_slice):
        """Create a training patch from the image and mask slices."""

        # Get middle slice index
        middle = self.context_size
        # Get all bone pixel locations
        coords = torch.stack(torch.where(img[middle] > 0.05), dim=1)
        coords = coords[coords[:, 0] < self.cutoff_height]

        offset_range = int(0.2 * self.patch_original_size)
        random_offset = np.random.randint(-offset_range, offset_range, (2,))

        if is_fracture_slice:
            # Look for patch with sufficient fracture pixels
            for random_coord in np.random.permutation(coords):
                random_coord_off = random_coord + random_offset
                if random_coord_off[0] >= (img.shape[-1] - self.patch_original_size // 2):
                    continue
                if random_coord_off[1] >= (img.shape[-1] - self.patch_original_size // 2):
                    continue
                if random_coord_off[0] <= (self.patch_original_size // 2):
                    continue
                if random_coord_off[1] <= (self.patch_original_size // 2):
                    continue
                img_patch = crop_patch(img, random_coord_off, self.patch_original_size)
                mask_patch = crop_patch(mask, random_coord_off, self.patch_original_size)
                if (
                    torch.sum(mask_patch) / mask_patch.numel()
                    > self.proportion_fracture_in_patch
                ):
                    break
                img_patch = crop_patch(img, random_coord, self.patch_original_size)
                mask_patch = crop_patch(mask, random_coord, self.patch_original_size)
                random_coord_off = random_coord
        else:
            # Look for patch with no fracture pixels
            for random_coord in np.random.permutation(coords):
                random_coord_off = random_coord + random_offset
                if random_coord_off[0] >= (img.shape[-1] - self.patch_original_size // 2):
                    continue
                if random_coord_off[1] >= (img.shape[-1] - self.patch_original_size // 2):
                    continue
                if random_coord_off[0] <= (self.patch_original_size // 2):
                    continue
                if random_coord_off[1] <= (self.patch_original_size // 2):
                    continue
                img_patch = crop_patch(img, random_coord_off, self.patch_original_size)
                mask_patch = crop_patch(mask, random_coord_off, self.patch_original_size)
                if torch.sum(mask_patch) == 0:
                    break
                img_patch = crop_patch(img, random_coord, self.patch_original_size)
                mask_patch = crop_patch(mask, random_coord, self.patch_original_size)
                random_coord_off = random_coord

        # Normalize image patch
        img_patch = self.normalize(img_patch)
        return img_patch, mask_patch, random_coord_off

    def drop_slices_without_context(self):
        """Drop slices without sufficient context from the DataFrame."""

        # Compute the number of slices in each scan
        sizes = self.df.sort_values(by=["img_filename", "slice_idx"])
        sizes = sizes.drop_duplicates(subset=["img_filename"], keep="last")
        sizes = sizes[["img_filename", "slice_idx"]].values

        # For each scan...
        for img_filename, size in sizes:
            # Drop slices without sufficient from above
            self.df.drop(
                self.df[
                    (self.df.img_filename == img_filename)
                    & (self.df.slice_idx > (size - self.context_size))
                ].index,
                inplace=True,
            )
            # Drop slices without sufficient from below
            self.df.drop(
                self.df[
                    (self.df.img_filename == img_filename)
                    & (self.df.slice_idx < self.context_size)
                ].index,
                inplace=True,
            )

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

    def create_local_pred_masks(self):
        sizes = self.df.sort_values(by=["img_filename", "slice_idx"])
        sizes = sizes.drop_duplicates(subset=["img_filename"], keep="last")
        sizes = sizes[["img_filename", "slice_idx"]].values

        pred_dir = os.path.join(self.root_dir, f"{self.partition}-pred-masks")
        os.mkdir(pred_dir) if not os.path.exists(pred_dir) else None

        for img_filename, size in tqdm(sizes, desc="Creating local prediction masks"):
            filename = (
                os.path.basename(img_filename)
                .replace("image", "pred_mask")
                .replace(".nii", ".npy")
                .replace(".gz", "")
            )
            s = self.img_size + 2 * (self.patch_original_size // 2)
            pred_mask = np.zeros((2, size + 1, s, s)).astype(np.float16)
            np.save(os.path.join(pred_dir, filename), pred_mask)

    def compute_img_size_and_num_patches(self):
        filename = os.path.join(self.root_dir, self.df.iloc[0]["img_filename"])
        img_size = nib.load(filename).get_fdata().shape[0]
        num_patches = np.floor((img_size / self.test_stride) + 1) ** 2
        return img_size, num_patches

    def create_data_info_csv(self):
        """Create a DataFrame with all available slices for this specific partition."""

        def process_folder(folder_name):
            data = []
            for filename in tqdm(
                os.listdir(os.path.join(self.root_dir, folder_name)), desc=folder_name
            ):
                f = os.path.join(self.root_dir, folder_name, filename)
                if filename.endswith(".nii") or filename.endswith(".nii.gz"):
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

    def get_train_sampler(self, seed):
        return BalancedFractureSampler(self.df, seed)

    def get_test_sampler(self):
        return TestSampler(self.df, self.num_patches)


class BalancedFractureSampler(Sampler):
    """Samples one fracture slice for each non-fracture slice."""

    def __init__(self, data_info: pd.DataFrame, seed: int):
        self.data_info = data_info
        self.seed = seed
        self.epoch = 0
        self.size = int(self.data_info[self.data_info.is_fracture_slice == True].shape[0] * 2)

    def __len__(self):
        return self.size

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


class TestSampler(Sampler):
    def __init__(self, data_info: pd.DataFrame, num_patches: int):
        self.data_info = data_info
        self.num_patches = num_patches

    def __len__(self):
        return int(self.data_info.shape[0] * self.num_patches)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        idx_list = []
        patch_idx = np.arange(self.num_patches)
        for i in range(len(self.data_info)):
            idx_list += (patch_idx + i * self.num_patches).astype(int).tolist()
        return iter(idx_list)


def crop_patch(image, center_coord, patch_size):
    patch = image[
        :,
        center_coord[0] - patch_size // 2 : center_coord[0] + patch_size // 2,
        center_coord[1] - patch_size // 2 : center_coord[1] + patch_size // 2,
    ]
    return patch
