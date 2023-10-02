"""
Util functions module for data analysis.

Contains:
    - compute_rib_data
    > analysis: fractures
        - fracture_label_analysis
    > visualization
        - imshow
        - imshow_mutiple
        - scatter3d
        - create_gif
    > array value operations
        - minmax
        - histogram_equalization
        - equalize_per_slice
    > analysis: values
        - analysis_cum_scans
"""

import os
from collections import defaultdict, Counter

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_theme()
import imageio


LABEL_CODE = {
    0: 'background',
    1: 'displaced',
    2: 'non-displaced',
    3: 'buckle',
    4: 'segmental',
    -1: 'undefined'
}



def compute_rib_data(info_paths: list):
    rib_data = defaultdict(list)

    for info_file in info_paths:
        for index, row in pd.read_csv(info_file).iterrows():
            public_id, label_id, label_code = row
            rib_data[public_id].append(label_code)

    return rib_data


# slice geometric operations

def _get_bbox(scan):
    """get the 3d bounding box of the volume. scan is binary boolean array with coords (z, x, y)"""
    z, x, y = np.where(scan)
    assert len(z) > 0, 'scan is empty'
    min_x, max_x, min_y, max_y, min_z, max_z = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
    return min_x, max_x, min_y, max_y, min_z, max_z

def _get_max_xy_box(scan):
    """gets the largest window size that could contain the area of the facture in each slice. scan is binary boolean array with coords (z, x, y)"""
    xs = np.any(scan, axis=2, keepdims=True)  # any in y direction
    ys = np.any(scan, axis=1, keepdims=True)  # any in x direction
    size_x = np.sum(xs, axis=1, keepdims=True)  # sum in x direction
    size_y = np.sum(ys, axis=2, keepdims=True)  # sum in y direction
    size_x = np.max(size_x)  # largest x size
    size_y = np.max(size_y)  # largest y size
    return size_x, size_y

def _get_volume(scan):
    """volume of the volume defined by the binary boolean array scan with coords (z, x, y)"""
    return np.sum(scan)

def _get_xy_areas(scan):
    """gets minimum and maximum 2d areas through z-planes. scan is binary boolean array with coords (z, x, y)"""
    areas = np.sum(scan, axis=(1, 2))  # sum in x and y direction
    return np.min(areas), np.max(areas)

def _center_of_mass(scan) -> tuple:
    """cneter of mass of the volume defined by the binary boolean array scan with coords (z, x, y)"""
    z, x, y = np.where(scan)
    assert len(z) > 0, 'scan is empty'
    com_z = np.mean(z)
    com_x = np.mean(x)
    com_y = np.mean(y)
    return com_z, com_x, com_y


# analysis: fractures

def fracture_label_analysis(labels_path, rib_data):
    """
    Processes train labels and produces 2 dataframes with the following information:

    df_scan:
        public_id: id of the scan (e.g. 'RibFrac100')
        size_x: size of the scan in x-direction (height)
        size_y: size of the scan in y-direction (width)
        size_z: size of the scan in z-direction (depth)

    df_frac:
        public_id: id of the scan (e.g. 'RibFrac100')
        frac_idx: index of the fracture within the scan (natural number to count each fracture in each scan)
        frac_code: code of the fracture (integer code to identify the fracture type)
        frac_code_name: name of the fracture (string name of the fracture type, human readable frac_code)
        min_x: minimum x-value of the 3d bounding box the encloses the fracture (absolute values in pixels)
        max_x: maximum x-value of the 3d bounding box the encloses the fracture (absolute values in pixels)
        min_y: minimum y-value of the 3d bounding box the encloses the fracture (absolute values in pixels)
        max_y: maximum y-value of the 3d bounding box the encloses the fracture (absolute values in pixels)
        min_z: minimum z-value of the 3d bounding box the encloses the fracture (absolute values in pixels)
        max_z: maximum z-value of the 3d bounding box the encloses the fracture (absolute values in pixels)
        volume: volume of the fracture (number of voxels)
        com_x: x-value of the center of mass (absolute values in pixels, assuming uniform density)
        com_y: y-value of the center of mass (absolute values in pixels, assuming uniform density)
        com_z: z-value of the center of mass (absolute values in pixels, assuming uniform density)
        max2dsize_x: largest 2d bounding box in xy-plane (max window size that could fit the fracture in every slice)
        max2dsize_y: largest 2d bounding box in xy-plane (max window size that could fit the fracture in every slice)
        min2darea: minimum area in xy-plane (area of the smallest fracture slice)
        max2darea: maximum area in xy-plane (area of the largest fracture slice)
        size_x: size of the 3d bounding box in x-direction (absolute values in pixels)
        size_y: size of the 3d bounding box in y-direction (absolute values in pixels)
        size_z: size of the 3d bounding box in z-direction (absolute values in pixels)
        loc_x: location of the center of the 3d bounding box in x-direction (absolute values in pixels)
        loc_y: location of the center of the 3d bounding box in y-direction (absolute values in pixels)
        loc_z: location of the center of the 3d bounding box in z-direction (absolute values in pixels)
        rel_min_x: minimum x-value of the 3d bounding box the encloses the fracture (relative size w.r.t scan size)
        rel_max_x: maximum x-value of the 3d bounding box the encloses the fracture (relative size w.r.t scan size)
        rel_min_y: minimum y-value of the 3d bounding box the encloses the fracture (relative size w.r.t scan size)
        rel_max_y: maximum y-value of the 3d bounding box the encloses the fracture (relative size w.r.t scan size)
        rel_min_z: minimum z-value of the 3d bounding box the encloses the fracture (relative size w.r.t scan size)
        rel_max_z: maximum z-value of the 3d bounding box the encloses the fracture (relative size w.r.t scan size)
        rel_com_x: x-value of the center of mass (relative size w.r.t scan size, assuming uniform density)
        rel_com_y: y-value of the center of mass (relative size w.r.t scan size, assuming uniform density)
        rel_com_z: z-value of the center of mass (relative size w.r.t scan size, assuming uniform density)
        rel_size_x: size of the 3d bounding box in x-direction (relative size w.r.t scan size)
        rel_size_y: size of the 3d bounding box in y-direction (relative size w.r.t scan size)
        rel_size_z: size of the 3d bounding box in z-direction (relative size w.r.t scan size)
        rel_loc_x: location of the center of the 3d bounding box in x-direction (relative size w.r.t scan size)
        rel_loc_y: location of the center of the 3d bounding box in y-direction (relative size w.r.t scan size)
        rel_loc_z: location of the center of the 3d bounding box in z-direction (relative size w.r.t scan size)
    """
    scan_data = []
    frac_data = []

    for filename in tqdm(os.listdir(labels_path)[2:], desc='analyzing train data'):

        if not (filename.endswith("label.nii.gz") or filename.endswith("label.nii")):
            print('WARNING: Directory structure is not as expected. Ignoring file', filename)
            continue

        filepath = os.path.join(labels_path, filename)
        label_scan = nib.load(filepath).get_fdata().T.astype(int)
        
        public_id = filename.split('-')[0]
        scan_data.append([
            public_id,
            label_scan.shape[1],
            label_scan.shape[2],
            label_scan.shape[0],
        ])
        labels_per_slice = defaultdict(list)
        for slice_idx, slice in enumerate(label_scan):
            unique_label_ids = np.unique(slice).tolist()
            labels_per_slice[slice_idx] = unique_label_ids
            
        for frac_idx, frac_code in enumerate(rib_data[public_id]):

            if frac_code == 0:  # ignore background
                continue

            idxs = []
            for slice_idx, unique_label_ids in labels_per_slice.items():
                if frac_idx in unique_label_ids:
                    idxs.append(slice_idx)
            idxs = sorted(idxs)

            ## compute values

            reduced_scan = label_scan[idxs[0]:idxs[-1]+1]  # ease computation with only relevant slices
            min_x, max_x, min_y, max_y, min_z, max_z = _get_bbox(reduced_scan == frac_idx)
            volume = _get_volume(reduced_scan == frac_idx)
            max2dsize_x, max2dsize_y = _get_max_xy_box(reduced_scan == frac_idx)
            min2darea, max2darea = _get_xy_areas(reduced_scan == frac_idx)

            # compensate for reduced scan offset
            min_z += idxs[0]
            max_z += idxs[0]
            assert min_z == idxs[0], f'{min_z} != {idxs[0]}'
            assert max_z == idxs[-1], f'{max_z} != {idxs[-1]}'

            # again reduce scan to only relevant volume
            reduced_scan = label_scan[idxs[0]:idxs[-1]+1, min_x:max_x+1, min_y:max_y+1]
            com_z, com_x, com_y = _center_of_mass(reduced_scan == frac_idx)  # com relative to chunk

            # compensate for reduced scan offset
            com_x += min_x
            com_y += min_y
            com_z += min_z

            ## store values
            
            frac_data.append([
                public_id,
                frac_idx,
                frac_code,
                LABEL_CODE[frac_code],
                # bounding box limits (absolute pixel values)
                min_x,
                max_x,
                min_y,
                max_y,
                min_z,
                max_z,
                # volume of fracture
                volume,
                # center of mass of chunk (absolute pixel values)
                com_x,
                com_y,
                com_z,
                # largest 2d bounding box in xy-plane
                max2dsize_x,
                max2dsize_y,
                # min and max area in xy-plane
                min2darea,
                max2darea,
            ])

    df_scan = pd.DataFrame(
        scan_data,
        columns=[
            "public_id",
            "size_x",
            "size_y",
            "size_z",
        ],
    )

    df_frac = pd.DataFrame(
        frac_data,
        columns=[
            "public_id",
            "frac_idx",
            "frac_code",
            "frac_code_name",
            # bounding box limits (absolute pixel values)
            "min_x",
            "max_x",
            "min_y",
            "max_y",
            "min_z",
            "max_z",
            # volume of fracture
            "volume",
            # center of mass of chunk (absolute pixel values)
            "com_x",
            "com_y",
            "com_z",
            # largest 2d bounding box in xy-plane
            "max2dsize_x",
            "max2dsize_y",
            # min and max area in xy-plane
            "min2darea",
            "max2darea",
        ],
    )
    
    for axis in ['x', 'y', 'z']:
        # size of bounding box
        df_frac['size_' + axis] = df_frac['max_' + axis] - df_frac['min_' + axis]
        # center of bounding box
        df_frac['loc_' + axis] = (df_frac['max_' + axis] + df_frac['min_' + axis]) / 2

    scan_sizes = df_scan.set_index('public_id')

    # relative dimensions
    attrs = ['min', 'max', 'com', 'size', 'loc']
    for attr in attrs:
        for axis in ['x', 'y', 'z']:
            abs_attr = attr + '_' + axis
            rel_attr = 'rel_' + abs_attr
            scan_size = scan_sizes.loc[df_frac['public_id'], 'size_' + axis].reset_index(drop=True)
            df_frac[rel_attr] = df_frac[abs_attr] / scan_size

    print('Done!')

    return df_scan, df_frac


# visualization

def imshow(img, title=None, range=None):
    sns.reset_orig()
    vmin, vmax = range if range is not None else (img.min(), img.max())
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.show()
    sns.set_theme()

def imshow_multiple(imgs: list, titles: list=None, suptitle: list=None, range=None):
    sns.reset_orig()
    n_images = len(imgs)
    fig, axes = plt.subplots(1, len(imgs), figsize=(5*n_images, 5))
    for i, im in enumerate(imgs):
        vmin, vmax = range if range is not None else (im.min(), im.max())
        axes[i].imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
        if titles:
            axes[i].set_title(titles[i])
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
    sns.set_theme()


def scatter3d(x, y, z, hue, title=None, view_angle=(30, 20)):

    # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    # default seaborn colormap
    colors = {
        'buckle':'tab:blue',
        'displaced':'tab:orange',
        'non-displaced':'tab:green',
        'undefined':'tab:red',
        'segmental':'tab:purple',
    }
    colors = [colors[h] for h in hue]

    # plot
    sc = ax.scatter(x, y, z, s=20, label=hue, color=colors, marker='o', alpha=.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # set view angle
    ax.view_init(*view_angle)

    # legend
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

    if title:
        plt.title(title)
    plt.show()


def create_gif(scan, save_path):
    """create animated gif through z-axis from array with shape (z, x, y). array values must be in [0, 1]"""
    images = []
    for i in range(scan.shape[0]):
        slice = scan[i]
        slice = np.uint8(slice * 255)
        images.append(slice)
    imageio.mimsave(save_path, images, duration=0.1)
    print('saved gif at', save_path)


# array value operations

def minmax(image, range=None):
    """normalize image using minmax normalization"""
    if range is not None:
        min_, max_ = range
    else:
        min_ = np.min(image)
        max_ = np.max(image)
    return (image - min_) / (max_ - min_)

def histogram_equalization(image):
    """
    equalize image using histogram equalization.
    input image is a float array with values in [0, 1].
    output is in range [0, 1].
    src: https://saturncloud.io/blog/mastering-histogram-equalization-in-python-without-numpy-or-plotting/
    """
    # convert to uint8
    image = (image * 255).astype(np.uint8)
    # calculate histogram
    histogram = Counter(image.flatten())
    # calculate cdf
    cdf = dict()
    cum_sum = 0
    for intensity, freq in sorted(histogram.items()):
        cum_sum += freq
        cdf[intensity] = cum_sum
    # normalize cdf
    cdf_min = min(cdf.values())
    normalized_cdf = {k: ((v-cdf_min)/(image.size-1))*255 for k, v in cdf.items()}
    # equalize image
    if len(image.shape) == 2:
        equalized_image = [[normalized_cdf[pixel] for pixel in row] for row in image]
    elif len(image.shape) == 3:
        equalized_image = [[[normalized_cdf[pixel] for pixel in row] for row in slice_] for slice_ in image]
    # convert back to float
    equalized_image = np.asarray(equalized_image) / 255.
    return equalized_image

def equalize_per_slice(scan):
    """equalizes each slice independently"""
    equalized_scan = np.zeros(scan.shape)
    for i in range(scan.shape[0]):
        equalized_scan[i] = histogram_equalization(scan[i])
    return equalized_scan


# analysis: pixel values

def analysis_pixel_values(images_path, fn_preprocess):
    """
    Accumulates all scans in the (train) images folder after preprocessing them with fn_preprocess

    Args:
        images_path: path to the (train) images folder
        fn_preprocess: function that takes a scan and returns the scan preprocessed. It must return a binary boolean array with coords (z, x, y)
    """
    max_num_slices = 721  # hardcoded from df_scan.describe()>size_z>max

    cum_scans = np.zeros((max_num_slices, 512, 512))

    for filename in tqdm(os.listdir(images_path)[2:], desc='overlaping slices'):

        if not (filename.endswith("image.nii.gz") or filename.endswith("image.nii")):
            print('WARNING: Directory structure is not as expected. Ignoring file', filename)
            continue

        # read scan
        filepath = os.path.join(images_path, filename)
        raw_scan = nib.load(filepath).get_fdata().T.astype(float)

        # threshold scan
        scan = fn_preprocess(raw_scan)  # (z, x, y)

        # accumulate slices
        cum_scans[:scan.shape[0]] += scan

    print('Done!')

    return cum_scans
