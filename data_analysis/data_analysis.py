"""
Util functions module for data analysis.

Contains:
    - LABEL_CODE
    - compute_rib_data
    > analysis: fractures
        - analysis_fracture_labels
    > visualization
        - imshow
        - imshow_mutiple
        - scatter3d
        - create_gif
    > file management
        - load_params_json
        - update_params_json
    > array value operations
        - minmax
        - histogram_equalization
        - equalize_per_slice
        - median_filter
    > analysis: pixel values
        - analysis_pixel_values
    > connected components
        - SQUARE_STRUCT
        - STAR_STRUCT
        - identify_components
        - largest_component
        - get_mask_geometry
        - get_component_stats_per_slice
        - get_component_stats_per_slice
    > analysis: connected components
        - analysis_conn_comp
    > compute-or-load
        - compute_or_load_fracture_analysis
        - compute_or_load_pixel_analysis
        - compute_or_load_eq
        - compute_or_load_conn_comp_analysis


"""

import json
import os
from collections import defaultdict, Counter
import random

import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from skimage import morphology
from skimage.transform import resize
from tqdm import tqdm


sns.set_theme()

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

def analysis_fracture_labels(labels_path, rib_data):
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

    for filename in tqdm(os.listdir(labels_path), desc='fracture labels analysis'):

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
    plt.colorbar()
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


# file managing

def load_params_json(json_path):
    """read params.json file from analysis_root"""
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as json_file:
        params = json.load(json_file)
    return params

def update_params_json(json_path, **params):
    """update params.json file from analysis_root with params"""
    params_dict = load_params_json(json_path)
    params_dict.update(params)
    if not os.path.exists(json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as json_file:
        json.dump(params_dict, json_file, indent=4)


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

def remove_noise(img, kernel_size=3):
    """remove salt-and-papper noise with median filter"""
    return ndimage.median_filter(img, size=kernel_size)


# analysis: pixel values

def analysis_pixel_values(images_path, fn_preprocess, align='top'):
    """
    Accumulates all scans in the (train) images folder after preprocessing them with fn_preprocess

    Args:
        images_path: path to the (train) images folder
        fn_preprocess: function that takes a scan and returns the scan preprocessed. It must return a binary boolean array with coords (z, x, y)
        align: how the scans with different z-sizes should be aligned. 'top'/'bottom'/'center'/'fit'. 'fit' stretches to relative height
    
    Returns:
        mean: mean value of all pixels in all scans
        std: standard deviation of all pixels in all scans
        cum_scans: cumulative sum of all scans
        avg_scan: average scan
    """
    max_size = 721  # hardcoded from df_scan.describe()>size_z>max
    fit_size = 512  # hardcoded to match x and y, value in between min=239 and max=721

    sum_values = 0  # for mean
    sum_values2 = 0  # squared, for std
    tot_elem = 0  # total number of elements

    if align == 'fit':
        cum_scans = np.zeros((fit_size, 512, 512))
        cum_scans_n = 0  # only number of scans needed
    else:
        cum_scans = np.zeros((max_size, 512, 512))
        cum_scans_n = np.zeros((max_size, 512, 512))

    for filename in tqdm(os.listdir(images_path), desc='pixel values analysis (align={})'.format(align)):

        if not (filename.endswith("image.nii.gz") or filename.endswith("image.nii")):
            print('WARNING: Directory structure is not as expected. Ignoring file', filename)
            continue

        # read scan
        filepath = os.path.join(images_path, filename)
        raw_scan = nib.load(filepath).get_fdata().T.astype(float)
        scan = fn_preprocess(raw_scan)  # (z, x, y)

        sum_values += np.sum(scan)
        sum_values2 += np.sum(scan**2)
        tot_elem += scan.size

        if align == 'top':
            cum_scans[:scan.shape[0]] += scan
            cum_scans_n[:scan.shape[0]] += 1
        elif align == 'bottom':
            cum_scans[-scan.shape[0]:] += scan
            cum_scans_n[-scan.shape[0]:] += 1
        elif align == 'center':
            cum_scans[(max_size-scan.shape[0])//2:(max_size+scan.shape[0])//2] += scan
            cum_scans_n[(max_size-scan.shape[0])//2:(max_size+scan.shape[0])//2] += 1
        elif align == 'fit':
            resized_scan = resize(scan, (fit_size, 512, 512), preserve_range=True)
            cum_scans += resized_scan
            cum_scans_n += 1

    # compute mean and std
    mean = sum_values / tot_elem
    std = np.sqrt(sum_values2 / tot_elem - mean**2)

    # compute cumulative scan
    avg_scan = cum_scans / cum_scans_n

    print('Done!')

    return mean, std, cum_scans, avg_scan


# connected components

SQUARE_STRUCT = np.ones((3, 3))
STAR_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

def identify_components(img, struct_elem=SQUARE_STRUCT):
    """
    img: 2d image
    elem: structuring element
    """
    # binarized_axial_image = np.where(img > 0.05, 1, 0)  # FIXME: hardcoded threshold (190)
    # multi_dilated = multi_dil(binarized_axial_image, 4)  # FIXME: hardcoded dilation (7)
    # area_closed = morphology.area_closing(multi_dilated, 10000)  # FIXME: hardcoded area (50000)
    # multi_eroded = multi_ero(area_closed, 1)  # FIXME: hardcoded erosion (7)
    # opened = morphology.opening(multi_eroded)

    im = np.where(img > 0.04, 1, 0)
    imshow(im, title='binarized')
    im = morphology.closing(im, np.ones((5,5)))
    imshow(im, title='closed')
    im = remove_noise(im, kernel_size=3)
    imshow(im, title='noise removed')
    for _ in range(5):
        im = morphology.dilation(im, np.ones((3,3)))
    imshow(im, title='dilated')
    im = morphology.area_closing(im, 50000)
    imshow(im, title='area closed')
    for _ in range(5):
        im = morphology.erosion(im, np.ones((3,3)))
    im = morphology.opening(im)
    imshow(im, title='opened')
    opened = im

    # Label the connected components in the segmented image
    labeled_image, num_labels = morphology.label(opened, connectivity=2, return_num=True)

    return labeled_image, num_labels

def largest_component(labeled_img, return_mask=False):
    areas = np.bincount(labeled_img.ravel())
    if len(areas) == 1:
        # no connected components other than background
        return (0, None) if return_mask else 0
    
    # largest object
    largest_object_label = np.argmax(areas[1:]) + 1  # +1 to account for background label 0

    if return_mask:
        mask = labeled_img == largest_object_label
        return largest_object_label, mask
    return largest_object_label

def _remove_largest_component(img):  # TODO: unused
    """
    img: 2d image in range [0,1]
    """
    _, mask = largest_component(img, return_mask=True)
    if mask is None:
        # no connected components other than background
        return img

    # TODO: dilate mask

    # remove largest object
    out_img = img.copy()
    out_img[mask] = 0

    return out_img

def get_mask_geometry(mask):
    """computes bounding box(min, max, center), center of mass(com) and area of a binary mask

    Args:
        mask: binary (boolean) mask

    Returns:
        min_x, max_x, min_y, max_y, ctr_x, ctr_y, com_x, com_y
    """
    x, y = np.where(mask.astype(bool))
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    ctr_x, ctr_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    com_x, com_y = np.mean(x), np.mean(y)
    area = np.sum(mask)
    return min_x, max_x, min_y, max_y, ctr_x, ctr_y, com_x, com_y, area


# analysis: connected components

def analysis_conn_comp(images_path, fn_preprocess, struct_elem=SQUARE_STRUCT, max_files=None, max_slices=None):

    data_conn_comp = []

    images_list = os.listdir(images_path)
    if max_files is not None and max_files < len(images_list):
        # select random files
        images_list = random.sample(images_list, max_files)

    for filename in tqdm(images_list, desc='analyze pixel values'):

        if not (filename.endswith("image.nii.gz") or filename.endswith("image.nii")):
            print('WARNING: Directory structure is not as expected. Ignoring file', filename)
            continue

        # read scan
        filepath = os.path.join(images_path, filename)
        raw_scan = nib.load(filepath).get_fdata().T.astype(float)
        scan = fn_preprocess(raw_scan)  # (z, x, y)

        public_id = filename.split('-')[0]

        slice_idxs = range(scan.shape[0])
        if max_slices is not None and max_slices < len(slice_idxs):
            # select random slices
            slice_idxs = random.sample(slice_idxs, max_slices)

        for slice_idx in slice_idxs:
            slice_ = scan[slice_idx]

            imshow(slice_, title=f'{public_id} - slice {slice_idx} - preprocessed')

            labeled_img, num_labels = identify_components(slice_, struct_elem=struct_elem)

            imshow(labeled_img, title=f'{public_id} - slice {slice_idx} - labeled components')

            if num_labels == 0:
                # no connected components other than background
                continue
            largest_component_label = largest_component(labeled_img)
            for lab in range(num_labels):
                mask = labeled_img == lab
                min_x, max_x, min_y, max_y, ctr_x, ctr_y, com_x, com_y, area = get_mask_geometry(mask)
                data_conn_comp.append(
                    [public_id, slice_idx, lab, lab == largest_component_label,
                     min_x, max_x, min_y, max_y,
                     ctr_x, ctr_y, com_x, com_y, area]
                )

    df_conn_comp = pd.DataFrame(data_conn_comp, columns=[
        'public_id', 'slice_idx', 'conn_comp_label', 'is_largest',
        'min_x', 'max_x', 'min_y', 'max_y',
        'ctr_x', 'ctr_y', 'com_x', 'com_y', 'area'
    ])

    print('Done!')

    return df_conn_comp


# compute-or-load

def compute_or_load_fracture_analysis(analysis_root, labels_path, rib_data):
    scan_path = os.path.join(analysis_root, 'scan.csv')
    frac_path = os.path.join(analysis_root, 'frac.csv')

    if os.path.exists(scan_path) and os.path.exists(frac_path):
        df_scan = pd.read_csv(scan_path)
        df_frac = pd.read_csv(frac_path)
        print('Loaded existing analysis')
    else:
        print('No existing analysis found, computing...')
        df_scan, df_frac = analysis_fracture_labels(labels_path, rib_data)
        os.makedirs(analysis_root, exist_ok=True)
        df_scan.to_csv(scan_path, index=False)
        df_frac.to_csv(frac_path, index=False)
        print('Saved files:',
              scan_path,
              frac_path,
              sep='\n\t')

    # fix: ignore background if present
    df_frac = df_frac[df_frac['frac_code'] != 0].reset_index(drop=True)

    return df_scan, df_frac


def compute_or_load_pixel_analysis(analysis_root, images_path, fn_preprocess, align='top'):  # TODO: tmp - ignores cum
    #cum_scans_path = os.path.join(analysis_root, f'cum_scan_align{align}.npy')
    avg_scan_path = os.path.join(analysis_root, f'avg_scan_align{align}.npy')
    params_path = os.path.join(analysis_root, 'params.json')

    if os.path.exists(avg_scan_path):
        #cum_scan = np.load(cum_scans_path)
        avg_scan = np.load(avg_scan_path)
        params = load_params_json(params_path)
        mean, std = params['mean'], params['std']
        print('Loaded existing analysis')
    else:
        print('No existing analysis found, computing...')
        mean, std, _, avg_scan = analysis_pixel_values(images_path, fn_preprocess, align=align)
        #np.save(cum_scans_path, cum_scan)
        np.save(avg_scan_path, avg_scan)
        update_params_json(params_path, mean=mean, std=std)
        print('Saved files:',
              params_path,
              #cum_scans_path,
              avg_scan_path,
              sep='\n\t')

    return mean, std, None, avg_scan


def compute_or_load_eq(analysis_root, cum_scan, avg_scan, *, align):
    cum_slice = cum_scan.sum(axis=0)
    avg_slice = avg_scan.sum(axis=0)

    if os.path.exists(os.path.join(analysis_root, f'cum_scan_align{align}_eq.npy')):
        cum_slice_eq = np.load(os.path.join(analysis_root, f'cum_slice_align{align}_eq.npy'))
        cum_scan_eq = np.load(os.path.join(analysis_root, f'cum_scan_align{align}_eq.npy'))
        cum_scan_eq_slicewise = np.load(os.path.join(analysis_root, f'cum_scan_align{align}_eq_slicewise.npy'))
        avg_slice_eq = np.load(os.path.join(analysis_root, f'avg_slice_align{align}_eq.npy'))
        avg_scan_eq = np.load(os.path.join(analysis_root, f'avg_scan_align{align}_eq.npy'))
        avg_scan_eq_slicewise = np.load(os.path.join(analysis_root, f'avg_scan_align{align}_eq_slicewise.npy'))
        print('Loaded existing analysis')
    else:
        print('No existing analysis found, computing...')
        cum_scan_ = minmax(cum_scan)
        cum_slice_ = minmax(cum_slice)
        avg_scan_ = minmax(avg_scan)
        avg_slice_ = minmax(avg_slice)
        cum_slice_eq = histogram_equalization(cum_slice_)
        cum_scan_eq = histogram_equalization(cum_scan_)
        cum_scan_eq_slicewise = equalize_per_slice(cum_scan_)
        avg_slice_eq = histogram_equalization(avg_slice_)
        avg_scan_eq = histogram_equalization(avg_scan_)
        avg_scan_eq_slicewise = equalize_per_slice(avg_scan_)
        np.save(os.path.join(analysis_root, f'cum_slice_align{align}_eq.npy'), cum_slice_eq)
        np.save(os.path.join(analysis_root, f'cum_scan_align{align}_eq.npy'), cum_scan_eq)
        np.save(os.path.join(analysis_root, f'cum_scan_align{align}_eq_slicewise.npy'), cum_scan_eq_slicewise)
        np.save(os.path.join(analysis_root, f'avg_slice_align{align}_eq.npy'), avg_slice_eq)
        np.save(os.path.join(analysis_root, f'avg_scan_align{align}_eq.npy'), avg_scan_eq)
        np.save(os.path.join(analysis_root, f'avg_scan_align{align}_eq_slicewise.npy'), avg_scan_eq_slicewise)
        print('Saved files:',
            os.path.join(analysis_root, f'cum_slice_align{align}_eq.npy'),
            os.path.join(analysis_root, f'cum_scan_align{align}_eq.npy'),
            os.path.join(analysis_root, f'cum_scan_align{align}_eq_slicewise.npy'),
            os.path.join(analysis_root, f'avg_slice_align{align}_eq.npy'),
            os.path.join(analysis_root, f'avg_scan_align{align}_eq.npy'),
            os.path.join(analysis_root, f'avg_scan_align{align}_eq_slicewise.npy'),
            sep='\n\t')

    out = (cum_scan, cum_slice, cum_slice_eq, cum_scan_eq, cum_scan_eq_slicewise,
           avg_scan, avg_slice, avg_slice_eq, avg_scan_eq, avg_scan_eq_slicewise)
    return out


def compute_or_load_conn_comp_analysis(analysis_root, images_path, fn_preprocess, max_files=None, max_slices=None):
    path = os.path.join(analysis_root, 'conn_comp.csv')

    if os.path.exists(path):
        df_conn_comp = pd.read_csv(path)
        print('Loaded existing analysis')
    else:
        print('No existing analysis found, computing...')
        df_conn_comp = analysis_conn_comp(
            images_path, fn_preprocess,
            struct_elem=SQUARE_STRUCT,
            max_files=max_files, max_slices=max_slices
        )
        os.makedirs(analysis_root, exist_ok=True)
        df_conn_comp.to_csv(path, index=False)
        print('Saved files:', path, sep='\n\t')

    return df_conn_comp


# main

if __name__ == '__main__':
    DATA_ROOT = os.path.join(os.environ['HOME'], 'rib-fracture', 'data')
    DATASET_ROOT = os.path.join(DATA_ROOT, 'ribfrac')
    ANALYSIS_ROOT = os.path.join(DATA_ROOT, 'analysis')
    TRAIN_IMAGES = os.path.join(DATASET_ROOT, 'ribfrac-train-images')
    TRAIN_LABELS = os.path.join(DATASET_ROOT, 'ribfrac-train-labels')
    TRAIN_INFO = os.path.join(DATASET_ROOT, 'ribfrac-train-info.csv')

    def preprocess1(img):
        """
        First step of preprocessing: clip values (top and bottom) and minmax-normalize.
        """
        clip_min_val = 100  # design hyperparameter
        clip_max_val = 8000  # design hyperparameter
        img = img.clip(clip_min_val, clip_max_val)
        img = (img - img.min()) / (img.max() - img.min())
        return img

    #rib_data = compute_rib_data([TRAIN_INFO])

    print('\nFRACTURE ANALYSIS')
    #_, _ = compute_or_load_fracture_analysis(ANALYSIS_ROOT, TRAIN_LABELS, rib_data)

    print('\nPIXEL ANALYSIS')
    for align in ['top', 'bottom', 'center', 'fit']:
        break  # TODO: compute this!
        print('\nALIGN={}'.format(align))
        _, _, cum_scan, avg_scan = compute_or_load_pixel_analysis(ANALYSIS_ROOT, TRAIN_IMAGES, preprocess1, align=align)
        #_ = compute_or_load_eq(ANALYSIS_ROOT, cum_scan, avg_scan, align=align)  # TODO: tmp - ignore eq

    print('\nCONNECTED COMPONENTS ANALYSIS')
    _ = compute_or_load_conn_comp_analysis(ANALYSIS_ROOT, TRAIN_IMAGES, preprocess1, max_files=10, max_slices=50)

