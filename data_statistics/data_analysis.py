import os

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools
import seaborn as sns

DATA_ROOT = os.environ['HOME'] + '/data'
DATASET_ROOT = DATA_ROOT + '/ribfrac'
TRAIN_ROOT = DATASET_ROOT + '/train'
VAL_ROOT = DATASET_ROOT + '/val'
ANALYSIS_ROOT = DATA_ROOT + '/analysis'

LABEL_CODE = {
    0: 'background',
    1: 'displaced',
    2: 'non-displaced',
    3: 'buckle',
    4: 'segmental',
    -1: 'undefined'
}

rib_frac_types = {name: 0 for name in LABEL_CODE.values()}
rib_data = defaultdict(list)
nr_frac = defaultdict(int)

info_files = [
    os.path.join(TRAIN_ROOT, 'ribfrac-train-info-1.csv'),
    os.path.join(TRAIN_ROOT, 'ribfrac-train-info-2.csv'),
    #os.path.join(VAL_ROOT, 'ribfrac-val-info.csv')
]
for info_file in info_files:
    for index, row in pd.read_csv(info_file).iterrows():
        public_id, label_id, label_code = row
        rib_data[public_id].append(label_code)
        nr_frac[public_id] += 1
        if label_code != 0:  # ignore background
            rib_frac_types[LABEL_CODE[label_code]] += 1
        
print('rib_data=', rib_data)
print('nr_frac=', nr_frac)  # nr_frac is just the len of each rib_data
#print(rib_frac_types)

total_rib_frac = sum(rib_frac_types.values())
avg_frac = total_rib_frac/len(nr_frac)
print('average fractures', avg_frac)
print('total fractures',total_rib_frac)

# def get_axis(scan, label, idx):
#     min_x = -999
#     max_x = -999
#     min_y = -999
#     max_y = -999
# #     xs = []
# #     ys = []
#     for i, slice_ in enumerate(scan[idx[0]:idx[-1]+1]):
#         #frac = np.where(slice_ == label, 1, 0)
#         #xs.append(max(np.sum(frac, axis = 1)))
#         #ys.append(max(np.sum(frac, axis = 0)))
#         x, y = np.where(slice_ == label)
#         min_x, max_x, min_y, max_y = np.min(x), np.max(x), np.min(y), np.max(y)
#     if min_x == -999:
#         print(len(scan), idx, scan)
#     return min_x, max_x, min_y, max_y

def get_bbox(scan):
    """get the 3d bounding box of the volume. scan is binary boolean array with coords (z, x, y)"""
    z, x, y = np.where(scan)
    assert len(z) > 0, 'scan is empty'
    min_x, max_x, min_y, max_y, min_z, max_z = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
    return min_x, max_x, min_y, max_y, min_z, max_z

def get_max_xy_box(scan):
    """gets the largest window size that could contain the area of the facture in each slice. scan is binary boolean array with coords (z, x, y)"""
    xs = np.any(scan, axis=2, keepdims=True)  # any in y direction
    ys = np.any(scan, axis=1, keepdims=True)  # any in x direction
    size_x = np.sum(xs, axis=1, keepdims=True)  # sum in x direction
    size_y = np.sum(ys, axis=2, keepdims=True)  # sum in y direction
    size_x = np.max(size_x)  # largest x size
    size_y = np.max(size_y)  # largest y size
    return size_x, size_y

def get_volume(scan):
    """volume of the volume defined by the binary boolean array scan with coords (z, x, y)"""
    return np.sum(scan)

def get_xy_areas(scan):
    """gets minimum and maximum 2d areas through z-planes. scan is binary boolean array with coords (z, x, y)"""
    areas = np.sum(scan, axis=(1, 2))  # sum in x and y direction
    return np.min(areas), np.max(areas)

def center_of_mass(scan) -> tuple:
    """cneter of mass of the volume defined by the binary boolean array scan with coords (z, x, y)"""
    z, x, y = np.where(scan)
    assert len(z) > 0, 'scan is empty'
    com_z = np.mean(z)
    com_x = np.mean(x)
    com_y = np.mean(y)
    return com_z, com_x, com_y


def fracture_label_analysis(train_folder_path, rib_data):
    """
    Produces 2 dataframes with the following information:

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

    for filename in tqdm(os.listdir(train_folder_path)[2:], desc='analyzing train data'):

        if not (filename.endswith("label.nii.gz") or filename.endswith("label.nii")):
            continue

        filepath = os.path.join(train_folder_path, filename)
        label_scan = nib.load(filepath).get_fdata().T.astype(int)
        
        public_id = filename.split('-')[0]
        scan_data.append([
            public_id,
            label_scan.shape[1],
            label_scan.shape[2],
            label_scan.shape[0],
        ])

        # data_dev = []
        # for slice_idx in range(len(label_scan)):
        #     unique_label_ids = np.unique(label_scan[slice_idx]).tolist()
        #     data_dev.append([slice_idx, unique_label_ids])
        # df_slices = pd.DataFrame(data_dev, columns=["slice_idx", "label_ids"])

        labels_per_slice = defaultdict(list)
        for slice_idx, slice in enumerate(label_scan):
            unique_label_ids = np.unique(slice).tolist()
            labels_per_slice[slice_idx] = unique_label_ids
            
        for frac_idx, frac_code in enumerate(rib_data[public_id]):

            # idx= list of slice indices where the fracture is present
            # idx = []
            # for ind in df_slices.index:
            #     if frac_idx in df_slices['label_ids'][ind]:
            #         idx.append(df_slices['slice_idx'][ind])
            # if idx == []:
            #     print('WARNING:', public_id, 'has no fracture')
            #     continue
            # idxs = sorted(idx)

            idxs = []
            for slice_idx, unique_label_ids in labels_per_slice.items():
                if frac_idx in unique_label_ids:
                    idxs.append(slice_idx)
            idxs = sorted(idxs)

            ## compute values

            reduced_scan = label_scan[idxs[0]:idxs[-1]+1]  # ease computation with only relevant slices
            min_x, max_x, min_y, max_y, min_z, max_z = get_bbox(reduced_scan == frac_idx)
            volume = get_volume(reduced_scan == frac_idx)
            max2dsize_x, max2dsize_y = get_max_xy_box(reduced_scan == frac_idx)
            min2darea, max2darea = get_xy_areas(reduced_scan == frac_idx)

            # compensate for reduced scan offset
            min_z += idxs[0]
            max_z += idxs[0]
            assert min_z == idxs[0], f'{min_z} != {idxs[0]}'
            assert max_z == idxs[-1], f'{max_z} != {idxs[-1]}'

            # again reduce scan to only relevant volume
            reduced_scan = label_scan[idxs[0]:idxs[-1]+1, min_x:max_x+1, min_y:max_y+1]
            com_z, com_x, com_y = center_of_mass(reduced_scan == frac_idx)  # com relative to chunk

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

        break

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

    return df_scan, df_frac

exists = os.path.exists(os.path.join(ANALYSIS_ROOT, 'scan.csv'))
if exists:
    df_scan = pd.read_csv(os.path.join(ANALYSIS_ROOT, 'scan.csv'))
    df_frac = pd.read_csv(os.path.join(ANALYSIS_ROOT, 'frac.csv'))
else:
    df_scan, df_frac = fracture_label_analysis(TRAIN_ROOT, rib_data)
    os.makedirs(ANALYSIS_ROOT, exist_ok=True)
    df_scan.to_csv(os.path.join(ANALYSIS_ROOT, 'scan.csv'), index=False)
    df_frac.to_csv(os.path.join(ANALYSIS_ROOT, 'frac.csv'), index=False)

print(df_frac.describe())