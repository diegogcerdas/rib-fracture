# rib-fracture

## Environment

```
conda create -n rib python=3.10
conda activate rib
pip install -r ~/rib-fracture/requirements.txt
```

## Data structure

Please create a folder `data` inside the repo folder, with the following structure:
```
- data/
|-- ribfrac-train-images/
|-- ribfrac-train-labels/
|-- ribfrac-val-images/
|-- ribfrac-val-labels/
|-- ribfrac-test-images/
|-- ribfrac-train-info.csv
|-- ribfrac-val-info.csv
```

Each subfolder should contain the corresponding `.nii` files.

**Note**: omit `ribfrac-test-images` on Snellius for now, otherwise there is no storage space.

### Download script

The data can be downloaded using the script `ribfrac_download.sh` as follows:

```
DATA_ROOT=data/
bash ribfrac_download.sh DATA_ROOT
```

The data will be downloaded in the provided path `DATA_ROOT`. The script will download data if missing or resume an existing download. The script also restructures the data once downloaded to match the directory tree structure explained above. Alternatively, use the `--download-data` flag in the training script (see below).

## Training
(See train.py for more arguments)

```
DATA_ROOT=data/
python train.py --max-epochs 5 --data_root DATA_ROOT --download_data
```