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
- data 
|-- ribfrac-train-images
|-- ribfrac-train-labels
|-- ribfrac-val-images
|-- ribfrac-val-labels
|-- ribfrac-test-images
```

Each subfolder should contain the corresponding `.nii` files.

**Note**: omit `ribfrac-test-images` on Snellius for now, otherwise there is no storage space.

## Training
(See train.py for more arguments)

```
python train.py --max-epochs 5 
```