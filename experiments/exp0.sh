# experiment 0: baseline unet3plus-ds-cgm, no context

DATA_ROOT=/scratch-shared/$USER/data/ribfrac

conda activate rib

python train.py --data-root DATA_ROOT --download-data --use-model unet3plus-ds-cgm --context-size 0
