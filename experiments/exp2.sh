# experiment 2: no context, no cgm

DATA_ROOT=/scratch-shared/$USER/data/ribfrac

conda activate rib

python train.py --data-root DATA_ROOT --download-data --use-model unet3plus-ds --context-size 0
