# experiment 3: context=8, no cgm

DATA_ROOT=/scratch-shared/$USER/data/ribfrac

conda activate rib

python train.py --data-root DATA_ROOT --download-data --use-model unet3plus-ds --context-size 8 --exp-name exp3
