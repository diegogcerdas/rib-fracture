# experiment 4: context=8, positional encodings

DATA_ROOT=/scratch-shared/$USER/data/ribfrac

conda activate rib

python train.py --data-root DATA_ROOT --download-data --use-model unet3plus-ds-cgm --context-size 8 --use-positional-encodings --exp-name exp4
