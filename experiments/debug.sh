# DEBUG

DATA_ROOT=/scratch-shared/$USER/data/ribfrac

# activate env
source ~/.venv/bin/activate
#conda activate rib

# download data
#bash ribfrac_download.sh $DATA_ROOT

# expX1: unet3plus, no context, no ds, no cgm
python train.py --data-root DATA_ROOT --use-model unet3plus --context-size 0 --max-epochs 1 --batch-size-train 1

# exp0: baseline unet3plus-ds-cgm, no context
python train.py --data-root DATA_ROOT --use-model unet3plus-ds-cgm --context-size 0 --max-epochs 1 --batch-size-train 1

# exp1: context=8, no cgm
python train.py --data-root DATA_ROOT --use-model unet3plus-ds-cgm --context-size 8 --max-epochs 1 --batch-size-train 1

# exp2: no context, no cgm
python train.py --data-root DATA_ROOT --use-model unet3plus-ds --context-size 0 --max-epochs 1 --batch-size-train 1

# exp3: context=8, no cgm
python train.py --data-root DATA_ROOT --use-model unet3plus-ds --context-size 8 --max-epochs 1 --batch-size-train 1

