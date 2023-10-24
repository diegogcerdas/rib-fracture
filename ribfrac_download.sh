#!/bin/bash

#####
#
# Downloads all.
# Comment commands to skip downloads.
# "wget <URL> -O <OUT>" to overwrite file.
# "wget -c <URL> -O <OUT>" to resume download.
#
# - Training Set Part 1 (300 scans + annotations)
# - Training Set Part 2 (120 scans + annotations)
# - Tuning/Validation Set (80 scans + annotations)
# - Test Set (160 scans)
#
#####

DATASET_ROOT=$1  # data/ribfrac/

mkdir -p $DATASET_ROOT
cd $DATASET_ROOT

echo "[INFO] Downloading RIB-FRACTURE dataset at $(pwd)"

mkdir -p download

# train1-info (48.6 kB)
wget -c "https://zenodo.org/record/3893508/files/ribfrac-train-info-1.csv?download=1" -O "download/ribfrac-train-info-1.csv"
echo "[INFO] Training Set Part 1 - info DONE"

# train2-info (19.8 kB)
wget -c "https://zenodo.org/record/3893498/files/ribfrac-train-info-2.csv?download=1" -O "download/ribfrac-train-info-2.csv"
echo "[INFO] Training Set Part 2 - info DONE"

# val-info (8.1 kB)
wget -c "https://zenodo.org/record/3893496/files/ribfrac-val-info.csv?download=1" -O "download/ribfrac-val-info.csv"
echo "[INFO] Tuning/Validation Set - info DONE"


# train1-labels (7.9 MB)
wget -c "https://zenodo.org/record/3893508/files/ribfrac-train-labels-1.zip?download=1" -O "download/ribfrac-train-labels-1.zip"
echo "[INFO] Training Set Part 1 - labels DONE"

# train2-labels (3.0 MB)
wget -c "https://zenodo.org/record/3893498/files/ribfrac-train-labels-2.zip?download=1" -O "download/ribfrac-train-labels-2.zip"
echo "[INFO] Training Set Part 2 - labels DONE"

# val-labels (2.1 MB)
wget -c "https://zenodo.org/record/3893496/files/ribfrac-val-labels.zip?download=1" -O "download/ribfrac-val-labels.zip"
echo "[INFO] Tuning/Validation Set - labels DONE"


# val-images (8.7 GB)
wget -c "https://zenodo.org/record/3893496/files/ribfrac-val-images.zip?download=1" -O "download/ribfrac-val-images.zip"
echo "[INFO] Tuning/Validation Set - images DONE"

# train2-images (14.6 GB)
wget -c "https://zenodo.org/record/3893498/files/ribfrac-train-images-2.zip?download=1" -O "download/ribfrac-train-images-2.zip"
echo "[INFO] Training Set Part 2 - images DONE"

# train1-images (36.6 GB)
wget -c "https://zenodo.org/record/3893508/files/ribfrac-train-images-1.zip?download=1" -O "download/ribfrac-train-images-1.zip"
echo "[INFO] Training Set Part 1 - images DONE"

# test-images (17.9 GB)
#wget -c "https://zenodo.org/record/3993380/files/ribfrac-test-images.zip?download=1" -O "download/ribfrac-test-images.zip"
#echo "[INFO] Test Set - images DONE"


echo "[INFO] Unzipping .zip folders"
mkdir -p tmp_train_images
mkdir -p tmp_train_labels
unzip download/ribfrac-train-images-1.zip -d tmp_train_images
unzip download/ribfrac-train-images-2.zip -d tmp_train_images
unzip download/ribfrac-train-labels-1.zip -d tmp_train_labels
unzip download/ribfrac-train-labels-2.zip -d tmp_train_labels
unzip download/ribfrac-val-images.zip -d .
unzip download/ribfrac-val-labels.zip -d .
#unzip download/ribfrac-test-images.zip -d .

mkdir -p ribfrac-train-images
mkdir -p ribfrac-train-labels
mv tmp_train_images/Part*/*.nii.gz ribfrac-train-images/
mv tmp_train_labels/Part*/*.nii.gz ribfrac-train-labels/

awk '(NR == 1) || (FNR > 1)' download/ribfrac-train-info-*.csv > ribfrac-train-info.csv  # merge csv files
cp download/ribfrac-val-info.csv .

rmdir tmp_train_*/Part*
rmdir tmp_train_*
rm -r download/  # optional, free space

echo "[INFO] Unzipping .nii.gz files"
gzip -d ribfrac-*-images/*.gz
gzip -d ribfrac-*-labels/*.gz

echo "[INFO] RIB-FRACTURE dataset downloaded at $(pwd)"
