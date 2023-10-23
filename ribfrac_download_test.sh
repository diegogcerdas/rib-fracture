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

# test-images (17.9 GB)
wget -c "https://zenodo.org/record/3993380/files/ribfrac-test-images.zip?download=1" -O "download/ribfrac-test-images.zip"
echo "[INFO] Test Set - images DONE"

unzip download/ribfrac-test-images.zip -d .

gzip -d ribfrac-*-images/*.gz
