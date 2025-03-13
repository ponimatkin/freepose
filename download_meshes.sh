#!/bin/bash

mkdir -p ./data/datasets

python scripts/download_objaverse.py

cd ./data/
wget https://data.ciirc.cvut.cz/public/projects/2025FreePose/objaverse_shards_ffa_22.npy
cd ./datasets/
wget https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/google_scanned_objects.zip
unzip google_scanned_objects.zip && rm google_scanned_objects.zip
cd ../../

python scripts/resize_meshes.py
