#!/bin/bash

cd ./data/datasets

wget https://data.ciirc.cvut.cz/public/projects/2025FreePose/hope_video.zip
wget https://data.ciirc.cvut.cz/public/projects/2025FreePose/ycbv.zip
wget https://data.ciirc.cvut.cz/public/projects/2025FreePose/videos.zip

unzip hope_video.zip && rm hope_video.zip
unzip ycbv.zip && rm ycbv.zip
unzip videos.zip && rm videos.zip

cd ..

wget https://data.ciirc.cvut.cz/public/projects/2025FreePose/video_gt.zip

unzip video_gt.zip && rm video_gt.zip

mkdir checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P checkpoints

cd ../..