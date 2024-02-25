#!/bin/bash

mkdir images
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg
wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg

mdir weights
wget -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -P weights https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
