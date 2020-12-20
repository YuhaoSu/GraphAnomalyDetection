#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "start"

CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --dataset cora

echo "fisnished"
