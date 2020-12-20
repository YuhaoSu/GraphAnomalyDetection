#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "test rsr and rsr scat"


echo "test rsr twodecoders"

CUDA_VISIBLE_DEVICES=3 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 20 --num_clique 15 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.01 --clique_size 20 --num_clique 15 --dataset citeseer

echo "test rsr onedecoder"
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 20 --num_clique 15 --decoder 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 20 --num_clique 15 --decoder 1 --dataset cora

echo "test rsr scat twodecoders"

CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 20 --num_clique 15 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 20 --num_clique 15 --dataset cora

echo "test rsr scat onedecoder"
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 0 --dataset cora

echo "test job finished!"

