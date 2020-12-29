#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "cora!"
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 --data_type 3 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 --data_type 3 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --data_type 3 --dataset cora
echo "cora finished!"

echo "citeseer!"
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 --data_type 3 --dataset citeseer
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 --data_type 3 --dataset citeseer
CUDA_VISIBLE_DEVICES=0 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --data_type 3 --dataset citeseer
echo "citeseer finished!"

echo "amazon_electronics_computers!"
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 0 --data_type 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 1 --data_type 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --data_type 3 --dataset amazon_electronics_computers
echo "amazon_electronics_computers finished!"

