#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "pubmed!"
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 --data_type 3 --dataset pubmed
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 --data_type 3 --dataset pubmed
CUDA_VISIBLE_DEVICES=1 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --data_type 3 --dataset pubmed
echo "pubmed finished!"

echo "amazon_electronics_photo!"
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 0 --data_type 3 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 1 --data_type 3 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=1 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --data_type 3 --dataset amazon_electronics_photo
echo "amazon_electronics_photo finished!"

echo "ms_academic_cs!"
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 --data_type 3 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 --data_type 3 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --data_type 3 --dataset ms_academic_cs
echo "ms_academic_cs finished!"