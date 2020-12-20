#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "We first want to compare the feture decoder and structure decoder for only one anomaly"




echo "Start cora"
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 20 --num_clique 15 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 20 --num_clique 15 --dataset cora


CUDA_VISIBLE_DEVICES=0 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 20 --num_clique 15 --decoder 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 20 --num_clique 15 --decoder 1 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 20 --num_clique 15 --decoder 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 20 --num_clique 15 --decoder 1 --dataset cora


CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 20 --num_clique 15 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 20 --num_clique 15 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 20 --num_clique 15 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 20 --num_clique 15 --dataset cora

CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 1 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 1 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 1 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 1 --dataset cora
echo "cora job finished!"

echo "Start amazon_electronics_photo"
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 30 --num_clique 30 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 30 --num_clique 30 --dataset amazon_electronics_photo


CUDA_VISIBLE_DEVICES=0 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 30 --num_clique 30 --decoder 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 30 --num_clique 30 --decoder 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 30 --num_clique 30 --decoder 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 30 --num_clique 30 --decoder 1 --dataset amazon_electronics_photo


CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 30 --num_clique 30 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 30 --num_clique 30 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 30 --num_clique 30 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 30 --num_clique 30 --dataset amazon_electronics_photo

CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=0 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 1 --dataset amazon_electronics_photo
echo "amazon_electronics_photo job finished!"