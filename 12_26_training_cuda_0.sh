#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "We first want to compare the feature decoder and structure decoder performance on only one anomaly"

echo "cora!"
echo "normal feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 --data_type 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 --data_type 1 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 --data_type 2 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 --data_type 3 --dataset cora

echo "normal structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 --data_type 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 --data_type 1 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 --data_type 2 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 --data_type 3 --dataset cora

echo "scat feature decoder pca on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --data_type 0 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --data_type 1 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --data_type 2 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --data_type 3 --decoder 0 --att 0 --dataset cora

echo "scat structure decoder pca on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --data_type 0 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --data_type 1 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --data_type 2 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --data_type 3 --decoder 1 --att 0 --dataset cora

echo "scat feature decoder fms on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --data_type 0 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --data_type 1 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --data_type 2 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --data_type 3 --decoder 0 --att 0 --dataset cora

echo "scat structure decoder fms on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --data_type 0 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --data_type 1 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --data_type 2 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --data_type 3 --decoder 1 --att 0 --dataset cora
echo "cora finished!"



echo "amazon_electronics_computers!"
echo "normal feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25 --clique_size 50 --num_clique 30 --decoder 0 --data_type 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25 --clique_size 50 --num_clique 30 --decoder 0 --data_type 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25 --clique_size 50 --num_clique 30 --decoder 0 --data_type 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25 --clique_size 50 --num_clique 30 --decoder 0 --data_type 3 --dataset amazon_electronics_computers

echo "normal structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25 --clique_size 50 --num_clique 30 --decoder 1 --data_type 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25 --clique_size 50 --num_clique 30 --decoder 1 --data_type 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25 --clique_size 50 --num_clique 30 --decoder 1 --data_type 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25 --clique_size 50 --num_clique 30 --decoder 1 --data_type 3 --dataset amazon_electronics_computers

echo "scat feature decoder pca on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0--clique_size 50 --num_clique 30 --data_type 0 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0--clique_size 50 --num_clique 30 --data_type 1 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0--clique_size 50 --num_clique 30 --data_type 2 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0--clique_size 50 --num_clique 30 --data_type 3 --decoder 0 --att 0 --dataset amazon_electronics_computers

echo "scat structure decoder pca on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0--clique_size 50 --num_clique 30 --data_type 0 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0--clique_size 50 --num_clique 30 --data_type 1 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0--clique_size 50 --num_clique 30 --data_type 2 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0--clique_size 50 --num_clique 30 --data_type 3 --decoder 1 --att 0 --dataset amazon_electronics_computers

echo "scat feature decoder fms on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1--clique_size 50 --num_clique 30 --data_type 0 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1--clique_size 50 --num_clique 30 --data_type 1 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1--clique_size 50 --num_clique 30 --data_type 2 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1--clique_size 50 --num_clique 30 --data_type 3 --decoder 0 --att 0 --dataset amazon_electronics_computers

echo "scat structure decoder fms on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1--clique_size 50 --num_clique 30 --data_type 0 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1--clique_size 50 --num_clique 30 --data_type 1 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1--clique_size 50 --num_clique 30 --data_type 2 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1--clique_size 50 --num_clique 30 --data_type 3 --decoder 1 --att 0 --dataset amazon_electronics_computers
echo "amazon_electronics_computers finished!"