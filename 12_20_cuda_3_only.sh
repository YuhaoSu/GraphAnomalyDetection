#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "We first want to compare the feature decoder and structure decoder performance on only one anomaly"

echo "cora!"
echo "feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 data_type 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 data_type 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 data_type 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 data_type 3 --dataset cora

echo "structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 data_type 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 data_type 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 data_type 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 data_type 3 --dataset cora

echo "scat feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 data_type 0 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 data_type 1 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 data_type 2 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 data_type 3 --decoder 0 --att 0 --dataset cora

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 data_type 0 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 data_type 1 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 data_type 2 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 data_type 3 --decoder 0 --att 0 --dataset cora

echo "scat structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 data_type 0 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 data_type 1 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 data_type 2 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 data_type 3 --decoder 0 --att 0 --dataset cora

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 data_type 0 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 data_type 1 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 data_type 2 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 data_type 3 --decoder 1 --att 0 --dataset cora
echo "cora finished!"

echo "citeseer!"
echo "feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 data_type 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 data_type 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 data_type 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 data_type 3 --dataset citeseer

echo "structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 data_type 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 data_type 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 data_type 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 data_type 3 --dataset citeseer

echo "scat feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 data_type 0 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 data_type 1 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 data_type 2 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 data_type 3 --decoder 0 --att 0 --dataset citeseer

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 data_type 0 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 data_type 1 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 data_type 2 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 data_type 3 --decoder 0 --att 0 --dataset citeseer

echo "scat structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 data_type 0 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 data_type 1 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 data_type 2 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 data_type 3 --decoder 0 --att 0 --dataset citeseer

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 data_type 0 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 data_type 1 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 data_type 2 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 data_type 3 --decoder 1 --att 0 --dataset citeseer
echo "citeseer finished!"

echo "pubmed!"
echo "feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 data_type 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 data_type 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 data_type 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 data_type 3 --dataset pubmed

echo "structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 data_type 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 data_type 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 data_type 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 data_type 3 --dataset pubmed

echo "scat feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 0 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 1 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 2 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 3 --decoder 0 --att 0 --dataset pubmed

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 0 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 1 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 2 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 3 --decoder 0 --att 0 --dataset pubmed

echo "scat structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 0 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 1 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 2 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 3 --decoder 0 --att 0 --dataset pubmed

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 0 --decoder 1 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 1 --decoder 1 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 2 --decoder 1 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 3 --decoder 1 --att 0 --dataset pubmed
echo "pubmed finished!"

echo "amazon_electronics_photo!"
echo "feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 0 data_type 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 0 data_type 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 0 data_type 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 0 data_type 3 --dataset amazon_electronics_photo

echo "structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 1 data_type 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 1 data_type 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 1 data_type 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 1 data_type 3 --dataset amazon_electronics_photo

echo "scat feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 data_type 0 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 data_type 1 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 data_type 2 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 data_type 3 --decoder 0 --att 0 --dataset amazon_electronics_photo

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 data_type 0 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 data_type 1 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 data_type 2 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 data_type 3 --decoder 0 --att 0 --dataset amazon_electronics_photo

echo "scat structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 data_type 0 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 data_type 1 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 data_type 2 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 data_type 3 --decoder 0 --att 0 --dataset amazon_electronics_photo

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 data_type 0 --decoder 1 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 data_type 1 --decoder 1 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 data_type 2 --decoder 1 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 data_type 3 --decoder 1 --att 0 --dataset amazon_electronics_photo
echo "amazon_electronics_photo finished!"

echo "amazon_electronics_computers!"
echo "feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 0 data_type 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 0 data_type 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 0 data_type 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 0 data_type 3 --dataset amazon_electronics_computers

echo "structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 1 data_type 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 1 data_type 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 1 data_type 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 1 data_type 3 --dataset amazon_electronics_computers

echo "scat feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 data_type 0 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 data_type 1 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 data_type 2 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 data_type 3 --decoder 0 --att 0 --dataset amazon_electronics_computers

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 data_type 0 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 data_type 1 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 data_type 2 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 data_type 3 --decoder 0 --att 0 --dataset amazon_electronics_computers

echo "scat structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 data_type 0 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 data_type 1 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 data_type 2 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 data_type 3 --decoder 0 --att 0 --dataset amazon_electronics_computers

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 data_type 0 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 data_type 1 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 data_type 2 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 data_type 3 --decoder 1 --att 0 --dataset amazon_electronics_computers
echo "amazon_electronics_computers finished!"

echo "ms_academic_cs!"
echo "feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 data_type 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 data_type 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 data_type 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 data_type 3 --dataset ms_academic_cs

echo "structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 data_type 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 data_type 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 data_type 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 data_type 3 --dataset ms_academic_cs

echo "scat feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 0 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 1 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 2 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 3 --decoder 0 --att 0 --dataset ms_academic_cs

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 0 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 1 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 2 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 3 --decoder 0 --att 0 --dataset ms_academic_cs

echo "scat structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 0 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 1 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 2 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 data_type 3 --decoder 0 --att 0 --dataset ms_academic_cs

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 0 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 1 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 2 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 data_type 3 --decoder 1 --att 0 --dataset ms_academic_cs
echo "ms_academic_cs finished!"

echo "cora_full!"
echo "feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 0 data_type 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 0 data_type 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 0 data_type 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 0 data_type 3 --dataset cora_full

echo "structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 1 data_type 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 1 data_type 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 1 data_type 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 1 data_type 3 --dataset cora_full

echo "scat feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 data_type 0 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 data_type 1 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 data_type 2 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 data_type 3 --decoder 0 --att 0 --dataset cora_full

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 data_type 0 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 data_type 1 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 data_type 2 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 data_type 3 --decoder 0 --att 0 --dataset cora_full

echo "scat structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 data_type 0 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 data_type 1 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 data_type 2 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 data_type 3 --decoder 0 --att 0 --dataset cora_full

CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 data_type 0 --decoder 1 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 data_type 1 --decoder 1 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 data_type 2 --decoder 1 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 data_type 3 --decoder 1 --att 0 --dataset cora_full
echo "cora_full finished!"
echo "Complete!!!"
