#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch

echo "citeseer!"
echo "normal feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 --data_type 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 --data_type 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 --data_type 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 --data_type 3 --dataset citeseer

echo "normal structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 --data_type 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 --data_type 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 --data_type 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 --data_type 3 --dataset citeseer

echo "scat feature decoder pca on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --data_type 0 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --data_type 1 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --data_type 2 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --data_type 3 --decoder 0 --att 0 --dataset citeseer

echo "scat structure decoder pca on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --data_type 0 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --data_type 1 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --data_type 2 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --data_type 3 --decoder 1 --att 0 --dataset citeseer

echo "scat feature decoder fms on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --data_type 0 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --data_type 1 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --data_type 2 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --data_type 3 --decoder 0 --att 0 --dataset citeseer

echo "scat structure decoder fms on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --data_type 0 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --data_type 1 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --data_type 2 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --data_type 3 --decoder 1 --att 0 --dataset citeseer
echo "citeseer finished!"


echo "ms_academic_cs!"
echo "normal feature decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  -clique_size 50 --num_clique 40 --decoder 0 --data_type 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  -clique_size 50 --num_clique 40 --decoder 0 --data_type 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  -clique_size 50 --num_clique 40 --decoder 0 --data_type 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  -clique_size 50 --num_clique 40 --decoder 0 --data_type 3 --dataset ms_academic_cs

echo "normal structure decoder on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  -clique_size 50 --num_clique 40 --decoder 1 --data_type 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  -clique_size 50 --num_clique 40 --decoder 1 --data_type 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  -clique_size 50 --num_clique 40 --decoder 1 --data_type 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  -clique_size 50 --num_clique 40 --decoder 1 --data_type 3 --dataset ms_academic_cs

echo "scat feature decoder pca on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 -clique_size 50 --num_clique 40 --data_type 0 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 -clique_size 50 --num_clique 40 --data_type 1 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 -clique_size 50 --num_clique 40 --data_type 2 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 -clique_size 50 --num_clique 40 --data_type 3 --decoder 0 --att 0 --dataset ms_academic_cs

echo "scat structure decoder pca on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 -clique_size 50 --num_clique 40 --data_type 0 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 -clique_size 50 --num_clique 40 --data_type 1 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 -clique_size 50 --num_clique 40 --data_type 2 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 -clique_size 50 --num_clique 40 --data_type 3 --decoder 1 --att 0 --dataset ms_academic_cs

echo "scat feature decoder fms on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 -clique_size 50 --num_clique 40 --data_type 0 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 -clique_size 50 --num_clique 40 --data_type 1 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 -clique_size 50 --num_clique 40 --data_type 2 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 -clique_size 50 --num_clique 40 --data_type 3 --decoder 0 --att 0 --dataset ms_academic_cs

echo "scat structure decoder fms on 4 anomaly settings!"
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 -clique_size 50 --num_clique 40 --data_type 0 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 -clique_size 50 --num_clique 40 --data_type 1 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 -clique_size 50 --num_clique 40 --data_type 2 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=1 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 -clique_size 50 --num_clique 40 --data_type 3 --decoder 1 --att 0 --dataset ms_academic_cs
echo "ms_academic_cs finished!"