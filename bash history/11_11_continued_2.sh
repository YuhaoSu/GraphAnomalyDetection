#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "continue citeseer, and ms_academic_cs, ms_academic_phy"


CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --att 0 --dataset citeseer
echo "citeseer job finished!"


echo "ms_academic_cs experiment with fixed GCN hidden layer"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 --dataset ms_academic_cs


CUDA_VISIBLE_DEVICES=3 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --dataset ms_academic_cs


CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 3 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 3 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 3 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 3 --dataset ms_academic_cs



CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 2 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 3 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 3 --dataset ms_academic_cs
echo "ms_academic_cs job finished!"


echo "ms_academic_phy experiment with fixed GCN hidden layer"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 --dataset ms_academic_phy


CUDA_VISIBLE_DEVICES=3 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --dataset ms_academic_phy


CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 0 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 0 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 0 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 0 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 1 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 1 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 1 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 1 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 2 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 2 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 2 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 2 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 3 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 3 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 3 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 3 --dataset ms_academic_phy



CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 0 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 0 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 2 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 2 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 3 --dataset ms_academic_phy
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 3 --dataset ms_academic_phy
echo "ms_academic_phy job finished!"

