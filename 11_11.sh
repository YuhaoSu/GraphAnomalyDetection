#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "cora, citeseer, pubmed, amz-photo, amz-computer, cora-full"
echo "cora experiment with fixed GCN hidden layer"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --decoder 1 --dataset cora


CUDA_VISIBLE_DEVICES=3 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 15 --dataset cora


CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 1 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 0 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 1 --att 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 0 --att 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --decoder 1 --att 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --decoder 0 --att 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 1 --att 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 0 --att 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 1 --att 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 0 --att 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --decoder 1 --att 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --decoder 0 --att 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 1 --att 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 0 --att 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 1 --att 3 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --decoder 0 --att 3 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --decoder 1 --att 3 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --decoder 0 --att 3 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 1 --att 3 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --decoder 0 --att 3 --dataset cora


CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --att 0 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --att 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --att 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --att 1 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --att 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --att 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --att 2 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 15 --att 3 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 15 --att 3 --dataset cora
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 15 --att 3 --dataset cora

echo "cora job finished!"

echo "citeseer experiment with fixed GCN hidden layer"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --decoder 1 --dataset citeseer


CUDA_VISIBLE_DEVICES=3 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 20 --num_clique 20 --dataset citeseer


CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 1 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 0 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 1 --att 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 0 --att 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --decoder 1 --att 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --decoder 0 --att 1 --dataset citeseer
!!!CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 1 --att 1 --dataset citeseer
!!!CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 0 --att 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 1 --att 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 0 --att 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --decoder 1 --att 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --decoder 0 --att 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 1 --att 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 0 --att 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 1 --att 3 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 0 --att 3 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --decoder 1 --att 3 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --decoder 0 --att 3 --dataset citeseer
!!!CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 1 --att 3 --dataset citeseer
!!!CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 0 --att 3 --dataset citeseer


CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --att 0 --dataset citeseer
!!!CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --att 0 --dataset citeseer
???CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --att 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --att 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --att 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --att 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --att 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --att 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --att 2 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 20 --num_clique 20 --att 3 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 20 --num_clique 20 --att 3 --dataset citeseer
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 20 --num_clique 20 --att 3 --dataset citeseer

echo "citeseer job finished!"

echo "pubmed experiment with fixed GCN hidden layer"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --decoder 1 --dataset pubmed


CUDA_VISIBLE_DEVICES=3 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 40 --dataset pubmed


CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 1 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 0 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 1 --att 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 0 --att 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 1 --att 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 0 --att 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --att 3 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --att 3 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 1 --att 3 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --decoder 0 --att 3 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 1 --att 3 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 0 --att 3 --dataset pubmed


CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --att 0 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --att 1 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --att 2 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 40 --att 3 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 40 --att 3 --dataset pubmed
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 40 --att 3 --dataset pubmed

echo "pubmed job finished!"

echo "amazon_electronics_photo experiment with fixed GCN hidden layer"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --decoder 1 --dataset amazon_electronics_photo


CUDA_VISIBLE_DEVICES=3 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 30 --num_clique 30 --dataset amazon_electronics_photo


CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 1 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --decoder 1 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 1 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 0 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 1 --att 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 0 --att 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --decoder 1 --att 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --decoder 0 --att 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 1 --att 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 0 --att 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 1 --att 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 0 --att 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --decoder 1 --att 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --decoder 0 --att 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 1 --att 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 0 --att 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 1 --att 3 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --decoder 0 --att 3 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --decoder 1 --att 3 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --decoder 0 --att 3 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 1 --att 3 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --decoder 0 --att 3 --dataset amazon_electronics_photo


CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --att 0 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --att 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --att 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --att 1 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --att 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --att 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --att 2 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 30 --num_clique 30 --att 3 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 30 --num_clique 30 --att 3 --dataset amazon_electronics_photo
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 30 --num_clique 30 --att 3 --dataset amazon_electronics_photo

echo "amazon_electronics_photo job finished!"

echo "amazon_electronics_computers experiment with fixed GCN hidden layer"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 1 --dataset amazon_electronics_computers


CUDA_VISIBLE_DEVICES=3 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --dataset amazon_electronics_computers


CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 1 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 0 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 1 --att 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 0 --att 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --decoder 1 --att 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --decoder 0 --att 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 1 --att 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 0 --att 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 1 --att 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 0 --att 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --decoder 1 --att 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --decoder 0 --att 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 1 --att 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 0 --att 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 1 --att 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 0 --att 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --decoder 1 --att 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --decoder 0 --att 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 1 --att 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 0 --att 3 --dataset amazon_electronics_computers


CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --att 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --att 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --att 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --att 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --att 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --att 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --att 2 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 30 --att 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 30 --att 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 30 --att 3 --dataset amazon_electronics_computers

echo "amazon_electronics_computers job finished!"

echo "cora_full experiment with fixed GCN hidden layer"
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --decoder 1 --dataset cora_full


CUDA_VISIBLE_DEVICES=3 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 45 --dataset cora_full


CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 1 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --decoder 1 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 1 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 0 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 1 --att 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 0 --att 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --decoder 1 --att 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --decoder 0 --att 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 1 --att 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 0 --att 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 1 --att 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 0 --att 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --decoder 1 --att 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --decoder 0 --att 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 1 --att 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 0 --att 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 1 --att 3 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 0 --att 3 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --decoder 1 --att 3 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --decoder 0 --att 3 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 1 --att 3 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 0 --att 3 --dataset cora_full


CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --att 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --att 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --att 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --att 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --att 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --att 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --att 2 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 0 --clique_size 50 --num_clique 45 --att 3 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 1 --clique_size 50 --num_clique 45 --att 3 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25 --dim_reduce 2 --clique_size 50 --num_clique 45 --att 3 --dataset cora_full``
echo "cora_full job finished!"

