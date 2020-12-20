#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
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