#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "rsr and rsr scat on cora, citeseer, cora_full, pubmed, amazon-computer and amazon photo"


echo "Start citeseer"
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 20 --num_clique 20 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 20 --num_clique 20 --dataset citeseer


CUDA_VISIBLE_DEVICES=1 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 20 --num_clique 20 --decoder 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 20 --num_clique 20 --decoder 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 20 --num_clique 20 --decoder 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 20 --num_clique 20 --decoder 1 --dataset citeseer


CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 20 --num_clique 20 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 20 --num_clique 20 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 20 --num_clique 20 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 20 --num_clique 20 --dataset citeseer

CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 0 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 20 --num_clique 20 --decoder 1 --dataset citeseer
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 20 --num_clique 20 --decoder 1 --dataset citeseer
echo "citeseer job finished!"

echo "start amz computer"
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 50 --num_clique 30 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 50 --num_clique 30 --dataset amazon_electronics_computers


CUDA_VISIBLE_DEVICES=1 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 50 --num_clique 30 --decoder 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 50 --num_clique 30 --decoder 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 50 --num_clique 30 --decoder 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 50 --num_clique 30 --decoder 1 --dataset amazon_electronics_computers


CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 50 --num_clique 30 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 50 --num_clique 30 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 50 --num_clique 30 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 50 --num_clique 30 --dataset amazon_electronics_computers

CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 0 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 50 --num_clique 30 --decoder 1 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=1 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 50 --num_clique 30 --decoder 1 --dataset amazon_electronics_computers

echo "amz computer job finished!"