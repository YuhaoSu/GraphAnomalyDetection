#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "rsr and rsr scat on cora, citeseer, cora_full, pubmed, amazon-computer, amazon photo and  ms_academic_cs_output"


echo "Start ms_academic_cs"
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 50 --num_clique 40 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 50 --num_clique 40 --dataset ms_academic_cs


CUDA_VISIBLE_DEVICES=2 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 50 --num_clique 40 --decoder 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 50 --num_clique 40 --decoder 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 50 --num_clique 40 --decoder 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 50 --num_clique 40 --decoder 1 --dataset ms_academic_cs


CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 50 --num_clique 40 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 50 --num_clique 40 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 50 --num_clique 40 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 50 --num_clique 40 --dataset ms_academic_cs

CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 0 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 50 --num_clique 40 --decoder 1 --dataset ms_academic_cs
CUDA_VISIBLE_DEVICES=2 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 50 --num_clique 40 --decoder 1 --dataset ms_academic_cs
echo "ms_academic_cs job finished!"

