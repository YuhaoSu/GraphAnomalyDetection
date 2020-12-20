CUDA_VISIBLE_DEVICES=3 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 50 --num_clique 45 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 50 --num_clique 45 --dataset cora_full


CUDA_VISIBLE_DEVICES=3 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 50 --num_clique 45 --decoder 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --clique_size 50 --num_clique 45 --decoder 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 50 --num_clique 45 --decoder 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --clique_size 50 --num_clique 45 --decoder 1 --dataset cora_full


CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 50 --num_clique 45 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 50 --num_clique 45 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 50 --num_clique 45 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_twodecoders.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 50 --num_clique 45 --dataset cora_full

CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.1 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 0 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 0 --clique_size 50 --num_clique 45 --decoder 1 --dataset cora_full
CUDA_VISIBLE_DEVICES=3 python3 train_rsr_scat_onedecoder.py --hidden1 0.5 --hidden2 0.25  --rsr_dim 0.02 --dim_reduce 2 --clique_size 50 --num_clique 45 --decoder 1 --dataset cora_full
