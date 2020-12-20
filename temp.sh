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