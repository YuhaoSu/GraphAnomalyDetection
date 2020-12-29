echo "amazon_electronics_computers!"
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 0 --data_type 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_onedecoder.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --decoder 1 --data_type 3 --dataset amazon_electronics_computers
CUDA_VISIBLE_DEVICES=0 python3 train_normal_twodecoders.py --hidden1 0.5 --hidden2 0.25  --clique_size 50 --num_clique 30 --data_type 3 --dataset amazon_electronics_computers
echo "amazon_electronics_computers finished!"