#!/bin/bash
source activate augus
cd /home/augus/ad/gae_pytorch
echo "We use F norm to compare"

python3 analyze_encoder_v2.py --address cora_output
python3 analyze_encoder_v2.py --address citeseer_output
python3 analyze_encoder_v2.py --address pubmed_output
python3 analyze_encoder_v2.py --address amazon_electronics_computers_output
python3 analyze_encoder_v2.py --address amazon_electronics_photo_output
python3 analyze_encoder_v2.py --address ms_academic_cs_output

echo "finished!"