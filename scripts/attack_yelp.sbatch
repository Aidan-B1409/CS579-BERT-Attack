#!/bin/bash
#SBATCH -w cn-m-1
#SBATCH -p soundbendor
#SBATCH -A soundbendor
#SBATCH -t 3-00:00:00
#SBATCH --mem 32G
#SBATCH --job-name=attack
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

#module load cuda
#export PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/lib64/
source /nfs/stak/users/beerya/.local/bin/micromamba
source activate bert_attack_2

python bertattack.py --data_path data_defense/yelp.tsv_t --mlm_path bert-base-uncased --tgt_path ../BERT/yelp --use_sim_mat 1 --output_dir data_defense/yelp_adv --num_label 2 --use_bpe 1 --k 48 --start 0 --end 100 --threshold_pred_score 0
