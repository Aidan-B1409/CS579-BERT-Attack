#!/bin/bash
#SBATCH -w cn-m-2
#SBATCH -p soundbendor
#SBATCH -A soundbendor
#SBATCH --job-name=bert_train
#SBATCH -t 3-00:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

#module load cuda
#export PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/lib64/
source /nfs/stak/users/beerya/.local/bin/micromamba
source activate textfooler
python3 TextFooler/BERT/run_classifier_IMDB.py
