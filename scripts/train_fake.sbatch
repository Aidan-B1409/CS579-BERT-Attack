#!/bin/bash
#SBATCH -w dgx2-2
#SBATCH -p dgx2
#SBATCH -A eecs
#SBATCH --job-name=victim_f
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --export=ALL

#module load cuda
#export PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/lib64/
source /nfs/stak/users/beerya/.local/bin/micromamba
source activate bert_attack_2

task="fake-news"

for model in roberta-base distilbert-base-cased uw-madison/nystromformer-512; do
	mkdir "tmp/${model}_${task}"
	python3 transformers/examples/pytorch/text-classification/run_glue.py \
		--model_name_or_path $model \
		--train_file data/fake_train.csv \
		--validation_file  data/fake_test.csv \
		--do_train \
		--do_eval \
		--max_seq_length 128 \
		--per_device_train_batch_size 32 \
		--learning_rate 2e-5 \
		--num_train_epochs 3 \
		--output_dir "tmp/${model}_${task}"
done
