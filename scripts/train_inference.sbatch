#!/bin/bash
#SBATCH -w cn-m-1
#SBATCH -p soundbendor
#SBATCH -A soundbendor
#SBATCH --job-name=victim-i
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:6
#SBATCH --export=ALL

#module load cuda
#export PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/lib64/
source /nfs/stak/users/beerya/.local/bin/micromamba
source activate bert_attack_2

task="rungalileo/snli"

for model in bert-base-cased; do
	mkdir "tmp/${model}_${task}"
	python3 transformers/examples/pytorch/text-classification/run_classification.py \
		--model_name_or_path $model \
		--dataset_name $task \
		--shuffle_train_dataset \
		--metric_name accuracy \
		--text_column_name "premise,hypothesis" \
		--do_train \
		--do_eval \
		--max_seq_len 128 \
		--per_device_train_batch_size 32 \
		--learning_rate 2e-5 \
		--ignore_mismatched_sizes \
		--num_train_epochs 2 \
		--output_dir "tmp/${model}_${task}" \
		
done
