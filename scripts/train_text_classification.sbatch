#!/bin/bash
#SBATCH -w dgx2-5
#SBATCH -p dgx2
#SBATCH -A eecs
#SBATCH --job-name=vitcim-models-textclass
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --export=ALL

#module load cuda
#export PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/lib64/
source /nfs/stak/users/beerya/.local/bin/micromamba
source activate bert_attack_2


for model in uw-madison/nystromformer-512 distilbert-base-cased; do
	for task in ag_news yelp_review_full imdb; do
		mkdir "tmp/${task}"
		python3 transformers/examples/pytorch/text-classification/run_glue.py \
			--model_name_or_path $model \
			--dataset_name $task \
			--do_train \
			--do_eval \
			--max_seq_length 128 \
			--per_device_train_batch_size 32 \
			--learning_rate 2e-5 \
			--num_train_epochs 3 \
			--output_dir "tmp/${model}_${task}"
	done
done
