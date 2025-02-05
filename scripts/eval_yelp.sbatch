#!/bin/bash
#SBATCH -w cn-m-1
#SBATCH -p soundbendor
#SBATCH -A soundbendor
#SBATCH --job-name=vitcim-models-textclass
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

#module load cuda
#export PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/nfs/guille/eecs_research/soundbendor/beerya/cuda/lib64/
source /nfs/stak/users/beerya/.local/bin/micromamba
source activate bert_attack_2

for model in roberta-base uw-madison/nystromformer-512 distilbert-base-cased; do
	for task in yelp_review_full; do
		python3 transformers/examples/pytorch/text-classification/run_glue.py \
			--model_name_or_path "tmp/${model}_${task}" \
			--train_file "data_defense/${task}_adv.csv" \
			--validation_file "data_defense/${task}_adv.csv" \
			--do_eval \
			--max_seq_length 128 \
			--per_device_train_batch_size 32 \
			--ignore_mismatched_sizes \
			--learning_rate 2e-5 \
			--output_dir "tmp/eval_${model}_${task}"
	done
done
