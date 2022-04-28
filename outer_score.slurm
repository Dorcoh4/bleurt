#! /bin/sh
#SBATCH --job-name=eli5-outer_score
#SBATCH --output=outer_score.out
#SBATCH --error=outer_score.err
#SBATCH --partition=studentbatch
#SBATCH --time=1500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
python outer_score.py sentence_pairs_random_train_ans.jsonl
python outer_score.py sentence_pairs_longest_gold_ans.jsonl
python outer_score.py sentence_pairs_copy_input.jsonl
