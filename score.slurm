#! /bin/sh
#SBATCH --job-name=eli5-score
#SBATCH --output=score.out
#SBATCH --error=score.err
#SBATCH --partition=studentbatch
#SBATCH --time=1500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
python -m bleurt.score_files  -sentence_pairs_file=sentence_pairs_longest_gold_ans.jsonl  -bleurt_checkpoint=bleurt-base-512 -read_buffer_size=40000 -scores_file=scores_base_longest_gold
python -m bleurt.score_files  -sentence_pairs_file=sentence_pairs_random_train_ans.jsonl  -bleurt_checkpoint=bleurt-base-512 -read_buffer_size=40000 -scores_file=scores_base_random_train
python -m bleurt.score_files  -sentence_pairs_file=sentence_pairs_copy_input.jsonl  -bleurt_checkpoint=bleurt-base-512 -read_buffer_size=40000 -scores_file=scores_base_copy_input
