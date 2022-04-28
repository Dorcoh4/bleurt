#! /bin/sh
#SBATCH --job-name=eli5-score_man2
#SBATCH --output=score_man2.out
#SBATCH --error=score_man2.err
#SBATCH --partition=studentbatch
#SBATCH --time=1500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
python -m bleurt.score_files  -sentence_pairs_file=sentence_pairs_copy_input.jsonl  -bleurt_checkpoint=eli52_bleurt-base-512_passover2/export/bleurt_best/1651004784 -read_buffer_size=40000 -scores_file=scores_copy_input
