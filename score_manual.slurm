#! /bin/sh
#SBATCH --job-name=eli5-score_man
#SBATCH --output=score_man.out
#SBATCH --error=score_man.err
#SBATCH --partition=studentbatch
#SBATCH --time=1500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
python -m bleurt.score_files  -sentence_pairs_file=sentence_pairs_manual.jsonl  -bleurt_checkpoint=eli5_bleurt-base-512_checkpoint_steps2 -read_buffer_size=40000 -scores_file=scores_steps2
