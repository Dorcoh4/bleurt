#! /bin/sh
#SBATCH --job-name=eli5
#SBATCH --output=eli5.out
#SBATCH --error=eli5.err
#SBATCH --partition=studentbatch
#SBATCH --time=820
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
python -m bleurt.finetune  -init_bleurt_checkpoint=bleurt-base-512  -model_dir=eli5_bleurt-base-512_checkpoint -train_set=train_eli5.json  -dev_set=validation_eli5.json -batch_size=1 -eval_batch_size=1
