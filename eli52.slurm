#! /bin/sh
#SBATCH --job-name=eli52
#SBATCH --output=eli52.out
#SBATCH --error=eli52.err
#SBATCH --partition=studentbatch
#SBATCH --time=4320
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
python -m bleurt.finetune  -init_bleurt_checkpoint=bleurt-base-512  -model_dir=eli52_bleurt-base-512_passover3 -train_set=train_eli5.json  -dev_set=validation_eli5.json -batch_size=4 -eval_batch_size=4 -num_train_steps=402222 -save_checkpoints_steps=50000
