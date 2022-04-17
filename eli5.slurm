#! /bin/sh
#SBATCH --job-name=eli5
#SBATCH --output=eli5.out
#SBATCH --error=eli5.err
#SBATCH --partition=studentbatch
#SBATCH --time=4320
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
export HF_DATASETS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/home/joberant/NLP_2122/dorcoh4/.cache/huggingface/"
python -m bleurt.finetune  -init_bleurt_checkpoint=eli5_bleurt-base-512_checkpoint_steps/export/bleurt_best/1647022069  -model_dir=eli5_bleurt-base-512_passover1 -train_set=shuf_train_eli5.json  -dev_set=shuf_validation_eli5.json -batch_size=4 -eval_batch_size=4 -num_train_steps=804444 -save_checkpoints_steps=402222
