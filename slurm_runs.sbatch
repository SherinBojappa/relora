#!/bin/bash
#SBATCH --job-name="relora"
#SBATCH --partition=g40x
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --hint=nomultithread        # We get physical cores not logical
#SBATCH --cpus-per-task=12          # Number of cores per tasks
#SBATCH --gres=gpu:1                # Number of gpus
#SBATCH --output=/fsx/vlialin/slurmlogs/%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=/fsx/vlialin/slurmlogs/%j.out  # Set this dir where you want slurm outs to go
#SBATCH --exclusive
#SBATCH --account eleuther
#SBATCH --time=23:59:59
#SBATCH --mail-type=ALL

source /fsx/vlialin/finetuning_relora/.venv/bin/activate
cd /fsx/vlialin/finetuning_relora

export WANDB_WATCH=false
export WANDB_DIR=/fsx/vlialin/wandb/wandb_dir
export WANDB_CACHE_DIR="/fsx/vlialin/wandb/.cache"
export TMPDIR=/fsx/vlialin/tmp
source /admin/home-vlialin/sbatch_exports.sh  # WANDB_API_KEY
export TRANSFORMERS_CACHE=/fsx/vlialin/cache/transformers_cache
export HF_DATASETS_CACHE=/fsx/vlialin/cache/datasets_cache

python -u -m accelerate.commands.launch run_glue_no_trainer.py \
    --task_name qqp --model_name_or_path t5-base --relora \
    --max_train_steps 4000 --reset_freq 2000 \
    --learning_rate 2e-4 --r 32

python -u -m accelerate.commands.launch run_glue_no_trainer.py \
    --task_name qqp --model_name_or_path t5-base --relora \
    --max_train_steps 4000 --reset_freq 2000 \
    --learning_rate 1e-4 --r 32

python -u -m accelerate.commands.launch run_glue_no_trainer.py \
    --task_name qqp --model_name_or_path t5-base --relora \
    --max_train_steps 4000 --reset_freq 2000 \
    --learning_rate 5e-4 --r 32

python -u -m accelerate.commands.launch run_glue_no_trainer.py \
    --task_name mnli --model_name_or_path t5-base --relora \
    --max_train_steps 4000 --reset_freq 2000 \
    --learning_rate 2e-4 --r 32

python -u -m accelerate.commands.launch run_glue_no_trainer.py \
    --task_name mnli --model_name_or_path t5-base --relora \
    --max_train_steps 4000 --reset_freq 2000 \
    --learning_rate 1e-4 --r 32

python -u -m accelerate.commands.launch run_glue_no_trainer.py \
    --task_name mnli --model_name_or_path t5-base --relora \
    --max_train_steps 4000 --reset_freq 2000 \
    --learning_rate 5e-4 --r 32