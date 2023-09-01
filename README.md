# ReLoRA
This repository contains fine-tuning code for GLUE benchmark. Both Full Fine Tuning and ReLoRA finetuning can be performed. The ReLoRA  code is from [ReLoRA](https://github.com/Guitaricet/gpt-neox/tree/relora/megatron/relora)

## To run ReLoRA on sst2:
```bash
python -u -m accelerate.commands.launch run_glue_no_trainer.py \
--task_name sst2 --model_name_or_path t5-base --relora \
--max_train_steps 4000 --reset_freq 2000 \
--learning_rate 2e-4 --r 32
```

## To run LoRA fine-tuning on sst2:
```bash
python -u -m accelerate.commands.launch run_glue_no_trainer.py \
--task_name sst2 --model_name_or_path t5-base --relora \
--max_train_steps 4000 --reset_freq 4000 \
--learning_rate 2e-4 --r 32
```

## To run full fine-tuning on sst2:
```bash
python -u -m accelerate.commands.launch run_glue_no_trainer.py \
--task_name sst2 --model_name_or_path t5-base \
--max_train_steps 4000 --learning_rate 2e-4
```
