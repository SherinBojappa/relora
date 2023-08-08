# ReLoRA
This repository contains fine-tuning code for GLUE benchmark. Both Full Fine Tuning and ReLoRA finetuning can be performed. The ReLoRA  code is from [ReLoRA](https://github.com/Guitaricet/gpt-neox/tree/relora/megatron/relora)

## Commands
### To run Full fine-tuning on sst2:
```python sft.py --num_epochs 1 --model model_20001/ --batch_size 64 --model_arch decoder --tokenizer t5-base --dataset sst2```

### To run ReLoRA fine-tuning on sst2:
```python sft.py --num_epochs 1 --model model_20001/ --batch_size 64 --model_arch decoder --tokenizer t5-base --relora --dataset sst2```
