import os
import time
import argparse
import random
from functools import partial

import wandb
import numpy as np
from loguru import logger
import transformers
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_metric
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, T5ForConditionalGeneration
from tqdm import tqdm
import evaluate
import torch

from preprocessors import glue, stsb, string_to_float
from functional import wrap_with_ReLoRa, merge_and_reinit_functional
from optim import get_ragged_cosine_schedule, reset_optimizer

ARCH_TO_CLASS = {
    "encoder": AutoModelForMaskedLM,
    "decoder": AutoModelForCausalLM,
    "encoder-decoder": T5ForConditionalGeneration
}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "sts-b": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Define a dictionary to map GLUE tasks to their metrics
metric_mapping = {
    'cola': 'matthews_correlation',
    'sst2': 'accuracy',
    'mrpc': 'accuracy',
    'qqp': 'accuracy',
    'sts-b': 'spearmanr',
    'mnli': 'accuracy',
    'qnli': 'accuracy',
    'rte': 'accuracy',
    'wnli': 'accuracy'
}

tasks_to_labels = {
    'cola': ['unacceptable', 'acceptable'],
    'sst2': ['negative', 'positive'],
    'mrpc': ['not_equivalent', 'equivalent'],
    'qqp': ['not_duplicate', 'duplicate'],
    # processed differently as this is a regression task
    'sts-b': [],
    'mnli': ['entailment', 'neutral', 'contradiction'],
    'qnli': ['entailment', 'not_entailment'],
    'rte': ['entailment', 'not_entailment'],
    'wnli': ['not_entailment', 'entailment']
}


def parse_args():
    parser = argparse.ArgumentParser("Supervised fine tuning using ReLoRA")
    parser.add_argument("--fixed_seed_val",
                        type=int,
                        default=1,
                        help="Seed to initialize the current run")

    parser.add_argument("--wandb_project",
                        type=str,
                        default="relora_sft",
                        help="Name of the wandb project")

    parser.add_argument("--model",
                        type=str,
                        #default="t5-3b",
                        default="t5-small",
                        help="Model to perform ReLoRA fine tuning on")

    parser.add_argument('--model_arch',
                        default=None,
                        type=str,
                        help='Model architecture. Choose from "encoder", "decoder", "encoder-decoder"')

    parser.add_argument("--dataset",
                        type=str,
                        default="glue",
                        help="The dataset to fine tune on")

    parser.add_argument("--task_name",
                        type=str,
                        default="sst2",
                        help="The subset of the dataset to fine tune on")

    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="batch size for fine tuning")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-5,
                        help="learning rate for fine tuning")

    parser.add_argument("--num_epochs",
                        type=int,
                        default=3,
                        help="Number of epochs to fine tune for")

    parser.add_argument("--eval_every",
                        type=int,
                        default=100,
                        help="Number of steps between evaluations")

    parser.add_argument("--max_length",
                        type=int,
                        default=128,
                        help="Maximum length of the input sequence")

    parser.add_argument("--relora",
                        action="store_true",
                        default=False,
                        required=False,
                        help="Whether to use ReLoRA or not")

    parser.add_argument("--r",
                        type=int,
                        default=128,
                        help="rank used in ReLoRA")

    parser.add_argument("--relora_frequency",
                        type=int,
                        default=100,
                        help="Frequency of ReLoRA")

    parser.add_argument("--num_training_steps",
                       type=int,
                       default=200,
                       help="Number of training steps to perform")

    parser.add_argument("--first_warmup_steps",
                       type=int,
                       default=50,
                       help="First warmup steps")

    parser.add_argument("--restart_warmup_steps",
                       type=int,
                       default=10,
                       help="Number of restart warmup steps to perform")


    args = parser.parse_args()
    return args

def eval(model, val_dataloader, global_step, task_name=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        model.eval()
        all_predictions = []
        all_labels = []
        for batch in val_dataloader:
            batch = batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            # logits corresponding to </s> is ignored so just take first element
            #outputs.logist -> (batch_size, seq_len, vocab_size); seq len is 2 for label </s>
            logits = outputs.logits[:, 0, :].argmax(dim=-1).cpu().numpy()

            labels = batch["labels"][:, 0].cpu().numpy()
            all_predictions.extend(logits)
            all_labels.extend(labels)
            # implement the metric based on the task
            metric = metric_mapping[task_name]
            metric_function = evaluate.load(metric)
            score = metric_function.compute(predictions=all_predictions, references=all_labels)
            logger.info(f'Task: {task_name}, Metric: {metric}, Score: {score}')
            wandb.log({"eval_loss": loss.item(), "step": global_step, "metric":score}, step=global_step)


def evaluate_glue(tokenizer, task_name=None):
    # Evaluate the model on each GLUE task
    with torch.no_grad():
        model.eval()
        for task in metric_mapping.keys():
            # Load the dataset and metric
            dataset = load_dataset('glue', task)
            metrics = metric_mapping[task]
            if not isinstance(metrics, list):
                metrics = [metrics]
            metric_functions = {metric: evaluate.load(metric) for metric in metrics}

            collator = DataCollatorWithPadding(tokenizer=tokenizer)
            # map label id to label string
            if args.task_name == 'sts-b':
                dataset = dataset.map(stsb, batched=True)
            else:
                glue_partial = partial(glue, benchmark_name=args.task_name, label_names=tasks_to_labels[args.task_name])
                dataset = dataset.map(glue_partial, batched=True)

            tokenized_dataset = dataset.map(preprocess_function, batched=True)

            # Create a PyTorch data loader for the validation set
            test_dataloader = torch.utils.data.DataLoader(tokenized_dataset['validation'], batch_size=args.batch_size, collate_fn=collator)

            # Run the model on the validation set and compute the metrics
            model.eval()
            all_predictions = []
            all_labels = []
            for batch in val_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)
                    # logits corresponding to </s> is ignored so just take first element
                    #outputs.logist -> (batch_size, seq_len, vocab_size); seq len is 2 for label </s>
                    logits = outputs.logits[:, 0, :].argmax(dim=-1).cpu().numpy()
                    labels = batch["labels"][:, 0].cpu().numpy()
                    all_predictions.extend(logits)
                    all_labels.extend(labels)

            # Compute and print the metrics
            for metric, metric_function in metric_functions.items():
                score = metric_function.compute(predictions=all_predictions, references=all_labels)
                logger.info(f'Task: {task}, Metric: {metric}, Score: {score}')

def main():
    args = parse_args()

    # log into file
    logger.remove()
    logger.add(f"relora_{args.task_name}.log", mode='w')
    logger.info(f"The args are: {args}")

    # fix seed
    torch.manual_seed(args.fixed_seed_val)
    random.seed(args.fixed_seed_val)
    np.random.seed(args.fixed_seed_val)
    transformers.set_seed(args.fixed_seed_val)

    # wandb init
    wandb.init(project=args.wandb_project, config=args)
    args.run_name = wandb.run.name

    logger.info(f"Initialized wandb with run name: {args.run_name}")

    # load pre-trained model
    model = ARCH_TO_CLASS[args.model_arch].from_pretrained(args.model)
    #model = T5ForConditionalGeneration.from_pretrained(args.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # relora
    if args.relora:
        model = wrap_with_ReLoRa(model=model, r=args.r)
        logger.info(f"Relora with rank {args.r} is used")
        params_to_update = [param for param in model.parameters() if param.requires_grad]
        #params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params = sum(p.numel() for p in params_to_update)
        logger.info(f"Number of trainable parameters: {params}")
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
        lr_scheduler = get_ragged_cosine_schedule(optimizer=optimizer, num_training_steps=args.num_training_steps, \
                                                  first_warmup_steps=args.first_warmup_steps, \
                                                  restart_warmup_steps = args.restart_warmup_steps, \
                                                  reset_freq=args.relora_frequency)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = model.to(device=device, dtype=torch.bfloat16)
    model = model.to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #tokenizer = T5Tokenizer.from_pretrained(args.model)
    logger.info(f"Loaded model: {args.model}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {params}")

    dataset = load_dataset(args.dataset, args.task_name)
    # map label id to label string
    #if args.model_arch == "t5-3b":
    if args.task_name == 'sts-b':
        dataset = dataset.map(stsb, batched=True)
    else:
        glue_partial = partial(glue, benchmark_name=args.task_name, label_names=tasks_to_labels[args.task_name])
        column_names = dataset['train'].column_names
        dataset = dataset.map(glue_partial, remove_columns=column_names)

    def preprocess_function_encoder_decoder(examples):
        inputs = examples['inputs']
        targets = examples['targets']
        inputs = tokenizer(inputs, truncation=True, max_length=args.max_length, padding=True, return_tensors='pt')
        targets = tokenizer(targets, truncation=True, max_length=args.max_length, padding=True, return_tensors='pt')
        # pad tokens are ignored in the loss
        targets[targets == tokenizer.pad_token_id] = -100
        return {'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask'],
                'labels': targets['input_ids']}

    def preprocess_function_decoder(examples):
        inputs = examples['inputs'] + examples['targets']
        inputs = tokenizer(inputs, truncation=True, max_length=args.max_length, padding="max_length", return_tensors='pt')
        # Create a mask where the elements corresponding to example['inputs'] are True
        mask = torch.full((len(inputs['input_ids'][0]),), fill_value=False, dtype=torch.bool)

        # there is a space in the end added to tokenizer(examples['inputs'])
        # it isnt present in tokenizer(examples['inputs']+examples['targets'])
        # hence subtract -1
        #print(len(tokenizer(examples['inputs'])['input_ids'])) # 14
        mask[:len(tokenizer(examples['inputs'])['input_ids'])-1] = True

        targets = inputs['input_ids'].clone()

        # pad tokens are ignored in the loss
        targets[targets == tokenizer.pad_token_id] = -100

        # inputs are also ignored in the loss
        targets[mask.unsqueeze(0)] = -100

        return {'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': targets.squeeze(0)}

    old_columns = dataset['train'].column_names
    if args.model_arch == "encoder-decoder":
        tokenized_dataset = dataset.map(preprocess_function_encoder_decoder, batched=True, remove_columns=old_columns)
    elif args.model_arch == "decoder":
        tokenized_dataset = dataset.map(preprocess_function_decoder, remove_columns=old_columns)

    #import pdb; pdb.set_trace()
    #collator = DataCollatorWithPadding(tokenizer=tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # report metrics on val dataset as some of the datasets on glue do not have labels for the test dataset
    split_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2, seed=args.fixed_seed_val)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collator)


    # training loop
    global_step = 0
    for epoch in tqdm(range(args.num_epochs)):
        for batch in train_dataloader:
            global_step += 1
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            if args.relora:
                lr_scheduler.step()
            else:
                optimizer.step()

            # relora merge weights, reinit weights and reset optimizer
            if global_step % args.relora_frequency == 0:
                merge_and_reinit_functional(model)
                reset_optimizer(optimizer, reset_params=params_to_update, pruning_amount=0.9)

            wandb.log(
                {"train_loss": loss.item(),
                "step": global_step,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"]},
                step=global_step
            )

            if global_step % args.eval_every == 0:
                logger.info(f"Computing the evaluation")
                eval(model, val_dataloader, global_step, args.task_name)

    logger.info("Finished fine tuning")

    # evaluate all glue datasets on the fine tuned model
    #evaluate_glue(test_dataloader, args.task_name)
    wandb.finish()

if __name__ == "__main__":
    main()