import os
import time
import argparse
import random
from functools import partial
from collections import namedtuple

import wandb
import numpy as np
from loguru import logger
import transformers
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_metric, DatasetDict
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, T5ForConditionalGeneration
from transformers import SchedulerType, get_scheduler, get_constant_schedule_with_warmup
from tqdm import tqdm
import evaluate
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from preprocessors import glue, stsb, string_to_float
from functional import wrap_with_ReLoRa, merge_and_reinit_functional
from optim import get_ragged_cosine_schedule, reset_optimizer
import templates

ARCH_TO_CLASS = {
    "encoder": AutoModelForMaskedLM,
    "decoder": AutoModelForCausalLM,
    "encoder-decoder": T5ForConditionalGeneration
}

task_to_keys = {
    "cola": ("sentence"),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence"),
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

def glue_preprocessing(data, benchmark_name):
    label_names = tasks_to_labels[benchmark_name]
    feature_keys = task_to_keys[benchmark_name]

    if benchmark_name == 'rte':
        premise = data['sentence1']
        hypothesis = data['sentence2']
    else:
        # create variables with the same name as the strings - will need it to fill in the template
        variables = task_to_keys.get(benchmark_name, ())
        variables = (variables,) if isinstance(variables, str) else variables
        locals().update({var: data[var] for var in variables})

        if 'premise' not in locals():
            premise = ''
        if 'hypothesis' not in locals():
            hypothesis = ''

    # find out how many labels are there
    num_labels = len(tasks_to_labels[benchmark_name])
    if num_labels == 3:
        options_ = f'A) {label_names[0]}\nB){label_names[1]}\nC){label_names[2]}' # You can modify the options as needed
        if data['label'] == 0:
            answer = 'A'
        elif data['label'] == 1:
            answer = 'B'
        elif data['label'] == 2:
            answer = 'C'
    elif num_labels == 2:
        options_ = f'A) {label_names[0]}\nB) {label_names[1]}' # You can modify the options as needed
        answer = 'A' if data['label'] == 0 else 'B' # Modify according to the label mapping
    prompt_template = templates.PATTERNS[benchmark_name][0][0] # extract first prompt and since its a tuple extract string
    filled_prompt = prompt_template.format(
        premise=premise,
        hypothesis=hypothesis,
        sentence=locals().get('sentence', ''),
        sentence1=locals().get('sentence1', ''),
        sentence2=locals().get('sentence2', ''),
        question=locals().get('question', ''),
        question1=locals().get('question1', ''),
        question2=locals().get('question2', ''),
        options_=options_,
    )

    template_filled = (filled_prompt, answer)
    return {'inputs': filled_prompt, 'targets': answer}


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
                        default=1,
                        help="Number of epochs to fine tune for")

    parser.add_argument("--eval_every",
                        type=int,
                        default=1000,
                        help="Number of steps between evaluations")

    parser.add_argument("--max_length",
                        type=int,
                        default=512,
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
                        default=20_000,
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

    parser.add_argument("--tokenizer",
                       type=str,
                       default=None,
                       help="Default tokenizer to use")

    parser.add_argument("--accumulation_steps",
                       type=int,
                       default=4,
                       help="Perform optimization every accumulation_steps mini batches")

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )


    args = parser.parse_args()
    return args

def eval(model, val_dataloader, global_step, args, tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    model = model.to(device=device, dtype=torch.bfloat16)
    metric_name = metric_mapping[args.task_name]
    with torch.no_grad():
        model.eval()
        all_predictions = []
        all_labels = []
        total_score = 0
        for batch in val_dataloader:
            # batch for decoder only models have additional parameters as inputs for generator
            if args.model_arch == "encoder_decoder":
                batch = batch.to(device)
                outputs = model(**batch)
            elif args.model_arch == "decoder":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # logits corresponding to </s> is ignored so just take first element
            #outputs.logist -> (batch_size, seq_len, vocab_size); seq len is 2 for label </s>
            if args.model_arch == "encoder_decoder":
                logits = outputs.logits[:, 0, :].argmax(dim=-1).cpu().numpy()
                labels = batch["labels"][:, 0].cpu().numpy()
                # implement the metric based on the task
                metric = metric_mapping[task_name]
                metric_function = evaluate.load(metric)
                score = metric_function.compute(predictions=all_predictions, references=all_labels)
                total_score += score
                logger.info(f'Task: {args.task_name}, Metric: {metric}, Score: {score}')
                wandb.log({"eval_loss": loss.item(), "step": global_step, "metric":score}, step=global_step)
            elif args.model_arch == "decoder":
                # becomes complicated for decoder model to get the metric using logits so use generate on the final model
                score = evaluate_glue_decoder(model, val_dataloader, args.task_name, tokenizer, args)
                total_score += score[metric_name]


        overall_score = total_score/len(val_dataloader)
        wandb.log({"eval_loss": loss.item(), metric_name: overall_score, "step": global_step}, step=global_step)
        return overall_score

def evaluate_glue_decoder(model, test_dataloader, task_name, tokenizer, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    # Evaluate the model on each GLUE task
    with torch.no_grad():
        model.eval()
        all_predictions = []
        all_labels = []
        #logger.info(f"Generated words")
        for batch_idx, batch in enumerate(test_dataloader):
            # do not need to add an extra dimension now that we have batch generate

            input_ids = batch["input_ids_generate"].to(device)
            attention_mask = batch["attention_mask_generate"].to(device)
            labels = batch["targets_generate"].to(device)
            max_token_length = 2
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            #output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_token_length, pad_token_id=pad_token_id, num_beams=3, early_stopping=True)
            # task is trivial now predict A, B, or C so no need of beams
            output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_token_length, pad_token_id=pad_token_id)
            # remove input ids from generation - need to be dynamic
            # input_ids are already padding to max length so remove them from the generated outputs
            generated_ids = output[:, args.max_length:]
            # you will need to iterate over each generated id to get the decoding
            for idx, generation in enumerate(generated_ids):
                output_text = tokenizer.decode(generation)

                # output_text should either be A B or C
                if 'A' in output_text:
                    label_index = 0
                elif 'B' in output_text:
                    label_index = 1
                elif 'C' in output_text:
                    label_index = 2
                else:
                    label_index = -1 #TODO fix me for sts-b which is a regression task


                decoded_labels = tokenizer.decode(labels[idx], skip_special_tokens=True)
                if 'A' in decoded_labels:
                    true_label = 0
                elif 'B' in decoded_labels:
                    true_label = 1
                elif 'C' in decoded_labels:
                    true_label = 2


                all_predictions.append(label_index)
                all_labels.append(true_label)
                # log first batch
                if batch_idx == 0:
                    logger.info(f"Actual Input {tokenizer.decode(input_ids[idx])}")
                    logger.info(f"Predicted text: {output_text} Actual label: {decoded_labels}")
                    logger.info(f"All predictions {all_predictions}")
                    logger.info(f"all labels {all_labels}")

        metric = metric_mapping[task_name]
        metric_function = evaluate.load(metric)
        score = metric_function.compute(predictions=all_predictions, references=all_labels)
        logger.info(f'Task: {task_name}, Metric: {metric}, Score: {score}')
    return score

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
        # lora -> reset freq large
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
        lr_scheduler = get_ragged_cosine_schedule(optimizer=optimizer, num_training_steps=args.num_training_steps, \
                                                  first_warmup_steps=args.first_warmup_steps, \
                                                  restart_warmup_steps = args.restart_warmup_steps, \
                                                  reset_freq=args.relora_frequency)


    if args.tokenizer is None:
        # load tokenizer correponding to the model
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, model_max_length=model.config.max_position_embeddings)
        #if args.tokenizer == "t5-base":
            #tokenizer.model_max_length = args.max_length

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # have left padding instead of right padding
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = model.to(device=device, dtype=torch.bfloat16)
    model = model.to(device=device, dtype=torch.bfloat16)

    #tokenizer = T5Tokenizer.from_pretrained(args.model)
    logger.info(f"Loaded model: {args.model}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {params}")

    dataset = load_dataset(args.dataset, args.task_name)

    if args.task_name == 'sts-b':
        #TODO fix me
        dataset = dataset.map(stsb, batched=True)
    else:
        glue_partial = partial(glue_preprocessing, benchmark_name=args.task_name)
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
        # Concatenate the inputs and targets
        inputs = examples['inputs'] + " " + examples['targets']
        tokenized_inputs = tokenizer(inputs, truncation=True, max_length=args.max_length, padding="max_length", return_tensors='pt')


        # Calculate the length of the inputs + ":" (including the space at the end)
        tokenized_inputs_no_padding = tokenizer(examples['inputs']+ " ")
        tokenized_inputs_with_target = tokenizer(examples['inputs'] + " " + examples['targets'], padding=False)
        len_inputs = len(tokenized_inputs_with_target['input_ids'])
        #logger.info(tokenized_inputs_with_colon)
        len_inputs_no_padding = len(tokenized_inputs_no_padding['input_ids'])

        # Clone the input_ids to create targets
        targets = tokenized_inputs['input_ids'].clone()

        # Set the tokens corresponding to the inputs and the colon to -100 in the targets
        #targets[:, :len_inputs_with_colon] = -100
        start_index = args.max_length - len_inputs - 1
        end_index = len_inputs_no_padding
        #logger.info(f"Inputs with targets {start_index}")
        #logger.info(f"Inputs without targets {end_index}")
        #logger.info(f"Start Index is {start_index}")
        #logger.info(f"End Index is {end_index}")
        targets[:, start_index:start_index+end_index] = -100  # left padding, change to above if right
        #logger.info(f"targets {targets}")

        # Pad tokens are also ignored in the loss
        targets[targets == tokenizer.pad_token_id] = -100

        # Create another set of inputs and targets to be used by the generate function
        inputs_generate = examples['inputs'] + " "
        inputs_generate = tokenizer(inputs_generate, add_special_tokens=False, truncation=True, max_length=args.max_length, padding="max_length", return_tensors='pt')
        targets_generate =  tokenizer(examples['targets'], truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt")

        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(0),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
            'labels': targets.squeeze(0),
            'input_ids_generate': inputs_generate['input_ids'].squeeze(0),
            'attention_mask_generate': inputs_generate['attention_mask'].squeeze(0),
            'targets_generate': targets_generate["input_ids"].squeeze(0)
        }

    # for validation do not tokenize the label
    def preprocess_function_decoder_val(examples):
        # separator ":" used to separate inputs from target
        inputs = examples['inputs'] + ":"
        targets = examples['targets']
        inputs = tokenizer(inputs, truncation=True, max_length=args.max_length, padding=False, return_tensors='pt')
        return {'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': examples["targets"]}

    old_columns = dataset['train'].column_names
    if args.model_arch == "encoder-decoder":
        tokenized_dataset = dataset.map(preprocess_function_encoder_decoder, batched=True, remove_columns=old_columns)
    elif args.model_arch == "decoder":
        tokenized_dataset = dataset.map(preprocess_function_decoder, remove_columns=old_columns)

    #do not use DataCollatorForLanguageModeling as this overwrites labels with the input ids,
    #however replacing pad tokens with -100 is not done in DataCollatorWithPadding, so do it before calling this
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # report metrics on val dataset as some of the datasets on glue do not have labels for the test dataset
    split_dataset = tokenized_dataset['train'].train_test_split(test_size=0.05, seed=args.fixed_seed_val)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    test_dataset = tokenized_dataset['validation']
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collator)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collator)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collator)
    logger.info(f"The length of the train dataloader is {len(train_dataloader)}")
    logger.info(f"The length of the validation dataloader is {len(val_dataloader)}")
    logger.info(f"The length of the test dataloader is {len(test_dataloader)}")

    if args.relora is False:
        total_steps = args.num_epochs * len(train_dataloader)
        #lr_scheduler =  OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps, pct_start=0.3)  # pct_start defines the fraction of cycle for warm-up
        #lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
        #                                                 num_warmup_steps=args.num_warmup_steps*args.accumulation_steps,
        #                                                )
        logger.info(f"Scheduler used is constant scheduler with warmup")

    # for now we do not need a custom collator as we are using model.generate with left padding and an entire batch
    """
    def custom_collator(batch_samples):
        sample = batch_samples[0]  # Since batch size is 1, get the only sample
        return {
            "input_ids": torch.tensor(sample['input_ids']),
            "attention_mask": torch.tensor(sample['attention_mask']),
            "labels": sample['labels'],
        }

    # for the val dataloader we do not need the inputs to have the targets and the targets need to be separate
    # so redo the tokenization
    if args.model_arch == "decoder":
        old_columns = dataset["validation"].column_names
        tokenized_dataset_test = dataset["validation"].map(preprocess_function_decoder_val, remove_columns=old_columns)
        test_dataloader = torch.utils.data.DataLoader(tokenized_dataset_test, batch_size=1, collate_fn=custom_collator)
    else:
        test_dataloader = torch.utils.data.DataLoader(tokenized_dataset_test, batch_size=args.batch_size, collate_fn=collator)
    """

    # training loop
    global_step = 0
    best_score = 0
    patience_counter = 0
    max_patience = 50
    for epoch in tqdm(range(args.num_epochs)):
        accumulated_loss = 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            global_step += 1
            model.train()
            optimizer.zero_grad()

            """
            if batch_idx == 0:
                logger.info(f"Input ids {tokenizer.decode(batch['input_ids'][0])}")
            """

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # Accumulate loss
            accumulated_loss += loss.item()
             # Backward pass (accumulate gradients)
            loss.backward()

            if (batch_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                #lr_scheduler.step()
                optimizer.zero_grad()

                # relora merge weights, reinit weights and reset optimizer
                #if global_step % args.relora_frequency == 0:
                    #merge_and_reinit_functional(model)
                    #reset_optimizer(optimizer, reset_params=params_to_update, pruning_amount=0.9)

                wandb.log(
                    {"train_loss": accumulated_loss / args.accumulation_steps,
                    "step": global_step,
                    "learning_rate": optimizer.param_groups[0]["lr"]},
                    step=global_step
                )

                # Reset accumulated loss
                #logger.info(f"optimization on {batch_idx}")
                accumulated_loss = 0.0

        logger.info(f"Computing the evaluation")
        current_score = eval(model, val_dataloader, global_step, args, tokenizer)
        if current_score >= best_score:
            best_score = current_score
            logger.info(f"Best Model Score is {best_score}")
            model.save_pretrained(f"best_model_{args.task_name}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience and current_score != 0:
                print(f"No improvement after", {max_patience}, "epochs. Stopping training.")
                break

            #TODO remove this later on for debugging purposes only
    logger.info("Finished fine tuning")

    # evaluate on the validation dataset
    if args.model_arch == "decoder":
        # load best model
        model = ARCH_TO_CLASS[args.model_arch].from_pretrained(f"best_model_{args.task_name}")
        model = model.to(device=device, dtype=torch.bfloat16)
        score = evaluate_glue_decoder(model, test_dataloader, args.task_name, tokenizer, args)
        logger.info(f"The best metric on the validation dataset is {score}")

    wandb.finish()

if __name__ == "__main__":
    main()