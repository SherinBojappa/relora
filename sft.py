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
from tqdm import tqdm
import evaluate
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

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
            max_token_length = 7
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_token_length, pad_token_id=pad_token_id, num_beams=3, early_stopping=True)
            # remove input ids from generation - need to be dynamic
            # input_ids are already padding to max length so remove them from the generated outputs
            generated_ids = output[:, args.max_length:]
            # you will need to iterate over each generated id to get the decoding
            for idx, generation in enumerate(generated_ids):
                output_text = tokenizer.decode(generation)
                # ideally output_text should be the same as the first word since we have trained the model with it
                # so ignore first_word search, if it is not we may need to add back the first word search
                # extract the first word as you get something like positive :positive :
                generated_strings = output_text.strip().lower().split(" ")

                decoded_labels = tokenizer.decode(labels[idx], skip_special_tokens=True)
                label_index = -1
                # you will always have one decoded_label even for strings like not_entailment
                if decoded_labels in generated_strings:
                    label_index = tasks_to_labels[task_name].index(decoded_labels)

                all_predictions.append(label_index)
                # log first batch
                if batch_idx == 0:
                    logger.info(f"Actual Input {tokenizer.decode(input_ids[idx])}")
                    logger.info(f"Predicted text: {output_text} Actual label: {decoded_labels}")
                    logger.info(f"All predictions {all_predictions}")
                    logger.info(f"all labels {all_labels}")
                all_labels.append(tasks_to_labels[task_name].index(decoded_labels))

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

    # hack to get away with mismatched size embedding layer and tokenizer vocab size only for t5-base and llama model
    """
    if args.tokenizer == "t5-base":
        new_vocab_size = tokenizer.vocab_size
        #LlamaForCausalLM class doesn't have a method named resize_token_embeddings, so manually copy over embeddings
        #model.resize.token_embeddings(new_vocab_size)
        # Get the current embedding layer
        old_embedding = model.model.embed_tokens
        old_embedding_size = old_embedding.weight.size(0)
        # Create a new embedding layer with the new vocabulary size
        embedding_dim = old_embedding.weight.size(1)
        new_embedding = nn.Embedding(new_vocab_size, embedding_dim)
        # Copy the existing weights from the old embedding layer
        new_embedding.weight.data[:old_embedding_size] = old_embedding.weight.data
        # Replace the old embedding layer with the new one
        model.model.embed_tokens = new_embedding
        assert model.model.embed_tokens.weight.size()[0] == tokenizer.vocab_size
        logger.info(f"Resized the embedding layer of the model to match with the tokenizer")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = model.to(device=device, dtype=torch.bfloat16)
    model = model.to(device=device, dtype=torch.bfloat16)

    #tokenizer = T5Tokenizer.from_pretrained(args.model)
    logger.info(f"Loaded model: {args.model}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {params}")

    dataset = load_dataset(args.dataset, args.task_name)
    # TODO - remove this code that tests on very few samples
    #smaller_dataset = DatasetDict()
    #for split in dataset.keys():
    #    subset_size = 100
    #    smaller_dataset[split] = dataset[split].shuffle(seed=42).select(range(subset_size))
    #dataset = smaller_dataset

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
        # add a separator between inputs and targets for the model to learn when to predict the targets
        inputs = examples['inputs'] + ":" + examples['targets']
        inputs = tokenizer(inputs, truncation=True, max_length=args.max_length, padding="max_length", return_tensors='pt')
        # ignore the mask for now so now you have all the next word predictions in your loss, if including the mask
        # you need to account for labels that are greater than 1 token long

        # Create a mask where the elements corresponding to example['inputs'] are True
        #mask = torch.full((len(inputs['input_ids'][0]),), fill_value=False, dtype=torch.bool)

        # there is a space in the end added to tokenizer(examples['inputs'])
        # it isnt present in tokenizer(examples['inputs']+examples['targets'])
        # hence subtract -1
        #print(len(tokenizer(examples['inputs'])['input_ids'])) # 14
        #len_inputs = len(tokenizer(examples['inputs']))
        #mask[:len_inputs] = True

        targets = inputs['input_ids'].clone()

        # pad tokens are ignored in the loss
        targets[targets == tokenizer.pad_token_id] = -100

        # create another set of inputs and targets to be used by the generate function
        inputs_generate = examples['inputs'] + ":"
        inputs_generate = tokenizer(inputs_generate, add_special_tokens=False, truncation=True, max_length=args.max_length, padding="max_length", return_tensors='pt')
        targets_generate =  tokenizer(examples['targets'], truncation=True, max_length=args.max_length, padding="max_length", return_tensors="pt")
        # inputs are also ignored in the loss
        #targets[mask.unsqueeze(0)] = -100

        return {'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': targets.squeeze(0),
                'input_ids_generate': inputs_generate['input_ids'].squeeze(0),
                'attention_mask_generate': inputs_generate['attention_mask'].squeeze(0),
                'targets_generate': targets_generate["input_ids"].squeeze(0)}

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
        lr_scheduler =  OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps, pct_start=0.3)  # pct_start defines the fraction of cycle for warm-up
        logger.info("One Cycle LR used")

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
    max_patience = 2
    for epoch in tqdm(range(args.num_epochs)):
        for batch in train_dataloader:
            global_step += 1
            model.train()
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

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

        logger.info(f"Computing the evaluation")
        current_score = eval(model, val_dataloader, global_step, args, tokenizer)
        if current_score >= best_score:
            best_score = current_score
            logger.info(f"Best Model Score is {best_score}")
            model.save_pretrained(f"best_model_{args.task_name}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
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