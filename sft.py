import os
import time
import argparse

from loguru import logger
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, load_metric
from transformers.data.data_collator import DataCollatorWithPadding
from tqdm import tqdm

from preprocessors import glue, stsb, string_to_float

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
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
    'sts-b': [],
    'mnli': ['entailment', 'neutral', 'contradiction'],
    'qnli': ['entailment', 'not_entailment'],
    'rte': ['entailment', 'not_entailment'],
    'wnli': ['not_entailment', 'entailment']
}


def arg_parse():
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

    parser.add_argument("--dataset",
                        type=str,
                        default="GLUE",
                        help="The dataset to fine tune on")

    parser.add_argument("--task_name",
                        type=str,
                        default="SST-2",
                        help="The subset of the dataset to fine tune on")

    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
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

    args = parser.parse()
    return args


def preprocess_function(examples):
    if setence2_key is None:
        return tokenizer(examples[setence1_key], truncation=True)
    return tokenizer(examples[setence1_key], examples[setence2_key], truncation=True)

def evaluate(dataloader, task_name=None):
    with torch.no_grad():
        model.eval()
        all_predictions = []
        all_labels = []
        for batch in val_dataloader:
            batch = batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_predictions.extend(logits)
            all_labels.extend(labels)
            wandb.log({"eval_loss": loss.item(), "step": global_step})

            # implement the metric based on the task
            metrics = metric_mapping[task_name]
            metric_functions = {metric: load_metric('glue', task_name, metric) for metric in metrics}
            # only one metric
            score = metric_function.compute(predictions=all_predictions, references=all_labels)
            logger.info(f'Task: {task}, Metric: {metric}, Score: {score}')

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
            metric_functions = {metric: load_metric('glue', task, metric) for metric in metrics}

            collator = DataCollatorWithPadding(tokenizer=tokenizer)
            tokenized_dataset = dataset.map(preprocess_function, batched=True)

            # Create a PyTorch data loader for the validation set
            test_dataloader = torch.utils.data.DataLoader(tokenized_dataset['test'], batch_size=args.batch_size, collate_fn=collator)

            # Run the model on the validation set and compute the metrics
            model.eval()
            all_predictions = []
            all_labels = []
            for batch in val_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)
                    logits = outputs.logits.argmax(dim=-1).cpu().numpy()
                    labels = batch["labels"].cpu().numpy()
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
    logger.add(f"relora_{args.task_name}.log", 'w')
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
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    logger.info(f"Loaded model: {args.model}")

    dataset = load_dataset(args.dataset, args.task_name)
    setence1_key, setence2_key = task_to_keys[args.task_name]

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    num_labels = 3 if args.task_name.startswith("mnli") else 1 if args.task_name.startswith("stsb") else 2
    metric_name = "pearson" if args.task_name.startswith("stsb") else "matthews_correlation" if args.task_name.startswith("cola") else "accuracy"


    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(tokenized_dataset["train"], batch_size=args.batch_size, collate_fn=collator)
    val_dataloader = torch.utils.data.DataLoader(tokenized_dataset["validation"], batch_size=args.batch_size, collate_fn=collator)
    #test_dataloader = torch.utils.data.DataLoader(tokenzied_dataset["test"], batch_size=args.batch_size, collate_fn=collator)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # training loop
    global_step = 0
    for epoch in tqdm(epoch):
        for batch in train_dataloader:
            global_step += 1
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            wandb.log(
                {"train_loss": loss.item(),
                "step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"]}
            )

        if global_step % args.eval_every == 0:
            evaluate(val_dataloader)

    logger.info("Finished fine tuning")

    # evaluate all glue datasets on the fine tuned model
    evaluate_glue(test_dataloader, args.task_name)
    wandb.finish()

if __name__ == "__main__":
    main()