import os
import time
import argparse

from loguru import logger

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
                        default="t5-3b",
                        help="Model to perform ReLoRA fine tuning on")

    parser.add_argument("--dataset",
                        type=str,
                        default="GLUE",
                        help="The dataset to fine tune on")

    parser.add_argument("--subset",
                        type=str,
                        default="SST-2",
                        help="The subset of the dataset to fine tune on")

    args = parser.parse()
    return args

def main():
    args = parse_args()

    # log into file
    logger.remove()
    logger.add('relora.log', 'w')
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




if __name__ == "__main__":
    main()

