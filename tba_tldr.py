import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import ModelConfig
from src.tba_trainer_tldr import TBAConfigTLDR, TBATrainerTLDR
from src.utils import TRLParser, WandbLogModelConfig


from src.dist_utils import init_distributed_env


@dataclass
class ScriptArguments:
    output_global_parent_dir: str = field(default=None)
    dataset_name: str = field(default=None, metadata={"help": "the dataset name"})
    dataset_train_split: str = field(default="train", metadata={"help": "the name of the training set of the dataset"})
    dataset_test_split: str = field(default="test", metadata={"help": "the name of the training set of the dataset"})
    max_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})
    config: str = field(default=None, metadata={"help": "Path to the optional config file"})
    wandb_run_id: Optional[str] = field(default=None)


def prepare_dataset(dataset, tokenizer):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        input_ids = tokenizer(
            element["query"],
            padding=False,
        )["input_ids"]
        return {"input_ids": input_ids, "lengths": [len(ids) for ids in input_ids]}

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=multiprocessing.cpu_count(),
    )


if __name__ == "__main__":
    # prevent accelerate from joining all processes -- see init_distributed_env for details
    comm, comm_world_rank, comm_world_size, accelerator = init_distributed_env(
            accelerate_ranks=[0]
    )
    print(f'Before trainer created, reporting from rank {comm_world_rank} of {comm_world_size}', flush=True)
    parser = TRLParser((ScriptArguments, TBAConfigTLDR, ModelConfig))
    args, config, model_config = parser.parse_args_and_config()
        

    if args.output_global_parent_dir is not None:
        config.output_dir = os.path.join(args.output_global_parent_dir, config.output_dir)


    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, 
        num_labels=1
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)

    train_dataset = raw_datasets[args.dataset_train_split]
    eval_dataset = raw_datasets[args.dataset_test_split]

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    # filtering
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= args.max_length)
    eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= args.max_length)
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    ################
    # Training
    ################

    TrainerCls = TBATrainerTLDR

    trainer = TrainerCls(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbLogModelConfig(model_config)],
    )
    trainer.train()
                
    if not config.sanity_check and accelerator is not None:
        trainer.save_model(config.output_dir)
        trainer.generate_completions()