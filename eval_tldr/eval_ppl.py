# Taken and modified from https://github.com/mnoukhov/async_rlhf
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

import wandb
from src.utils import TRLParser


@dataclass
class EvalScriptArguments:
    model_name_or_path: str = None
    ref_model_name: Optional[str] = None
    sanity_check: Optional[bool] = False
    wandb_run_id: Optional[str] = field(default=None)
    gold_model_name: Optional[str] = field(default="EleutherAI/pythia-410m", metadata={"help": "the model name"})
    gold_model_revision: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default="auto")
    batch_size: Optional[int] = field(default=16)
    gold_tokenizer_name: Optional[str] = field(default=None, metadata={"help": "the tokenizer name"})
    dataset_path: str = None


def evaluate(args, all_reference, all_generations, all_episodes, log_to_wandb=False):
    state = PartialState()
    torch_dtype = args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype)
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map={"": state.process_index},
    )

    tokenizer_name = args.gold_tokenizer_name if args.gold_tokenizer_name is not None else args.gold_model_name

    ppl_pipeline = pipeline(
        task="perplexity",
        model=args.ref_model_name,
        model_kwargs=model_kwargs,
    )

    step = 0
    for step_str, all_query_response in all_generations.items():
        gen_rewards = []
        gen_ppls = []
        episode = all_episodes[step_str]
        with state.split_between_processes(all_query_response) as query_response:
            for out in tqdm(
                ppl_pipeline(query_response, prompt_template="TL;DR:", batch_size=args.batch_size),
                total=len(query_response),
                disable=not state.is_local_main_process,
                desc=f"PPL Step {step_str}",
            ):
                gen_ppls += [r["ppl"] for r in out]

        gen_ppls = gather_object(gen_ppls)
        gen_ppls = np.array(gen_ppls)
        mean_ppl = gen_ppls.mean().item()

        if step_str.startswith("checkpoint-"):
            step_str = step_str.removeprefix("checkpoint-")

        if step_str.isdigit():
            step = int(step_str)
        else:
            state.print(f"Warning step name {step_str} is not an integer")
            step = step + 1

        if log_to_wandb and state.is_main_process:
            num_samples = 32
            wandb.log(
                {
                    "gold/ppl": mean_ppl
                },
            )

        state.print(f"step {step}: ppl {mean_ppl}")


if __name__ == "__main__":
    parser = TRLParser([EvalScriptArguments])
    args = parser.parse_args_and_config()[0]

    if args.dataset_path is not None:
        generated_dataset_path = args.dataset_path
    else:
        generated_dataset_path = os.path.join(args.model_name_or_path, "_generations")

    dataset = load_from_disk(generated_dataset_path)

    with open(os.path.join(generated_dataset_path, "trainer_states.json"), "r") as f:
        trainer_states = json.load(f)

    prompts = dataset["query"]
    reference = KeyDataset(dataset, "query_reference_response")

    generations_cols = [name for name in dataset.column_names if name.startswith("generation")]
    generations = {}
    episodes = {}
    for col_name in generations_cols:
        # column name should be generations_{step name}
        checkpoint_name = col_name.split("_")[1]
        generations[checkpoint_name] = KeyDataset(dataset, col_name)
        if "episode" in trainer_states[checkpoint_name]:
            eps = trainer_states[checkpoint_name]["episode"]
        elif "dpo" in args.model_name_or_path:
            # assume offline dpo, which uses a pref dataset of 92858, although this is slightly off in practice
            eps = round(trainer_states[checkpoint_name]["epoch"] * 92858)
        else:
            # for sft and others
            eps = 0
        episodes[checkpoint_name] = eps

    if args.sanity_check:
        args.wandb_run_id = None
        first_ckpt = next(iter(generations.keys()))
        generations = {first_ckpt: generations[first_ckpt]}
        generations[first_ckpt].dataset = generations[first_ckpt].dataset.select(range(100))
        reference.dataset = reference.dataset.select(range(100))

    if args.wandb_run_id == "snow":
        # remove extra / at end
        normpath = os.path.normpath(args.model_name_or_path)
        path_parts = normpath.split("/")
        config_name = path_parts[-1]
        run_id = path_parts[-2]
        args.wandb_run_id = run_id + "_" + config_name

    log_to_wandb = args.wandb_run_id is not None
    state = PartialState()
    if 'short_' in args.wandb_run_id:
        args.wandb_run_id = args.wandb_run_id.replace('short_','S')
    if log_to_wandb and state.is_main_process:
        wandb.init(id=args.wandb_run_id, resume="allow")
        print(f"Logging to WandB {args.wandb_run_id}")

    evaluate(args, reference, generations, episodes, log_to_wandb)
