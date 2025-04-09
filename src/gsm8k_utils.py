import re
import torch
import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, GenerationConfig
from torch.utils.data import DataLoader
from trl.trainer.utils import generate

# Regex to find numbers
FIND_NUMBERS_REGEX = re.compile(r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)")
NO_NUMBER_FOUND_NUMBER = -999999.222222

def parse_number(text):
    """Strip whitespace/commas and convert a string to float (or return None if it fails)."""
    try:
        return float(text.strip().replace(',', '').lower())
    except Exception:
        return NO_NUMBER_FOUND_NUMBER

def extract_prediction(text):
    numbers = FIND_NUMBERS_REGEX.findall(text.replace(",", ""))
    return parse_number(numbers[-1]) if numbers else NO_NUMBER_FOUND_NUMBER

def format_and_tokenize(example, tokenizer):
    """
    Format the prompt and tokenize it.
    Also extract the gold answer using the same extraction logic.
    """
    prompt = f"[MATH TASK] Problem:\n{example['question']}\n\nSolution:\n"
    tokens = tokenizer(prompt, padding=False, return_attention_mask=True)
    gold = extract_prediction(example["answer"])
    assert gold is not NO_NUMBER_FOUND_NUMBER, FIND_NUMBERS_REGEX.findall(example["answer"].replace(",", ""))
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "lengths": len(tokens["input_ids"]),
        "response_ids": gold,
    }

def prepare_dataset(dataset, tokenizer):
    """Map tokenization over the dataset using all available CPU cores."""
    return dataset.map(
        lambda ex: format_and_tokenize(ex, tokenizer),
        remove_columns=dataset.column_names,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
    )

def evaluate(model, eval_dataloader, generation_config, tokenizer):
    correct, total = 0, 0
    for batch in eval_dataloader:
        input_ids = batch["input_ids"].cuda()
        golds = batch["response_ids"]
        query_response, _ = generate(
            model,
            input_ids,
            tokenizer.pad_token_id,
            generation_config,
        )
        context_length = input_ids.shape[1]
        response = query_response[:, context_length:]
        
        pred_answer = tokenizer.batch_decode(response)

        for i, text in enumerate(pred_answer):
            pred = extract_prediction(text)
            is_correct = (pred == golds[i]) if (pred is not None and golds[i] is not None) else False
            correct += is_correct
            total += 1

        print(f"Intermediate Accuracy: {correct / total * 100:.2f}% ({correct}/{total} correct)")
    return correct, total
            
def main(model_name):
    # --- Load dataset and model ---
    gsm8k = load_dataset("openai/gsm8k", "main")
    eval_dataset = gsm8k["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.bfloat16)
    model.eval().cuda()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Pre-tokenize the evaluation dataset
    eval_dataset_prepared = prepare_dataset(eval_dataset, tokenizer)

    # --- Create DataLoader for batching ---
    collator = DataCollatorWithPadding(tokenizer)
    eval_dataloader = DataLoader(
        eval_dataset_prepared,
        batch_size=64,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    # --- Evaluation loop ---
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=[tokenizer.eos_token_id]
        #temperature=0.35
    )
    correct, total = evaluate(model, eval_dataloader, generation_config, tokenizer)
    accuracy = correct / total * 100
    print(f"Final Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model on GSM8K dataset')
    parser.add_argument('--model_name', type=str, 
                       default="realtreetune/rho-1b-sft-GSM8K",
                       help='Name or path of the model to evaluate')
    args = parser.parse_args()

    main(args.model_name)