#!/bin/bash
# Taken and modified from https://github.com/mnoukhov/async_rlhf
export PYTHONPATH="$PYTHONPATH:$(pwd)"

MODEL_PATH_ARG=$@

GOLD_MODEL=cleanrl/EleutherAI_pythia-6.9b-deduped__reward__tldr

echo evaluating model $MODEL_PATH_ARG

if [[ "$MODEL_PATH_ARG" == *"410m"* ]]; then
    REF_ARG="mnoukhov/pythia410m-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"1b"* ]]; then
    REF_ARG="mnoukhov/pythia1b-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"2.8b"* ]]; then
    REF_ARG="mnoukhov/pythia2.8b-sft-tldr"
else
    echo "output path doesn't contain one of model names"
    exit 1
fi

echo using base model $REF_ARG

echo using gold model $GOLD_MODEL

export CUDA_VISIBLE_DEVICES=0

python eval_tldr/generate_for_eval.py --config configs/generate_tldr.yml --model_name_or_path $MODEL_PATH_ARG

python eval_tldr/load_and_eval.py --config configs/evaluate_tldr.yml --model_name_or_path $MODEL_PATH_ARG --ref_model_name $REF_ARG --gold_model_name $GOLD_MODEL

python eval_tldr/eval_ppl.py --config configs/evaluate_tldr.yml --model_name_or_path $MODEL_PATH_ARG --ref_model_name $REF_ARG --gold_model_name $GOLD_MODEL