#!/bin/bash

num_processes=4
model_size=1b # can also be 2.8b or 410m
model_name_or_path=mnoukhov/pythia${model_size}-sft-tldr 
sft_model_path=mnoukhov/pythia${model_size}-sft-tldr 
reward_model_path=mnoukhov/pythia${model_size}-rm-tldr6.9b
if [ "$model_size" = "1b" ]; then
    local_rollout_forward_batch_size=20
elif [ "$model_size" = "2.8b" ]; then
    local_rollout_forward_batch_size=10
elif [ "$model_size" = "410m" ]; then
    local_rollout_forward_batch_size=40
fi
config=tba_pythia2.8b_tldr
    
n_searchers=$((num_processes - 1))
echo "Starting experiment with:
    - Number of searchers: ${n_searchers}
    - Model size: ${model_size}
    - Model name: ${model_name_or_path}
    - SFT model path: ${sft_model_path}
    - Reward model path: ${reward_model_path}
    - Local rollout forward batch size: ${local_rollout_forward_batch_size}
    - Config: ${config}"


PYTHON_CMD="""python tba_tldr.py --config configs/${config}.yml \
    --model_name_or_path ${model_name_or_path} \
    --sft_model_path ${sft_model_path} \
    --reward_model_path ${reward_model_path} \
    --local_rollout_forward_batch_size ${local_rollout_forward_batch_size} \
    --run_name tldr_model_size_${model_size}_run_0 \
    --output_dir tldr_model_size_${model_size}_run_0
"""

# Check if srun is available.
if command -v srun &> /dev/null; then
    # With srun, we don't need to set CUDA_VISIBLE_DEVICES manually.
    srun --gpus-per-task=1 -N 1 -n ${num_processes} bash -c "$PYTHON_CMD"
else
    # With mpirun, add a command to export CUDA_VISIBLE_DEVICES based on the local rank.
    mpirun -np ${num_processes} bash -c "export CUDA_VISIBLE_DEVICES=\$((OMPI_COMM_WORLD_LOCAL_RANK)); $PYTHON_CMD"
fi
