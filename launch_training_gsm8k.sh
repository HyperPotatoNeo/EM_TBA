#!/bin/bash
#conda activate tba
#cd EM_TBA

WR=0.0714 # corresponds to 0.05 for 1000 steps, but we do 700 steps because acc doesn't change much from 700 to 1000
episodes=98000 # 98000 steps for 700 episodes at batch size 140
WSD_decay_steps=600
WSD_stable_steps=350
kl_coef=0.014
config=tba_qwen_gsm8k
num_processes=4
n_searchers=$((num_processes - 1))

echo "Starting experiment with:
    - Number of searchers: ${n_searchers}
    - Warmup ratio: ${WR}
    - WSD decay steps: ${WSD_decay_steps}
    - WSD stable steps: ${WSD_stable_steps}
    - KL coefficient: ${kl_coef}
    - Episodes: ${episodes}
    - Config: ${config}"
    
PYTHON_CMD="""python em_gsm8k.py --config configs/${config}.yml \
    --warmup_ratio ${WR} \
    --WSD_decay_steps ${WSD_decay_steps} \
    --WSD_stable_steps ${WSD_stable_steps} \
    --kl_coef ${kl_coef} \
    --total_episodes ${episodes} \
    --run_name em_run \
    --output_dir em_run
"""

# Check if srun is available.
if command -v srun &> /dev/null; then
    # With srun, we don't need to set CUDA_VISIBLE_DEVICES manually.
    srun -G 4 -N 1 -n ${num_processes} bash -c "$PYTHON_CMD"
else
    # With mpirun, add a command to export CUDA_VISIBLE_DEVICES based on the local rank.
    mpirun -np ${num_processes} bash -c "export CUDA_VISIBLE_DEVICES=\$((OMPI_COMM_WORLD_LOCAL_RANK)); $PYTHON_CMD"
fi
