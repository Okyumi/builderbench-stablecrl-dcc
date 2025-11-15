#!/bin/bash

PYTHON_SCRIPT="ppo.py"
GPU_IDS=(6 7)
SEEDS=(42 43)
EPISODE_LENGTH=1000

# Get the number of instances from the length of the GPU array
NUM_INSTANCES=${#GPU_IDS[@]}

echo "Starting $NUM_INSTANCES instances of $PYTHON_SCRIPT in parallel..."

# Loop from 0 to NUM_INSTANCES-1
for i in $(seq 0 $((NUM_INSTANCES - 1))); do

    
    # Get the specific GPU and ENV ID for this instance
    GPU_ID=${GPU_IDS[$i]}
    SEED=${SEEDS[$i]}

    (
        cd ..
        
        echo "Starting instance with SEED $SEED on CUDA device: $GPU_ID (from $(pwd))..."

        # Run the command from the parent directory
        CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT \
            --env_id="creative-4-task1" \
            --wandb_name_tag="no-permutation_invariant_reward" \
            --seed=$SEED \
            --no-permutation_invariant_reward \
            --env_episode_length=$EPISODE_LENGTH \
            --num_timesteps=3000000000 \
            --track
    ) &

done

echo "Waiting for all instances to finish..."
wait
echo "All parallel instances have completed."