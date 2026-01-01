#!/bin/bash

PYTHON_SCRIPT="ppo.py"
NUM_CUBES=(3 4 5 6)
GPU_IDS=(0 1 2 3)
SEEDS=(0 0 0 0)

# Get the number of instances from the length of the GPU array
NUM_INSTANCES=${#GPU_IDS[@]}

echo "Starting $NUM_INSTANCES instances of $PYTHON_SCRIPT in parallel..."

# Loop from 0 to NUM_INSTANCES-1
for i in $(seq 0 $((NUM_INSTANCES - 1))); do

    
    # Get the specific GPU and ENV ID for this instance
    GPU_ID=${GPU_IDS[$i]}
    SEED=${SEEDS[$i]}
    NUM_CUBE=${NUM_CUBES[$i]}
    
    (
        cd ..

        echo "Starting instance with SEED $SEED on cube-$NUM_CUBE on CUDA device: $GPU_ID (from $(pwd))..."

        # Run the command from the parent directory
        CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT \
            --wandb_name_tag="no-permutation" \
            --env_id="planar-position-4-cube-$NUM_CUBE" \
            --seed=$SEED \
            --num_timesteps=1_000_000_000 \
            --track \
            --no-permutation_invariant_reward
    ) &

done

echo "Waiting for all instances to finish..."
wait
echo "All parallel instances have completed."