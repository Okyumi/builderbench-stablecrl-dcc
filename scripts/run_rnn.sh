#!/bin/bash

PYTHON_SCRIPT="ppo_rnn.py"
GPU_IDS=(0 5 6 7)
ENV_IDS=(2 3 4 5)

# Get the number of instances from the length of the GPU array
NUM_INSTANCES=${#GPU_IDS[@]}

echo "Starting $NUM_INSTANCES instances of $PYTHON_SCRIPT in parallel..."

# Loop from 0 to NUM_INSTANCES-1
for i in $(seq 0 $((NUM_INSTANCES - 1))); do

    # Get the specific GPU and ENV ID for this instance
    GPU_ID=${GPU_IDS[$i]}
    ENV_ID=${ENV_IDS[$i]}

    (
        cd ..
        
        echo "Starting instance with ENV_ID $ENV_ID on CUDA device: $GPU_ID..."

        # Run the command with the specific IDs
        CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT \
            --env_id="creative-$ENV_ID-task1" \
            --num_timesteps=5000000000 \
            --track &
    ) &

done

echo "Waiting for all instances to finish..."
wait
echo "All parallel instances have completed."