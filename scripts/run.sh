#!/bin/bash

PYTHON_SCRIPT="ppo.py"
NUM_CUBES=(3 3 4 4)
GPU_IDS=(0 1 2 3)
SEEDS=(42 43 42 43)
EPISODE_LENGTH=500

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

        echo "Starting instance with SEED $SEED on task creative-$NUM_CUBE on CUDA device: $GPU_ID (from $(pwd))..."

        # Run the command from the parent directory
        CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT \
            --env_id="creative-$NUM_CUBE-task1" \
            --wandb_name_tag="direct-force-control" \
            --seed=$SEED \
            --env_episode_length=$EPISODE_LENGTH \
            --num_timesteps=3000000000 \
            --track
    ) &

done

echo "Waiting for all instances to finish..."
wait
echo "All parallel instances have completed."