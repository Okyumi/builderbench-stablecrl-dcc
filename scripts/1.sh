#!/bin/bash

PYTHON_SCRIPT="crl.py"
NUM_CUBES=(3 4 3 4)
GPU_IDS=(0 1 2 3)
SEEDS=(0 0 1 1)
SEQUENCE_LENGTHS=(180 240 180 240)
ROLLOUT_LENGTHS=(180 240 180 240)
AUG_NOISE_STDS=(0.01 0.01 0.001 0.001)

# Get the number of instances from the length of the GPU array
NUM_INSTANCES=${#GPU_IDS[@]}

echo "Starting $NUM_INSTANCES instances of $PYTHON_SCRIPT in parallel..."

# Loop from 0 to NUM_INSTANCES-1
for i in $(seq 0 $((NUM_INSTANCES - 1))); do

    
    # Get the specific GPU and ENV ID for this instance
    GPU_ID=${GPU_IDS[$i]}
    SEED=${SEEDS[$i]}
    NUM_CUBE=${NUM_CUBES[$i]}
    SEQ_LEN=${SEQUENCE_LENGTHS[$i]}
    ROLLOUT_LEN=${ROLLOUT_LENGTHS[$i]}
    AUG_NOISE_STD=${AUG_NOISE_STDS[$i]}
    
    (
        cd ..

        echo "Starting instance with SEED $SEED on task creative-$NUM_CUBE on CUDA device: $GPU_ID (from $(pwd))..."

        # Run the command from the parent directory
        CUDA_VISIBLE_DEVICES=$GPU_ID python $PYTHON_SCRIPT \
            --env_id="sparse-planar-position-4-cube-$NUM_CUBE" \
            --wandb_name_tag="datacoll_train_aug_noise_std_$AUG_NOISE_STD" \
            --aug_noise_std=$AUG_NOISE_STD \
            --seed=$SEED \
            --num_timesteps=300_000_000 \
            --sequence_length=$SEQ_LEN \
            --rollout_length=$ROLLOUT_LEN \
            --track
    ) &

done

echo "Waiting for all instances to finish..."
wait
echo "All parallel instances have completed."