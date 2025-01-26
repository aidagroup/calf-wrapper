#!/bin/bash

# Directory containing checkpoints
CHECKPOINT_DIR="/home/anton/repos/stable-baselines-playground/run/ppo_vanila/artifacts/checkpoints/ppo_InvertedPendulum-swingup_42"

# Count total number of checkpoints
total_checkpoints=$(ls $CHECKPOINT_DIR/ppo_checkpoint_*_steps.zip | wc -l)
step_size=$((total_checkpoints / 20))  # Divide by 20 to get 5% intervals

# Iterate through relax probabilities (0.0 to 1.0 with step 0.05)
for relax_prob in $(seq 0.0 0.05 1.0); do
    # Format relax_prob to 2 decimal places
    relax_prob_formatted=$(printf "%.2f" $relax_prob)
    
    # Find all checkpoint files, sort them, and select every nth file
    counter=0
    for checkpoint in $(ls $CHECKPOINT_DIR/ppo_checkpoint_*_steps.zip | sort -V); do
        counter=$((counter + 1))
        
        # Only process every nth checkpoint (5% intervals)
        if [ $((counter % step_size)) -ne 0 ] && [ $counter -ne $total_checkpoints ]; then
            continue
        fi
        
        # Extract step number from filename
        steps=$(echo $checkpoint | grep -o '[0-9]\+_steps' | grep -o '[0-9]\+')
        
        echo "Evaluating checkpoint at step $steps (${counter}/${total_checkpoints}) with relax_prob=${relax_prob_formatted}"
        echo "Checkpoint path: $checkpoint"
        
        python eval.py \
            --seed=1...30 \
            --controller=calf_wrapper \
            --mlflow.experiment-name=eval/InvPendulum/calf \
            --mlflow.run-name=calf_${steps}_relax_${relax_prob_formatted} \
            --model-path=$checkpoint \
            --env-id=InvertedPendulum-swingup \
            --calf.relax_prob=$relax_prob_formatted
        
        echo "Completed evaluation"
        echo "----------------------------------------"
    done
done