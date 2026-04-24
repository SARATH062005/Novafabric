#!/bin/bash
# Train a policy for the SO-100 robot using LeRobot

# Configuration
REPO_ID="sarath/so100_demo"
POLICY_TYPE="act" # Options: act, diffusion
OUTPUT_DIR="outputs/train/act_so100"
STEPS=5000 # Number of training steps
BATCH_SIZE=4 # Reduced for 4GB VRAM with 2 cameras

# Activate environment
source ~/miniforge3/bin/activate lerobot

echo "Starting training for $POLICY_TYPE on dataset $REPO_ID..."

# Run training
# Note: We disable wandb by default to avoid login prompts
python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=$REPO_ID \
    --dataset.root=datasets/sarath/so100_demo \
    --policy.type=$POLICY_TYPE \
    --policy.push_to_hub=false \
    --output_dir=$OUTPUT_DIR \
    --steps=$STEPS \
    --batch_size=$BATCH_SIZE \
    --wandb.enable=false

echo "Training complete. Checkpoints saved to $OUTPUT_DIR"
