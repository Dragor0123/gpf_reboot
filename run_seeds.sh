#!/bin/bash

# Simple script to run train_prompt_tuning.py with different seeds
# Usage: ./run_seeds.sh <source_dataset> <target_dataset>
# Example: ./run_seeds.sh citeseer computers

set -e

# Check arguments
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <source_dataset> <target_dataset>"
    echo "Example: $0 citeseer computers"
    exit 1
fi

SOURCE_DATASET=$1
TARGET_DATASET=$2
CONFIG_FILE="config.yaml"
START_SEED=42
END_SEED=46

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================="
echo "Running experiments: $SOURCE_DATASET -> $TARGET_DATASET"
echo "Seeds: $START_SEED to $END_SEED"
echo "========================================="

# Run experiments for each seed
for seed in $(seq $START_SEED $END_SEED); do
    echo -e "${BLUE}[SEED $seed]${NC} Starting experiment..."
    
    # Create temporary config
    temp_config="temp_config_${SOURCE_DATASET}_${TARGET_DATASET}_seed${seed}.yaml"
    cp "$CONFIG_FILE" "$temp_config"
    
    # Update source_dataset, target_dataset, and seed
    sed -i "s/source_dataset: .*/source_dataset: \"$SOURCE_DATASET\"/" "$temp_config"
    sed -i "s/target_dataset: .*/target_dataset: \"$TARGET_DATASET\"/" "$temp_config"
    sed -i "s/seed: .*/seed: $seed/" "$temp_config"
    
    # Run experiment
    if CONFIG_PATH="$temp_config" python train_prompt_tuning.py; then
        echo -e "${GREEN}[SEED $seed]${NC} Completed successfully"
    else
        echo -e "${RED}[SEED $seed]${NC} Failed"
    fi
    
    # Clean up temp config
    rm -f "$temp_config"
    
    echo ""
done

echo -e "${GREEN}All experiments completed!${NC}"