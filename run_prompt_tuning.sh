#!/bin/bash

# Usage: ./run_prompt_tuning.sh {source_dataset} {target_dataset}
# Example: ./run_prompt_tuning.sh cora citeseer

if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_dataset> <target_dataset>"
    echo "Example: $0 cora citeseer"
    exit 1
fi

SOURCE_DATASET=$1
TARGET_DATASET=$2

echo "Running prompt tuning experiments: $SOURCE_DATASET -> $TARGET_DATASET"
echo "Seeds: 42, 43, 44, 45, 46"

# Run for seeds 42 to 46
for seed in {42..46}; do
    echo "----------------------------------------"
    echo "Running with seed: $seed"
    echo "Source: $SOURCE_DATASET, Target: $TARGET_DATASET"
    echo "----------------------------------------"
    
    python train_prompt_tuning.py \
        --source_dataset "$SOURCE_DATASET" \
        --target_dataset "$TARGET_DATASET" \
        --seed "$seed"
    
    if [ $? -ne 0 ]; then
        echo "Error: train_prompt_tuning.py failed with seed $seed"
        exit 1
    fi
    
    echo "Completed seed $seed"
    echo ""
done

echo "All experiments completed successfully!"