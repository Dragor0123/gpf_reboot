#!/bin/bash

# Extended script to run train_prompt_tuning.py with different seeds across multiple target datasets
# Usage: ./run_seeds_extension.sh <source_dataset>
# Example: ./run_seeds_extension.sh citeseer

set -e

# Check arguments
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <source_dataset>"
    echo "Example: $0 citeseer"
    exit 1
fi

SOURCE_DATASET=$1
ALL_DATASETS=("cora" "citeseer" "pubmed" "computers" "photo")
CONFIG_FILE="config.yaml"
START_SEED=42
END_SEED=46

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Validate source dataset
if [[ ! " ${ALL_DATASETS[@]} " =~ " ${SOURCE_DATASET} " ]]; then
    echo -e "${RED}Error: Invalid source dataset '$SOURCE_DATASET'${NC}"
    echo "Valid datasets: ${ALL_DATASETS[@]}"
    exit 1
fi

echo "========================================="
echo "Running cross-domain experiments from: $SOURCE_DATASET"
echo "Seeds: $START_SEED to $END_SEED"
echo "========================================="

# Get target datasets (all except source)
TARGET_DATASETS=()
for dataset in "${ALL_DATASETS[@]}"; do
    if [[ "$dataset" != "$SOURCE_DATASET" ]]; then
        TARGET_DATASETS+=("$dataset")
    fi
done

echo -e "${YELLOW}Target datasets: ${TARGET_DATASETS[@]}${NC}"
echo ""

# Run experiments for each target dataset
for target_dataset in "${TARGET_DATASETS[@]}"; do
    echo "========================================="
    echo -e "${BLUE}Starting experiments: $SOURCE_DATASET -> $target_dataset${NC}"
    echo "========================================="
    
    # Run experiments for each seed
    for seed in $(seq $START_SEED $END_SEED); do
        echo -e "${BLUE}[SEED $seed]${NC} Starting experiment..."
        
        # Create temporary config
        temp_config="temp_config_${SOURCE_DATASET}_${target_dataset}_seed${seed}.yaml"
        cp "$CONFIG_FILE" "$temp_config"
        
        # Update source_dataset, target_dataset, and seed
        sed -i "s/source_dataset: .*/source_dataset: \"$SOURCE_DATASET\"/" "$temp_config"
        sed -i "s/target_dataset: .*/target_dataset: \"$target_dataset\"/" "$temp_config"
        sed -i "s/seed: .*/seed: $seed/" "$temp_config"
        
        # Create log filename
        log_file="${SOURCE_DATASET}_to_${target_dataset}_seed${seed}.log"
        
        # Run experiment with output redirection
        if CONFIG_PATH="$temp_config" python train_prompt_tuning.py > "$log_file" 2>&1; then
            echo -e "${GREEN}[SEED $seed]${NC} Completed successfully -> $log_file"
        else
            echo -e "${RED}[SEED $seed]${NC} Failed -> $log_file"
        fi
        
        # Clean up temp config
        rm -f "$temp_config"
        
        echo ""
    done
    
    echo -e "${GREEN}Completed all seeds for $SOURCE_DATASET -> $target_dataset${NC}"
    echo ""
done

echo "========================================="
echo -e "${GREEN}All experiments completed!${NC}"
echo "Log files generated: ${SOURCE_DATASET}_to_*_seed*.log"
echo "Total files: $((${#TARGET_DATASETS[@]} * 5))"
echo "========================================="