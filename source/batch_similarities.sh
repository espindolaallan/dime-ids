#!/bin/bash

# batch_similarities.sh
#
# This script automates running "similarity_based_classification.py" across multiple pools (1 to 30)
# and user-specified similarity metrics (euclidean, jaccard, cosine).
# It processes pools in batches of 10 to control how many classification jobs run in parallel.
#
# Usage:
#   ./batch_similarities.sh [--generalization] [--normalize] <similarity_metric1> <similarity_metric2> ...
# Example:
#   ./batch_similarities.sh euclidean jaccard
#   ./batch_similarities.sh --generalization cosine
#   ./batch_similarities.sh --normalize euclidean
#   ./batch_similarities.sh --generalization --normalize jaccard cosine

# Check if any arguments were provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [--generalization] [--normalize] [--constraint] <similarity_metric1> <similarity_metric2> ..."
    echo "Available metrics: euclidean, jaccard, cosine"
    exit 1
fi

# Flags for optional arguments
GENERALIZATION_MODE=false
NORMALIZE_MODE=false
CONSTRAINTS_MODE=false

# Process optional flags
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --generalization) GENERALIZATION_MODE=true ;;
        --normalize) NORMALIZE_MODE=true ;;
        --constraints) CONSTRAINTS_MODE=true ;;
        *) break ;;  # Stop processing flags when a similarity metric is encountered
    esac
    shift  # Remove the processed flag from arguments
done

# Collect similarity metrics
SELECTED_SIMILARITIES=("$@")

# Validate input
VALID_METRICS=("euclidean" "jaccard" "cosine")
for sim in "${SELECTED_SIMILARITIES[@]}"; do
    if [[ ! " ${VALID_METRICS[@]} " =~ " ${sim} " ]]; then
        echo "Error: Invalid similarity metric '$sim'."
        echo "Available metrics: euclidean, jaccard, cosine"
        exit 1
    fi
done

# Define the number of pools to process
NUM_POOLS=30

# Define ranges
POOLS=$(seq 1 $NUM_POOLS)
BATCH_SIZE=20     # Number of pools to run in parallel

# Iterate over the pools in batches
for start in $(seq 11 $BATCH_SIZE $NUM_POOLS); do
    end=$((start + BATCH_SIZE - 1))
    if [ $end -gt $NUM_POOLS ]; then
        end=$NUM_POOLS  # Ensure we don't exceed NUM_POOLS
    fi

    echo "=================================="
    echo "  Running Pools $start to $end for selected similarities: ${SELECTED_SIMILARITIES[*]}"
    echo "  Generalization Mode: $GENERALIZATION_MODE"
    echo "  Normalize Mode: $NORMALIZE_MODE"
    echo "  Constraints Mode: $CONSTRAINTS_MODE"
    echo "=================================="

    # Launch all pools in this batch
    for pool in $(seq $start $end); do
        for sim in "${SELECTED_SIMILARITIES[@]}"; do
            echo "Running Pool $pool with $sim similarity..."
            
            # Construct command
            CMD=("python" "similarity_based_classification.py" "--pool_id" "$pool" "--similarity_metric" "$sim")
            if [ "$GENERALIZATION_MODE" == "true" ]; then
                CMD+=("--generalization")
            fi
            if [ "$NORMALIZE_MODE" == "true" ]; then
                CMD+=("--normalize")
            fi
            if [ "$CONSTRAINTS_MODE" == "true" ]; then
                CMD+=("--constraints")
            fi
            # Run the command in the background
            "${CMD[@]}" &
        done
    done

    wait  # Ensures all processes in the batch finish before moving to the next batch

    echo "Finished Pools $start to $end."
done

echo "All classifications completed!"
