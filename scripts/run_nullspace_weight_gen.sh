#!/bin/bash

echo "Starting nullspace weight generation..."

LAYER_MIN=1
LAYER_MAX=48
INPUT_PATHS="runs/qwen3_30b/cities_neg runs/qwen3_30b/cities_pos runs/qwen3_30b/claims runs/qwen3_30b/counterfactual runs/qwen3_30b/smaller_than runs/qwen3_30b/larger_than runs/qwen3_30b/sp_en_trans runs/qwen3_30b/sp_en_trans_pro"
OUTPUT_DIR="runs/nullspace_weights/qwen3_30b"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

python src/nullspace_weight_gen.py \
      --layer-min $LAYER_MIN \
      --layer-max $LAYER_MAX \
      --paths $INPUT_PATHS \
      --outdir "$OUTPUT_DIR" \
      --mode pooled
