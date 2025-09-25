#!/bin/bash

echo "Starting nullspace..."

# Check if CUDA is available in PyTorch
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); exit(0 if torch.cuda.is_available() else 1)"
if [ $? -ne 0 ]; then
    echo "CUDA is not available in PyTorch. Running on CPU may be very slow."
    echo "Consider installing PyTorch with CUDA support if you have a compatible GPU."
fi

INPUT_JSON="data/factorial/math_factorial.json"
OUTPUT_DIR="out/diffmean/qwen3_30b/math"
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
TRAIN_BATCH_SIZE=32
EVAL_BATCH_SIZE=32

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

python src/diffmean_analysis.py \
    --input_json "$INPUT_JSON" \
    --model_name "$MODEL_NAME" \
    --all_layers \
    --label_fields syc ga pr ag rand syc_all \
    --output_dir "$OUTPUT_DIR" \
    --null_ablate \
    --null_labels ga pr syc ag rand syc_all \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --dtype float32
