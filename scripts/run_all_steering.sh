#!/bin/bash

echo "Starting steering..."

# Check if CUDA is available in PyTorch
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); exit(0 if torch.cuda.is_available() else 1)"
if [ $? -ne 0 ]; then
    echo "CUDA is not available in PyTorch. Running on CPU may be very slow."
    echo "Consider installing PyTorch with CUDA support if you have a compatible GPU."
fi

LAYER=32
INPUT_JSON="prompts/subset/math_genuine_agreement.json"
TRUTHFUL_TRAIN_JSON="data/truthfulqa/train.jsonl"
TRUTHFUL_EVAL_JSON="data/truthfulqa/test.jsonl"
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_DIR="out/steering/qwen3_30b/math/syc"
TRUTHFUL_OUTPUT_DIR="out/steering/qwen3_30b/truthful/syc"
TRUTHFUL_LOG_DIR="out/steering_logs/qwen3_30b/truthful/syc"
STEER_ALPHA_GRID="-4,-2,0,2,4"
CI_BOOTSTRAPS=2000
CI_LEVEL=0.95
W_RAW_FILE="out/diffmean/qwen3_30b/math/wDiffMean_raw_syc_L${LAYER}.npy"
PRAISE_DETECTOR_MODEL="out/praise_roberta"
MAX_EXAMPLES=200
GEN_MAX_NEW_TOKENS=4

# Train praise detector first
python src/praise_roberta.py \
	--train_json data/praise.json \
	--eval_json  data/praise.json \
	--model_name roberta-base \
	--output_dir "$PRAISE_DETECTOR_MODEL"

python src/cross_effects.py \
    --input_json "$INPUT_JSON" \
    --model_name "$MODEL_NAME" \
    --layer $LAYER \
    --steer_eval \
    --output_dir "$OUTPUT_DIR" \
    --steer_alpha_grid="$STEER_ALPHA_GRID" \
    --ci_bootstraps $CI_BOOTSTRAPS \
    --ci_level $CI_LEVEL \
    --w_raw_file "$W_RAW_FILE" \
    --praise_detector_model "$PRAISE_DETECTOR_MODEL" \
    --max_examples $MAX_EXAMPLES \
    --prefill_until_equals \
    --gen_max_new_tokens $GEN_MAX_NEW_TOKENS

python src/praise_steering.py \
    --input_json "$INPUT_JSON" \
    --model_name "$MODEL_NAME" \
    --layer $LAYER \
    --steer_eval \
    --output_dir "$OUTPUT_DIR" \
    --steer_alpha_grid="$STEER_ALPHA_GRID" \
    --ci_bootstraps $CI_BOOTSTRAPS \
    --ci_level $CI_LEVEL \
    --w_raw_file "$W_RAW_FILE" \
    --praise_detector_model "$PRAISE_DETECTOR_MODEL" \
    --max_examples $MAX_EXAMPLES \
    --prefill_praise \
    --gen_max_new_tokens $GEN_MAX_NEW_TOKENS

python src/truthful_steering.py \
    --model_name "$MODEL_NAME" \
    --input_json "$TRUTHFUL_TRAIN_JSON" \
    --steer_eval \
    --w_raw_file "$W_RAW_FILE" \
    --steer_layer $LAYER \
    --steer_alpha_grid="$STEER_ALPHA_GRID" \
    --gen_max_new_tokens $GEN_MAX_NEW_TOKENS \
    --output_dir "$TRUTHFUL_OUTPUT_DIR" \
    --save_outputs_json "$TRUTHFUL_LOG_DIR/outputs_L$LAYER.json" \
    --eval_json "$TRUTHFUL_EVAL_JSON"
