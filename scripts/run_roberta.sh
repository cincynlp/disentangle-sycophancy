#!/bin/bash

echo "Starting RoBERTa praise detector training..."

# Check if CUDA is available in PyTorch
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); exit(0 if torch.cuda.is_available() else 1)"
if [ $? -ne 0 ]; then
    echo "CUDA is not available in PyTorch. Running on CPU may be very slow."
    echo "Consider installing PyTorch with CUDA support if you have a compatible GPU."
fi

OUTPUT_DIR="out/praise_roberta"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

python src/praise_roberta.py \
	--train_json data/praise.json \
	--eval_json  data/praise.json \
	--model_name roberta-base \
	--output_dir "$OUTPUT_DIR"
