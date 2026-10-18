#!/bin/bash

###############################################################################
# 极简神经网络分析
# 策略：最小网络 + 最强正则化，验证是否存在非线性效应
###############################################################################

if [ -z "$1" ]; then
    echo "Usage: $0 <sae_path> [top_k] [hidden_dim]"
    echo ""
    echo "Example:"
    echo "  $0 ./outputs_*/sae_checkpoints/*/sae_final.pt"
    echo "  $0 ./outputs_*/sae_checkpoints/*/sae_final.pt 100 32"
    echo ""
    exit 1
fi

SAE_PATH=$1
TOP_K=${2:-100}  # 默认100个特征
HIDDEN=${3:-32}  # 默认32维隐藏层

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
TRAIN_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json"
TEST_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json"
DEVICE="cuda:0"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./simple_nn_k${TOP_K}_h${HIDDEN}_${TIMESTAMP}"

echo "=========================================="
echo "Simple NN Analysis"
echo "=========================================="
echo "SAE: $SAE_PATH"
echo "Top-K: $TOP_K, Hidden: $HIDDEN"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python analyze_sae_features_simple.py \
    --model_name "$MODEL_NAME" \
    --sae_path "$SAE_PATH" \
    --train_data_file "$TRAIN_DATA" \
    --test_data_file "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --top_k $TOP_K \
    --hidden_dim $HIDDEN

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Success! Check: $OUTPUT_DIR"
    echo "=========================================="
fi




