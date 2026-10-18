#!/bin/bash

###############################################################################
# SAE Nonlinear Analysis V2 (with Feature Selection)
# 先用Ridge选择重要特征，再训练神经网络，避免维度灾难
###############################################################################

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DEVICE="cuda:0"

# 特征选择和网络参数
TOP_K=500  # 选择Top 500个特征（可调整）
HIDDEN_DIM=128
EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=0.001
DROPOUT=0.3

# 数据路径
TRAIN_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json"
TEST_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json"

if [ -z "$1" ]; then
    echo "Usage: $0 <sae_checkpoint_path> [top_k]"
    echo ""
    echo "Example:"
    echo "  $0 ./outputs_20260105_124630/sae_checkpoints/sae_layer15_exp64_*/sae_final.pt"
    echo "  $0 ./outputs_20260105_124630/sae_checkpoints/sae_layer15_exp64_*/sae_final.pt 1000"
    echo ""
    echo "Default top_k: 500"
    echo "Recommended top_k values: 200-1000"
    exit 1
fi

SAE_PATH=$1

# 验证第二个参数是否是数字
if [ ! -z "$2" ]; then
    if [[ "$2" =~ ^[0-9]+$ ]]; then
        TOP_K=$2
    else
        echo "ERROR: Second argument must be a number (top_k), got: $2"
        echo "Using default top_k=${TOP_K}"
    fi
fi

if [ ! -f "$SAE_PATH" ]; then
    echo "ERROR: SAE checkpoint not found at $SAE_PATH"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./sae_analysis_nn_v2_k${TOP_K}_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "SAE Nonlinear Analysis V2"
echo "=========================================="
echo "SAE Path: ${SAE_PATH}"
echo "Feature Selection: Top-${TOP_K}"
echo "Hidden Dim: ${HIDDEN_DIM}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python analyze_sae_features_nonlinear_v2.py \
    --model_name "${MODEL_NAME}" \
    --sae_path "${SAE_PATH}" \
    --train_data_file "${TRAIN_DATA}" \
    --test_data_file "${TEST_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --top_k ${TOP_K} \
    --hidden_dim ${HIDDEN_DIM} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --dropout ${DROPOUT}

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Success! Results saved to:"
    echo "  ${OUTPUT_DIR}"
    echo "=========================================="
else
    echo "ERROR: Analysis failed!"
    exit 1
fi

