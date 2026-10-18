#!/bin/bash

###############################################################################
# 单独运行SAE特征分析
###############################################################################

# 参数配置
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DEVICE="cuda:0"
TOP_K=50

# 数据文件
TRAIN_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json"
TEST_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json"

# SAE模型路径（需要根据实际情况修改）
SAE_PATH="./sae_checkpoints/sae_layer15_exp8_20260105_120000/sae_final.pt"

# 输出目录
OUTPUT_DIR="./sae_analysis"

# 检查SAE路径是否存在
if [ ! -f "${SAE_PATH}" ]; then
    echo "ERROR: SAE checkpoint not found at ${SAE_PATH}"
    echo "Please update SAE_PATH in this script or train an SAE first."
    exit 1
fi

echo "=========================================="
echo "Analyzing SAE Features"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "SAE Path: ${SAE_PATH}"
echo "=========================================="

python analyze_sae_features.py \
    --model_name "${MODEL_NAME}" \
    --sae_path "${SAE_PATH}" \
    --train_data_file "${TRAIN_DATA}" \
    --test_data_file "${TEST_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --top_k ${TOP_K}

echo "=========================================="
echo "Analysis complete!"
echo "=========================================="

