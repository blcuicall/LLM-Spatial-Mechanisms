#!/bin/bash

###############################################################################
# 单独运行SAE特征干预实验
###############################################################################

# 参数配置
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DEVICE="cuda:0"
NUM_SAMPLES=100
INTERVENTION_MAG=2.0

# 数据文件
TEST_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json"

# SAE和分析结果路径（需要根据实际情况修改）
SAE_PATH="./sae_checkpoints/sae_layer15_exp8_20260105_120000/sae_final.pt"
ANALYSIS_RESULTS="./sae_analysis/sae_analysis_layer15.json"

# 输出目录
OUTPUT_DIR="./sae_intervention"

# 检查文件是否存在
if [ ! -f "${SAE_PATH}" ]; then
    echo "ERROR: SAE checkpoint not found at ${SAE_PATH}"
    exit 1
fi

if [ ! -f "${ANALYSIS_RESULTS}" ]; then
    echo "ERROR: Analysis results not found at ${ANALYSIS_RESULTS}"
    echo "Please run analyze_sae_features.py first."
    exit 1
fi

echo "=========================================="
echo "Running Feature Intervention Experiments"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "SAE Path: ${SAE_PATH}"
echo "=========================================="

python intervene_sae_features.py \
    --model_name "${MODEL_NAME}" \
    --sae_path "${SAE_PATH}" \
    --test_data_file "${TEST_DATA}" \
    --analysis_results "${ANALYSIS_RESULTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --num_samples ${NUM_SAMPLES} \
    --intervention_magnitude ${INTERVENTION_MAG}

echo "=========================================="
echo "Intervention experiments complete!"
echo "=========================================="




