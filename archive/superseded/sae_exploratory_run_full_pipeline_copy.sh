#!/bin/bash

###############################################################################
# SAE Analysis Full Pipeline
# 完整的SAE分析流程，包括训练、分析、干预和可视化
###############################################################################

# 默认参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LAYER=15
EXPANSION_FACTOR=32 #8
BATCH_SIZE=1024 #4
NUM_TOKENS=500000 #100000
L1_COEF=0.0001 #0.001
DEVICE="cuda:0"
TOP_K=50
NUM_TEST_SAMPLES=100
INTERVENTION_MAG=2.0

# 数据文件路径（需要根据实际情况修改）
TRAIN_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json"
TEST_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json"

# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_ROOT="./outputs_${TIMESTAMP}"
SAE_CHECKPOINT_DIR="${OUTPUT_ROOT}/sae_checkpoints"
ANALYSIS_DIR="${OUTPUT_ROOT}/sae_analysis"
INTERVENTION_DIR="${OUTPUT_ROOT}/sae_intervention"
VISUALIZATION_DIR="${OUTPUT_ROOT}/sae_visualizations"

mkdir -p ${OUTPUT_ROOT}
mkdir -p ${SAE_CHECKPOINT_DIR}
mkdir -p ${ANALYSIS_DIR}
mkdir -p ${INTERVENTION_DIR}
mkdir -p ${VISUALIZATION_DIR}

echo "=========================================="
echo "SAE Analysis Full Pipeline"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Layer: ${LAYER}"
echo "Expansion Factor: ${EXPANSION_FACTOR}"
echo "Output Directory: ${OUTPUT_ROOT}"
echo "=========================================="

###############################################################################
# Step 1: 训练SAE
###############################################################################
echo ""
echo "Step 1: Training SAE..."
echo "=========================================="

python train_sae_relation.py \
    --model_name "${MODEL_NAME}" \
    --train_data_file "${TRAIN_DATA}" \
    --layer ${LAYER} \
    --expansion_factor ${EXPANSION_FACTOR} \
    --batch_size ${BATCH_SIZE} \
    --num_tokens ${NUM_TOKENS} \
    --l1_coefficient ${L1_COEF} \
    --output_dir "${SAE_CHECKPOINT_DIR}" \
    --device "${DEVICE}"

if [ $? -ne 0 ]; then
    echo "ERROR: SAE training failed!"
    exit 1
fi

# 找到训练好的SAE模型路径
SAE_PATH=$(find ${SAE_CHECKPOINT_DIR} -name "sae_final.pt" | head -n 1)
echo "SAE model saved at: ${SAE_PATH}"

###############################################################################
# Step 2: 分析SAE特征
###############################################################################
echo ""
echo "Step 2: Analyzing SAE features..."
echo "=========================================="

python analyze_sae_features.py \
    --model_name "${MODEL_NAME}" \
    --sae_path "${SAE_PATH}" \
    --train_data_file "${TRAIN_DATA}" \
    --test_data_file "${TEST_DATA}" \
    --output_dir "${ANALYSIS_DIR}" \
    --device "${DEVICE}" \
    --top_k ${TOP_K}

if [ $? -ne 0 ]; then
    echo "ERROR: SAE feature analysis failed!"
    exit 1
fi

ANALYSIS_RESULTS="${ANALYSIS_DIR}/sae_analysis_layer${LAYER}.json"
echo "Analysis results saved at: ${ANALYSIS_RESULTS}"

###############################################################################
# Step 3: SAE特征干预实验
###############################################################################
echo ""
echo "Step 3: Running feature intervention experiments..."
echo "=========================================="

python intervene_sae_features.py \
    --model_name "${MODEL_NAME}" \
    --sae_path "${SAE_PATH}" \
    --test_data_file "${TEST_DATA}" \
    --analysis_results "${ANALYSIS_RESULTS}" \
    --output_dir "${INTERVENTION_DIR}" \
    --device "${DEVICE}" \
    --num_samples ${NUM_TEST_SAMPLES} \
    --intervention_magnitude ${INTERVENTION_MAG}

if [ $? -ne 0 ]; then
    echo "ERROR: Feature intervention experiment failed!"
    exit 1
fi

INTERVENTION_RESULTS="${INTERVENTION_DIR}/intervention_results_layer${LAYER}.json"
echo "Intervention results saved at: ${INTERVENTION_RESULTS}"

###############################################################################
# Step 4: 可视化结果
###############################################################################
echo ""
echo "Step 4: Generating visualizations..."
echo "=========================================="

python visualize_sae_results.py \
    --analysis_results "${ANALYSIS_RESULTS}" \
    --intervention_results "${INTERVENTION_RESULTS}" \
    --output_dir "${VISUALIZATION_DIR}"

if [ $? -ne 0 ]; then
    echo "ERROR: Visualization generation failed!"
    exit 1
fi

echo "Visualizations saved at: ${VISUALIZATION_DIR}"

###############################################################################
# 完成
###############################################################################
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "All outputs saved to: ${OUTPUT_ROOT}"
echo ""
echo "Key outputs:"
echo "  - SAE checkpoint: ${SAE_PATH}"
echo "  - Analysis results: ${ANALYSIS_RESULTS}"
echo "  - Intervention results: ${INTERVENTION_RESULTS}"
echo "  - Visualizations: ${VISUALIZATION_DIR}"
echo "=========================================="

