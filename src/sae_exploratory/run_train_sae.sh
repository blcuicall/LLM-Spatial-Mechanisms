#!/bin/bash

###############################################################################
# 单独运行SAE训练
###############################################################################

# 参数配置
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
LAYER=15
EXPANSION_FACTOR=8
BATCH_SIZE=4
NUM_TOKENS=100000
L1_COEF=0.001
DEVICE="cuda:0"

# 数据文件
TRAIN_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json"

# 输出目录
OUTPUT_DIR="./sae_checkpoints"

echo "=========================================="
echo "Training SAE"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Layer: ${LAYER}"
echo "Expansion Factor: ${EXPANSION_FACTOR}"
echo "=========================================="

python train_sae_relation.py \
    --model_name "${MODEL_NAME}" \
    --train_data_file "${TRAIN_DATA}" \
    --layer ${LAYER} \
    --expansion_factor ${EXPANSION_FACTOR} \
    --batch_size ${BATCH_SIZE} \
    --num_tokens ${NUM_TOKENS} \
    --l1_coefficient ${L1_COEF} \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}"

echo "=========================================="
echo "Training complete!"
echo "=========================================="




