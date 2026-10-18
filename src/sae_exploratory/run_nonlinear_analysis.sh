#!/bin/bash

###############################################################################
# SAE Nonlinear Analysis Pipeline
# 使用神经网络进行SAE特征分析，对比线性模型(Ridge/Lasso)的性能
###############################################################################

# 默认参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DEVICE="cuda:0"
TOP_K=50

# 神经网络参数
MODEL_TYPE="mlp"  # mlp 或 attention
HIDDEN_DIMS="256,128,64"  # 更小的网络避免过拟合
EPOCHS=200
BATCH_SIZE=64
LEARNING_RATE=0.0001  # 更小的学习率
DROPOUT=0.5  # 更强的dropout

# 数据文件路径
TRAIN_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_with_prompt.json"
TEST_DATA="../data_generation/task_family_1/spatial_procedure_dataset_EN_test_with_prompt.json"

# 使用已有的SAE模型（需要指定）
if [ -z "$1" ]; then
    echo "Usage: $0 <sae_checkpoint_path> [model_type]"
    echo ""
    echo "Example:"
    echo "  $0 ./outputs_20260105_124630/sae_checkpoints/sae_layer15_exp64_*/sae_final.pt mlp"
    echo "  $0 ./outputs_20260105_124630/sae_checkpoints/sae_layer15_exp64_*/sae_final.pt attention"
    echo ""
    echo "Available model types: mlp, attention"
    exit 1
fi

SAE_PATH=$1

# 可选：指定模型类型
if [ ! -z "$2" ]; then
    MODEL_TYPE=$2
fi

# 验证SAE路径
if [ ! -f "$SAE_PATH" ]; then
    echo "ERROR: SAE checkpoint not found at $SAE_PATH"
    exit 1
fi

# 输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./sae_analysis_nonlinear_${MODEL_TYPE}_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "SAE Nonlinear Analysis with Neural Network"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "SAE Path: ${SAE_PATH}"
echo "Predictor Type: ${MODEL_TYPE}"
echo "Hidden Dims: ${HIDDEN_DIMS}"
echo "Epochs: ${EPOCHS}, Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}, Dropout: ${DROPOUT}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================="

###############################################################################
# 运行非线性分析
###############################################################################
echo ""
echo "Running nonlinear SAE feature analysis..."
echo "=========================================="

python analyze_sae_features_nonlinear.py \
    --model_name "${MODEL_NAME}" \
    --sae_path "${SAE_PATH}" \
    --train_data_file "${TRAIN_DATA}" \
    --test_data_file "${TEST_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --top_k ${TOP_K} \
    --model_type ${MODEL_TYPE} \
    --hidden_dims "${HIDDEN_DIMS}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --dropout ${DROPOUT}

if [ $? -ne 0 ]; then
    echo "ERROR: Nonlinear analysis failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Nonlinear Analysis Complete!"
echo "=========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key outputs:"
echo "  - Analysis results: ${OUTPUT_DIR}/sae_analysis_nn_layer*.json"
echo "  - Trained model: ${OUTPUT_DIR}/predictor_layer*.pt"
echo "  - Training curve: ${OUTPUT_DIR}/training_curve_layer*.png"
echo "  - Predictions: ${OUTPUT_DIR}/predictions_vs_true_layer*.png"
echo "=========================================="

