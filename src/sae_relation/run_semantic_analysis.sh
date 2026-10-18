#!/bin/bash

###############################################################################
# SAE特征语义可解释性分析
# 通过最大激活样本分析来判断特征是否真的捕捉到语义
###############################################################################

echo "=================================================="
echo "SAE Feature Semantic Interpretability Analysis"
echo "=================================================="

# 配置参数
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DEVICE="cuda"
LAYER=20
TOP_K=20

echo "Configuration:"
echo "  Model: ${MODEL_NAME} (using config.py PATHS)"
echo "  Device: ${DEVICE}"
echo "  Layer: ${LAYER}"
echo "  Top K samples: ${TOP_K}"
echo "=================================================="

# ========================================
# 英文数据集分析
# ========================================
echo ""
echo "Analyzing ENGLISH features..."
echo "----------------------------------------"

SAE_PATH="./outputs_20260105_202027/sae_checkpoints/sae_layer20_exp32_20260105_202045/sae_final.pt"
DATA_PATH="../data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test.json"
OUTPUT_DIR="./feature_semantic_analysis_EN"

# 检查SAE路径是否存在
if [ ! -f "${SAE_PATH}" ]; then
    echo "ERROR: English SAE checkpoint not found at ${SAE_PATH}"
    echo "Please check the path or train an SAE first."
    exit 1
fi

# 检查数据文件是否存在
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: Test data not found at ${DATA_PATH}"
    exit 1
fi

# 分析顶级特征 (从之前的分析结果)
# Top 5 overall: 88728, 81718, 30656, 51936, 112164
# Top Y-axis: 88728, 30656, 108799, 56997, 108126
# Top X-axis: 112164, 105158, 76388
# Top Z-axis: 81718, 66301, 87656

echo "Analyzing top overall features..."
python analyze_feature_semantics.py \
  --sae_path "$SAE_PATH" \
  --data_path "$DATA_PATH" \
  --feature_ids "88728,81718,30656,51936,112164" \
  --model_name "$MODEL_NAME" \
  --layer $LAYER \
  --top_k $TOP_K \
  --output_dir "$OUTPUT_DIR/overall" \
  --device "$DEVICE"

echo ""
echo "Analyzing top Y-axis features..."
python analyze_feature_semantics.py \
  --sae_path "$SAE_PATH" \
  --data_path "$DATA_PATH" \
  --feature_ids "88728,30656,108799" \
  --model_name "$MODEL_NAME" \
  --layer $LAYER \
  --top_k $TOP_K \
  --output_dir "$OUTPUT_DIR/y_axis" \
  --device "$DEVICE"

echo ""
echo "Analyzing top X-axis features..."
python analyze_feature_semantics.py \
  --sae_path "$SAE_PATH" \
  --data_path "$DATA_PATH" \
  --feature_ids "112164,105158,76388" \
  --model_name "$MODEL_NAME" \
  --layer $LAYER \
  --top_k $TOP_K \
  --output_dir "$OUTPUT_DIR/x_axis" \
  --device "$DEVICE"

echo ""
echo "Analyzing top Z-axis features..."
python analyze_feature_semantics.py \
  --sae_path "$SAE_PATH" \
  --data_path "$DATA_PATH" \
  --feature_ids "81718,66301,87656" \
  --model_name "$MODEL_NAME" \
  --layer $LAYER \
  --top_k $TOP_K \
  --output_dir "$OUTPUT_DIR/z_axis" \
  --device "$DEVICE"

# ========================================
# 中文数据集分析
# ========================================
echo ""
echo ""
echo "Analyzing CHINESE features..."
echo "----------------------------------------"

SAE_PATH="./outputs_20260105_205825/sae_checkpoints/sae_layer20_exp32_20260105_205846/sae_final.pt"
DATA_PATH="../data_generation/task_family_1/data_relation/spatial_reasoning_dataset_CN_test.json"
OUTPUT_DIR="./feature_semantic_analysis_CN"

# 检查SAE路径是否存在
if [ ! -f "${SAE_PATH}" ]; then
    echo "ERROR: Chinese SAE checkpoint not found at ${SAE_PATH}"
    echo "Please check the path or train an SAE first."
    exit 1
fi

# 检查数据文件是否存在
if [ ! -f "${DATA_PATH}" ]; then
    echo "ERROR: Test data not found at ${DATA_PATH}"
    exit 1
fi

# 分析顶级特征 (从之前的分析结果)
# Top 5 overall: 24306, 59922, 106404, 12248, 6742
# Top Y-axis: 24306, 44769, 59922
# Top X-axis: 6742, 101550, 95555
# Top Z-axis: 106404, 58743, 15188

echo "Analyzing top overall features..."
python analyze_feature_semantics.py \
  --sae_path "$SAE_PATH" \
  --data_path "$DATA_PATH" \
  --feature_ids "24306,59922,106404,12248,6742" \
  --model_name "$MODEL_NAME" \
  --layer $LAYER \
  --top_k $TOP_K \
  --output_dir "$OUTPUT_DIR/overall" \
  --device "$DEVICE"

echo ""
echo "Analyzing top Y-axis features..."
python analyze_feature_semantics.py \
  --sae_path "$SAE_PATH" \
  --data_path "$DATA_PATH" \
  --feature_ids "24306,44769,59922" \
  --model_name "$MODEL_NAME" \
  --layer $LAYER \
  --top_k $TOP_K \
  --output_dir "$OUTPUT_DIR/y_axis" \
  --device "$DEVICE"

echo ""
echo "Analyzing top X-axis features..."
python analyze_feature_semantics.py \
  --sae_path "$SAE_PATH" \
  --data_path "$DATA_PATH" \
  --feature_ids "6742,101550,95555" \
  --model_name "$MODEL_NAME" \
  --layer $LAYER \
  --top_k $TOP_K \
  --output_dir "$OUTPUT_DIR/x_axis" \
  --device "$DEVICE"

echo ""
echo "Analyzing top Z-axis features..."
python analyze_feature_semantics.py \
  --sae_path "$SAE_PATH" \
  --data_path "$DATA_PATH" \
  --feature_ids "106404,58743,15188" \
  --model_name "$MODEL_NAME" \
  --layer $LAYER \
  --top_k $TOP_K \
  --output_dir "$OUTPUT_DIR/z_axis" \
  --device "$DEVICE"

echo ""
echo "=================================================="
echo "Analysis Complete!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - ./feature_semantic_analysis_EN/"
echo "  - ./feature_semantic_analysis_CN/"
echo ""
echo "Key questions answered:"
echo "  1. Do features correspond to specific spatial concepts?"
echo "  2. Are top-activating samples semantically coherent?"
echo "  3. Is there enrichment of specific keywords?"
echo ""
echo "Next steps:"
echo "  1. Review the generated JSON files"
echo "  2. Check if Y-axis features activate on 'front/behind' words"
echo "  3. Compare EN and CN features for cross-lingual patterns"
echo "=================================================="

