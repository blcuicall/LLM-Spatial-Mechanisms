#!/bin/bash

# =====================================================
# SAE Gradient Attribution Analysis Script
# 使用梯度归因分析SAE特征对空间推理的重要性
# =====================================================

set -e  # Exit on error

# =====================================================
# 配置参数
# =====================================================

# 模型配置
MODEL_NAME="Qwen2.5-0.5B-Instruct"  # 修改为你的模型名称
# MODEL_NAME="Qwen2.5-1.5B-Instruct"

# 数据配置
TRAIN_DATA="../data/spatial_train.json"  # 训练数据（用于分析）
TEST_DATA="../data/spatial_test.json"    # 测试数据

# SAE配置
SAE_PATH="../sae_saelens/sae_results_new/L8_F2048_L11e-05_20260103_102842/sae_checkpoint.pt"  # 修改为你的SAE路径

# 输出配置
OUTPUT_DIR="./gradient_attribution_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR}_${TIMESTAMP}"

# 设备配置
DEVICE="cuda:0"

# 分析参数
NUM_SAMPLES=200      # 使用的测试样本数量
TOP_K=50            # 分析的top特征数量

# =====================================================
# 检查必要文件
# =====================================================

echo "========================================"
echo "Checking required files..."
echo "========================================"

if [ ! -f "$TEST_DATA" ]; then
    echo "Error: Test data not found: $TEST_DATA"
    exit 1
fi

if [ ! -f "$SAE_PATH" ]; then
    echo "Error: SAE checkpoint not found: $SAE_PATH"
    exit 1
fi

echo "✓ All required files found"

# =====================================================
# 创建输出目录
# =====================================================

mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# =====================================================
# 运行梯度归因分析
# =====================================================

echo ""
echo "========================================"
echo "Running Gradient Attribution Analysis"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "SAE: $SAE_PATH"
echo "Test Data: $TEST_DATA"
echo "Num Samples: $NUM_SAMPLES"
echo "Top K: $TOP_K"
echo "Device: $DEVICE"
echo "========================================"
echo ""

python gradient_attribution_sae.py \
    --model_name "$MODEL_NAME" \
    --sae_path "$SAE_PATH" \
    --test_data_file "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --num_samples "$NUM_SAMPLES" \
    --top_k "$TOP_K"

# =====================================================
# 检查结果
# =====================================================

echo ""
echo "========================================"
echo "Checking Results"
echo "========================================"

if [ -f "${OUTPUT_DIR}/gradient_attribution_layer*.json" ]; then
    echo "✓ Gradient attribution results generated successfully"
    ls -lh "${OUTPUT_DIR}/"*.json
else
    echo "✗ No results found!"
    exit 1
fi

if [ -f "${OUTPUT_DIR}/top_features_grad_x_act.png" ]; then
    echo "✓ Visualizations generated successfully"
    ls -lh "${OUTPUT_DIR}/"*.png
else
    echo "✗ No visualizations found!"
fi

# =====================================================
# 总结
# =====================================================

echo ""
echo "========================================"
echo "Gradient Attribution Analysis Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - gradient_attribution_layer*.json : Attribution analysis results"
echo "  - top_features_grad_x_act.png     : Top features visualization"
echo "  - attribution_distribution_grad_x_act.png : Distribution analysis"
echo "  - attribution_heatmap_grad_x_act.png : Per-dimension heatmap"
echo "  - feature_overlap_grad_x_act.png  : Feature overlap analysis"
echo "========================================"

