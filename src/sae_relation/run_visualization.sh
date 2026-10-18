#!/bin/bash

###############################################################################
# 单独运行可视化
###############################################################################

# 分析和干预结果路径（需要根据实际情况修改）
ANALYSIS_RESULTS="./sae_analysis/sae_analysis_layer15.json"
INTERVENTION_RESULTS="./sae_intervention/intervention_results_layer15.json"

# 输出目录
OUTPUT_DIR="./sae_visualizations"

# 检查文件是否存在
if [ ! -f "${ANALYSIS_RESULTS}" ]; then
    echo "ERROR: Analysis results not found at ${ANALYSIS_RESULTS}"
    exit 1
fi

echo "=========================================="
echo "Generating Visualizations"
echo "=========================================="
echo "Analysis Results: ${ANALYSIS_RESULTS}"
echo "Intervention Results: ${INTERVENTION_RESULTS}"
echo "=========================================="

python visualize_sae_results.py \
    --analysis_results "${ANALYSIS_RESULTS}" \
    --intervention_results "${INTERVENTION_RESULTS}" \
    --output_dir "${OUTPUT_DIR}"

echo "=========================================="
echo "Visualization complete!"
echo "=========================================="

