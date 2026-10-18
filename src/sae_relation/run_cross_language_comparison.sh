#!/bin/bash

###############################################################################
# 跨语言SAE分析结果对比工具
# 用于比较同一模型在不同语言数据上的SAE分析结果
###############################################################################

# 使用说明
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --en_dir DIR       英文结果目录路径"
    echo "  --cn_dir DIR       中文结果目录路径"
    echo "  --output DIR       输出目录 (默认: ./cross_language_comparison)"
    echo "  --top_k N          比较的top特征数量 (默认: 50)"
    echo "  --top_n N          比较的top干预特征数量 (默认: 10)"
    echo "  --help             显示此帮助信息"
    echo ""
    echo "Example:"
    echo "  $0 --en_dir outputs_EN --cn_dir outputs_CN"
    echo ""
}

# 默认参数
EN_DIR=""
CN_DIR=""
OUTPUT_DIR="./cross_language_comparison"
TOP_K=50
TOP_N=10

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --en_dir)
            EN_DIR="$2"
            shift 2
            ;;
        --cn_dir)
            CN_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --top_n)
            TOP_N="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$EN_DIR" ] || [ -z "$CN_DIR" ]; then
    echo "ERROR: Both --en_dir and --cn_dir are required!"
    echo ""
    print_usage
    exit 1
fi

# 检查目录是否存在
if [ ! -d "$EN_DIR" ]; then
    echo "ERROR: English directory not found: $EN_DIR"
    exit 1
fi

if [ ! -d "$CN_DIR" ]; then
    echo "ERROR: Chinese directory not found: $CN_DIR"
    exit 1
fi

echo "=========================================="
echo "Cross-Language SAE Analysis Comparison"
echo "=========================================="
echo "English Directory: ${EN_DIR}"
echo "Chinese Directory: ${CN_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Top K Features: ${TOP_K}"
echo "Top N Intervention: ${TOP_N}"
echo "=========================================="
echo ""

# 运行对比分析
python compare_cross_language.py \
    --lang_dirs "EN:${EN_DIR}" "CN:${CN_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --top_k ${TOP_K} \
    --top_n_intervention ${TOP_N}

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Comparison Complete!"
    echo "=========================================="
    echo "Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Generated files:"
    echo "  - regression_comparison.csv & .png"
    echo "  - top_features_comparison.json"
    echo "  - feature_overlap.png"
    echo "  - intervention_comparison.csv & .png (if available)"
    echo "  - importance_correlation.json"
    echo "  - summary_report.txt"
    echo "=========================================="
else
    echo ""
    echo "ERROR: Comparison failed!"
    exit 1
fi

