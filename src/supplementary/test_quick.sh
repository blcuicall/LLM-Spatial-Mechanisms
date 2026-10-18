#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 快速测试脚本 - 用少量样本测试是否能正常运行

set -e

echo "======================================================================"
echo "Quick Test: Running experiments with 10 samples"
echo "======================================================================"

# 查找 SAE checkpoint
SAE_DIR=$(ls -d ../sae_saelens/sae_results/L8_F2048_* 2>/dev/null | head -n 1)

if [ -z "$SAE_DIR" ]; then
    echo "Error: No SAE checkpoint found"
    echo "Please train SAE first or adjust the path"
    exit 1
fi

echo "Using SAE: $SAE_DIR"

# 数据文件
DATA_FILE="${REPO_ROOT}/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test_with_prompt.json"

if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

echo "Using data: $DATA_FILE"
echo ""

# 测试 Dose-Response（最容易出问题的）
echo "[1/3] Testing Dose-Response..."
python dose_response_experiment.py \
  -c "$SAE_DIR" \
  -e "$DATA_FILE" \
  --k_values "5,10" \
  --ablation_type "mean" \
  --max_samples 10 \
  --device cuda:0

echo "✓ Dose-Response test passed"
echo ""

# 测试 Keep-Spatial-Only
echo "[2/3] Testing Keep-Spatial-Only..."
python keep_spatial_only_experiment.py \
  -c "$SAE_DIR" \
  -e "$DATA_FILE" \
  --ablation_type "mean" \
  --max_samples 10 \
  --device cuda:0

echo "✓ Keep-Spatial-Only test passed"
echo ""

# 测试 Statistical Significance
echo "[3/3] Testing Statistical Significance..."
python statistical_significance_experiment.py \
  -c "$SAE_DIR" \
  -e "$DATA_FILE" \
  --n_random_trials 2 \
  --max_samples 10 \
  --device cuda:0

echo "✓ Statistical Significance test passed"
echo ""

echo "======================================================================"
echo "All quick tests passed! ✓"
echo "======================================================================"
echo ""
echo "Now you can run the full experiments with more samples:"
echo "  bash run_all_experiments.sh \"$SAE_DIR\" 200"
echo ""




