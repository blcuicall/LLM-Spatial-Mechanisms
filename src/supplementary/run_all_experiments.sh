#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 一键运行所有补充实验
# 
# 用法:
#   bash run_all_experiments.sh <SAE_CHECKPOINT_DIR> [MAX_SAMPLES] [DEVICE]
#
# 示例:
#   bash run_all_experiments.sh ../sae_saelens/sae_results/L8_F2048_20241231_123456 200 cuda:0

set -e  # 遇到错误立即退出

# =========================
# 参数解析
# =========================
if [ -z "$1" ]; then
    echo "Error: Please provide SAE checkpoint directory"
    echo "Usage: bash run_all_experiments.sh <SAE_CHECKPOINT_DIR> [MAX_SAMPLES] [DEVICE]"
    echo "Example: bash run_all_experiments.sh ../sae_saelens/sae_results/L8_F2048_* 200 cuda:0"
    exit 1
fi

SAE_CHECKPOINT="$1"
MAX_SAMPLES="${2:-200}"  # 默认 200
DEVICE="${3:-cuda:0}"     # 默认 cuda:0

# 数据文件（根据你的实际路径修改）
EVAL_DATA="${REPO_ROOT}/src/data_generation/task_family_1/data_relation/spatial_reasoning_dataset_EN_test_with_prompt.json"

# 检查数据文件是否存在
if [ ! -f "$EVAL_DATA" ]; then
    echo "Warning: Eval data not found at $EVAL_DATA"
    echo "Please update EVAL_DATA path in this script"
    exit 1
fi

# 检查 SAE checkpoint 是否存在
if [ ! -f "$SAE_CHECKPOINT/sae_checkpoint.pt" ]; then
    echo "Error: SAE checkpoint not found at $SAE_CHECKPOINT/sae_checkpoint.pt"
    exit 1
fi

echo "======================================================================"
echo "Running All Supplementary Experiments"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  SAE Checkpoint: $SAE_CHECKPOINT"
echo "  Eval Data: $EVAL_DATA"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Device: $DEVICE"
echo ""
echo "======================================================================"

# 查找梯度归因结果（如果存在）
GRADIENT_RESULTS=""
if [ -d "$SAE_CHECKPOINT/gradient_attribution" ]; then
    GRADIENT_FILE=$(ls $SAE_CHECKPOINT/gradient_attribution/gradient_attribution_grad_x_act_*.json 2>/dev/null | head -n 1)
    if [ -n "$GRADIENT_FILE" ]; then
        GRADIENT_RESULTS="$GRADIENT_FILE"
        echo "✓ Found gradient attribution results: $GRADIENT_RESULTS"
    else
        echo "⚠ Gradient attribution directory exists but no results found"
        echo "  Will use random ordering for dose-response experiment"
    fi
else
    echo "⚠ No gradient attribution results found"
    echo "  Will use random ordering for dose-response experiment"
fi

echo ""

# =========================
# 实验 1: Dose-Response
# =========================
echo "======================================================================"
echo "[1/6] Running Dose-Response Experiment"
echo "======================================================================"

DOSE_RESPONSE_CMD="python dose_response_experiment.py \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  --k_values '5,10,20,40,80,136' \
  --ablation_type mean \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE"

if [ -n "$GRADIENT_RESULTS" ]; then
    DOSE_RESPONSE_CMD="$DOSE_RESPONSE_CMD -g $GRADIENT_RESULTS"
fi

echo "Command: $DOSE_RESPONSE_CMD"
echo ""

eval $DOSE_RESPONSE_CMD

if [ $? -eq 0 ]; then
    echo "✓ Dose-Response experiment completed successfully"
else
    echo "✗ Dose-Response experiment failed"
    exit 1
fi

echo ""

# =========================
# 实验 2: Keep-Spatial-Only
# =========================
echo "======================================================================"
echo "[2/6] Running Keep-Spatial-Only Experiment"
echo "======================================================================"

python keep_spatial_only_experiment.py \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  --ablation_type mean \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE

if [ $? -eq 0 ]; then
    echo "✓ Keep-Spatial-Only experiment completed successfully"
else
    echo "✗ Keep-Spatial-Only experiment failed"
    exit 1
fi

echo ""

# =========================
# 实验 3: Statistical Significance
# =========================
echo "======================================================================"
echo "[3/6] Running Statistical Significance Experiment"
echo "======================================================================"

python statistical_significance_experiment.py \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  --n_random_trials 5 \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE

if [ $? -eq 0 ]; then
    echo "✓ Statistical Significance experiment completed successfully"
else
    echo "✗ Statistical Significance experiment failed"
    exit 1
fi

echo ""

# =========================
# 实验 4: Ablation Robustness
# =========================
echo "======================================================================"
echo "[4/6] Running Ablation Robustness Experiment"
echo "======================================================================"

python ablation_robustness_experiment.py \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE

if [ $? -eq 0 ]; then
    echo "✓ Ablation Robustness experiment completed successfully"
else
    echo "✗ Ablation Robustness experiment failed"
    exit 1
fi

echo ""

# =========================
# 实验 5: Layer Localization
# =========================
echo "======================================================================"
echo "[5/6] Running Layer Localization Experiment"
echo "======================================================================"

# 获取 SAE results 目录（包含多个层）
SAE_RESULTS_DIR=$(dirname $SAE_CHECKPOINT)

# 检查是否有多个层
NUM_LAYERS=$(ls -d $SAE_RESULTS_DIR/L*_F* 2>/dev/null | wc -l)

if [ $NUM_LAYERS -gt 1 ]; then
    echo "Found $NUM_LAYERS layer checkpoints"
    
    python layer_localization_experiment.py \
      -s $SAE_RESULTS_DIR \
      -e $EVAL_DATA \
      --ablation_type mean \
      --max_samples $MAX_SAMPLES \
      --device $DEVICE
    
    if [ $? -eq 0 ]; then
        echo "✓ Layer Localization experiment completed successfully"
    else
        echo "✗ Layer Localization experiment failed"
        exit 1
    fi
else
    echo "⚠ Skipping Layer Localization (only 1 layer found)"
    echo "  To run this experiment, train SAE on multiple layers"
fi

echo ""

# =========================
# 实验 6: Matched Random Control
# =========================
echo "======================================================================"
echo "[6/6] Running Matched Random Control Experiment"
echo "======================================================================"

MATCHED_CMD="python matched_random_control_experiment.py \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  --n_random_samples 10 \
  --matching_method both \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE"

if [ -n "$GRADIENT_RESULTS" ]; then
    MATCHED_CMD="$MATCHED_CMD -g $GRADIENT_RESULTS"
fi

eval $MATCHED_CMD

if [ $? -eq 0 ]; then
    echo "✓ Matched Random Control experiment completed successfully"
else
    echo "✗ Matched Random Control experiment failed"
    exit 1
fi

echo ""

# =========================
# 总结
# =========================
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "======================================================================"
echo ""
echo "Results saved in:"
echo "  - $SAE_CHECKPOINT/dose_response_results/"
echo "  - $SAE_CHECKPOINT/keep_spatial_only_results/"
echo "  - $SAE_CHECKPOINT/statistical_significance_results/"
echo "  - $SAE_CHECKPOINT/ablation_robustness_results/"
if [ $NUM_LAYERS -gt 1 ]; then
    echo "  - $SAE_RESULTS_DIR/layer_localization_results/"
fi
echo "  - $SAE_CHECKPOINT/matched_random_results/"
echo ""
echo "Next steps:"
echo "  1. Review the plots (.png files)"
echo "  2. Read the summaries (.txt files)"
echo "  3. Check detailed results (.json files)"
echo "  4. Write up your paper! 📝"
echo ""
echo "======================================================================"




