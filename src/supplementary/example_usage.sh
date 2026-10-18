#!/bin/bash
# 示例用法脚本 - 展示如何使用每个实验脚本
# 
# 注意：修改以下变量以匹配你的实际路径

# =========================
# 配置路径
# =========================

# SAE checkpoint 目录
SAE_CHECKPOINT="../sae_saelens/sae_results/L8_F2048_20241231_123456"

# 评估数据文件
EVAL_DATA="../../data_generation_task_3/spatial_procedure_dataset_EN_test_with_prompt.json"

# 梯度归因结果（可选）
GRADIENT_FILE="$SAE_CHECKPOINT/gradient_attribution/gradient_attribution_grad_x_act_20241231_123456.json"

# 设备
DEVICE="cuda:0"  # 或 "cpu"

# 样本数
MAX_SAMPLES=200  # 快速测试用 20-50，论文用 200-500

# =========================
# 示例 1: Dose-Response Experiment
# =========================
echo "========================================"
echo "Example 1: Dose-Response Experiment"
echo "========================================"

python dose_response_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  -g $GRADIENT_FILE \
  --k_values "5,10,20,40,80,136" \
  --ablation_type "mean" \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE

echo ""
echo "✓ Results saved in: $SAE_CHECKPOINT/dose_response_results/"
echo ""

# =========================
# 示例 2: Keep-Spatial-Only Experiment
# =========================
echo "========================================"
echo "Example 2: Keep-Spatial-Only Experiment"
echo "========================================"

python keep_spatial_only_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  --ablation_type "mean" \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE

echo ""
echo "✓ Results saved in: $SAE_CHECKPOINT/keep_spatial_only_results/"
echo ""

# =========================
# 示例 3: Statistical Significance Experiment
# =========================
echo "========================================"
echo "Example 3: Statistical Significance"
echo "========================================"

python statistical_significance_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  --n_random_trials 5 \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE

echo ""
echo "✓ Results saved in: $SAE_CHECKPOINT/statistical_significance_results/"
echo ""

# =========================
# 示例 4: Ablation Robustness Experiment
# =========================
echo "========================================"
echo "Example 4: Ablation Robustness"
echo "========================================"

python ablation_robustness_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE

echo ""
echo "✓ Results saved in: $SAE_CHECKPOINT/ablation_robustness_results/"
echo ""

# =========================
# 示例 5: Layer Localization Experiment
# =========================
echo "========================================"
echo "Example 5: Layer Localization"
echo "========================================"
echo "Note: Requires multiple layer SAE results"
echo ""

# SAE results 目录（包含多个层）
SAE_RESULTS_DIR="../sae_saelens/sae_results/"

python layer_localization_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -s $SAE_RESULTS_DIR \
  -e $EVAL_DATA \
  --ablation_type "mean" \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE

echo ""
echo "✓ Results saved in: $SAE_RESULTS_DIR/layer_localization_results/"
echo ""

# =========================
# 示例 6: Matched Random Control Experiment
# =========================
echo "========================================"
echo "Example 6: Matched Random Control"
echo "========================================"

python matched_random_control_experiment.py \
  -m "Qwen/Qwen2.5-7B-Instruct" \
  -c $SAE_CHECKPOINT \
  -e $EVAL_DATA \
  -g $GRADIENT_FILE \
  --n_random_samples 10 \
  --matching_method "both" \
  --max_samples $MAX_SAMPLES \
  --device $DEVICE

echo ""
echo "✓ Results saved in: $SAE_CHECKPOINT/matched_random_results/"
echo ""

# =========================
# 完成
# =========================
echo "========================================"
echo "All examples completed!"
echo "========================================"
echo ""
echo "Check the generated plots and summaries in the results directories."
echo ""




