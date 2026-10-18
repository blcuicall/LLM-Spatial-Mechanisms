"""
Statistical Significance Experiment: 更敏感的指标 + 显著性检验
================================================================

目的：避免"3 个点只是噪声"的质疑，并提升统计可信度

做法：
1. 对每样本记录：
   - 正确选项 margin（logit* − max other）
   - NLL / log-prob(y*)
2. 进行 paired t-test 或 Wilcoxon 比较：
   - Spatial ablation vs Random ablation
   - Baseline vs Spatial ablation
   - Baseline vs Random ablation
"""

import sys
sys.path.append("../")
sys.path.append("../../")
from config import PATHS

import os
import json
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Statistical Significance Experiment')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True)
parser.add_argument("--eval_data_file", "-e", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--n_random_trials", type=int, default=5,
                    help="Number of random ablation trials for robustness")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

# =========================
# 基本配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = args.device if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
SEED = args.seed

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

checkpoint_dir = Path(args.sae_checkpoint)
if args.output_dir is None:
    output_dir = checkpoint_dir / "statistical_significance_results"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime('%Y%m%d_%H%M%S')

# =========================
# 加载 SAE
# =========================
print("\n" + "="*60)
print("Loading SAE checkpoint...")
print("="*60)

checkpoint_file = checkpoint_dir / 'sae_checkpoint.pt'
checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
sae_config = checkpoint['config']
sae_state = checkpoint['model_state_dict']

LAYER_IDX = sae_config['layer']
N_FEATURES = sae_config['n_features']
D_IN = sae_config['d_in']

W_enc = sae_state['W_enc'].to(DEVICE)
b_enc = sae_state['b_enc'].to(DEVICE)
W_dec = sae_state['W_dec'].to(DEVICE)
b_dec = sae_state.get('b_dec', torch.zeros(D_IN)).to(DEVICE)

print(f"SAE: Layer {LAYER_IDX}, {N_FEATURES} features")
HOOK_POINT = f"blocks.{LAYER_IDX}.hook_mlp_out"

# =========================
# 加载 Spatial Features
# =========================
print("\n" + "="*60)
print("Loading spatial features...")
print("="*60)

analysis_dir = checkpoint_dir / "analysis"
dimension_features_file = analysis_dir / "dimension_features.json"

with open(dimension_features_file, 'r') as f:
    dimension_features = json.load(f)

spatial_feature_ids = set()
for key in dimension_features:
    for feat in dimension_features[key]:
        spatial_feature_ids.add(feat['feature_idx'])

spatial_feature_ids = sorted(list(spatial_feature_ids))
print(f"Spatial features: {len(spatial_feature_ids)}")

# =========================
# 加载模型
# =========================
print("\n" + "="*60)
print("Loading model...")
print("="*60)

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    trust_remote_code=True
)
hf_model = hf_model.to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = HookedTransformer.from_pretrained(
    MODEL_NAME,
    hf_model=hf_model,
    tokenizer=tokenizer,
    dtype=DTYPE,
    device=DEVICE,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    fold_value_biases=False,
)
model.eval()

# =========================
# SimpleSAE + Ablator
# =========================
class SimpleSAE:
    def __init__(self, W_enc, b_enc, W_dec, b_dec, device):
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec
        self.device = device
    
    def encode(self, x):
        x_centered = x - self.b_dec
        pre_relu = x_centered @ self.W_enc + self.b_enc
        return torch.nn.functional.relu(pre_relu)
    
    def decode(self, h):
        return h @ self.W_dec + self.b_dec

sae = SimpleSAE(W_enc, b_enc, W_dec, b_dec, DEVICE)

class SAEAblator:
    def __init__(self, sae: SimpleSAE, feature_ids: List[int]):
        self.sae = sae
        self.feature_ids = feature_ids
        
    def __call__(self, act, hook):
        original_shape = act.shape
        
        if len(original_shape) == 3:
            batch_size, seq_len, d = original_shape
            act_flat = act.reshape(-1, d)
        else:
            act_flat = act
        
        with torch.no_grad():
            h = self.sae.encode(act_flat)
            h[:, self.feature_ids] = 0.0
            act_hat = self.sae.decode(h)
        
        if len(original_shape) == 3:
            act_hat = act_hat.reshape(batch_size, seq_len, d)
        
        return act_hat

# =========================
# 加载评估数据
# =========================
print("\n" + "="*60)
print("Loading evaluation data...")
print("="*60)

with open(args.eval_data_file, 'r') as f:
    eval_data = json.load(f)

if args.max_samples:
    eval_data = eval_data[:args.max_samples]

print(f"Loaded {len(eval_data)} samples")

OPTION_TOKENS = {}
for option in ['A', 'B', 'C', 'D']:
    tokens = tokenizer.encode(option, add_special_tokens=False)
    if len(tokens) > 0:
        OPTION_TOKENS[option] = tokens[0]

# =========================
# 评估函数（收集逐样本指标）
# =========================
def evaluate_per_sample(
    model: HookedTransformer,
    data: List[Dict],
    ablator: Optional[SAEAblator] = None,
    desc: str = "Evaluating"
) -> List[Dict]:
    """
    返回每个样本的详细指标
    """
    per_sample_results = []
    
    for sample in tqdm(data, desc=desc):
        prompt = sample.get("prompt", sample.get("question", ""))
        correct_option = sample.get("correct_option", sample.get("answer", "A"))
        
        tokens = model.to_tokens(prompt, truncate=True)
        
        if ablator is not None:
            with model.hooks(fwd_hooks=[(HOOK_POINT, ablator)]):
                with torch.no_grad():
                    logits = model(tokens)
        else:
            with torch.no_grad():
                logits = model(tokens)
        
        last_logits = logits[0, -1, :]
        option_logits = {opt: last_logits[OPTION_TOKENS[opt]].item() 
                        for opt in OPTION_TOKENS}
        
        pred_option = max(option_logits, key=option_logits.get)
        is_correct = (pred_option == correct_option)
        
        # Margin: correct - max(other)
        correct_logit = option_logits[correct_option]
        other_logits = [v for k, v in option_logits.items() if k != correct_option]
        max_other_logit = max(other_logits)
        margin = correct_logit - max_other_logit
        
        # NLL
        all_logits = torch.tensor([option_logits[opt] for opt in sorted(OPTION_TOKENS.keys())])
        probs = torch.softmax(all_logits, dim=0)
        correct_idx = sorted(OPTION_TOKENS.keys()).index(correct_option)
        log_prob = torch.log(probs[correct_idx] + 1e-10).item()
        nll = -log_prob
        
        # Confidence (probability of predicted option)
        pred_idx = sorted(OPTION_TOKENS.keys()).index(pred_option)
        confidence = probs[pred_idx].item()
        
        per_sample_results.append({
            'sample_id': sample.get('id', len(per_sample_results)),
            'correct': 1 if is_correct else 0,
            'margin': margin,
            'nll': nll,
            'log_prob': log_prob,
            'confidence': confidence,
            'correct_logit': correct_logit,
            'pred_logit': option_logits[pred_option],
        })
    
    return per_sample_results

# =========================
# 主实验
# =========================
print("\n" + "="*60)
print("Running Statistical Significance Experiment")
print("="*60)

# 1. Baseline
print("\n[1] Baseline: No ablation...")
baseline_results = evaluate_per_sample(model, eval_data, ablator=None, desc="Baseline")

# 2. Spatial ablation
print("\n[2] Spatial ablation...")
spatial_ablator = SAEAblator(sae, spatial_feature_ids)
spatial_results = evaluate_per_sample(model, eval_data, ablator=spatial_ablator, desc="Spatial")

# 3. Random ablation (multiple trials)
print(f"\n[3] Random ablation ({args.n_random_trials} trials)...")
random_results_list = []

for trial in range(args.n_random_trials):
    # 每次选择不同的随机特征
    available_ids = [i for i in range(N_FEATURES) if i not in spatial_feature_ids]
    random_ids = random.sample(available_ids, min(len(spatial_feature_ids), len(available_ids)))
    
    random_ablator = SAEAblator(sae, random_ids)
    trial_results = evaluate_per_sample(
        model, eval_data, ablator=random_ablator, 
        desc=f"Random trial {trial+1}/{args.n_random_trials}"
    )
    random_results_list.append(trial_results)

# 平均 random trials
random_results = []
for i in range(len(eval_data)):
    avg_metrics = {
        'sample_id': baseline_results[i]['sample_id'],
        'correct': np.mean([trial[i]['correct'] for trial in random_results_list]),
        'margin': np.mean([trial[i]['margin'] for trial in random_results_list]),
        'nll': np.mean([trial[i]['nll'] for trial in random_results_list]),
        'log_prob': np.mean([trial[i]['log_prob'] for trial in random_results_list]),
        'confidence': np.mean([trial[i]['confidence'] for trial in random_results_list]),
    }
    random_results.append(avg_metrics)

# =========================
# 统计检验
# =========================
print("\n" + "="*60)
print("Statistical Tests")
print("="*60)

# 提取指标向量
baseline_margins = np.array([r['margin'] for r in baseline_results])
spatial_margins = np.array([r['margin'] for r in spatial_results])
random_margins = np.array([r['margin'] for r in random_results])

baseline_nlls = np.array([r['nll'] for r in baseline_results])
spatial_nlls = np.array([r['nll'] for r in spatial_results])
random_nlls = np.array([r['nll'] for r in random_results])

baseline_correct = np.array([r['correct'] for r in baseline_results])
spatial_correct = np.array([r['correct'] for r in spatial_results])
random_correct = np.array([r['correct'] for r in random_results])

# Paired t-test
def paired_test(x, y, name_x, name_y, metric_name):
    """Perform paired t-test and Wilcoxon test"""
    # T-test (parametric)
    t_stat, t_pval = stats.ttest_rel(x, y)
    
    # Wilcoxon signed-rank test (non-parametric)
    w_stat, w_pval = stats.wilcoxon(x, y)
    
    # Effect size (Cohen's d)
    diff = x - y
    cohen_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
    
    print(f"\n{name_x} vs {name_y} ({metric_name}):")
    print(f"  Mean diff: {np.mean(diff):.4f}")
    print(f"  Paired t-test: t={t_stat:.4f}, p={t_pval:.4e}")
    print(f"  Wilcoxon test: W={w_stat:.1f}, p={w_pval:.4e}")
    print(f"  Cohen's d: {cohen_d:.4f}")
    print(f"  Significant (p<0.05): {t_pval < 0.05}")
    
    return {
        'mean_diff': float(np.mean(diff)),
        't_statistic': float(t_stat),
        't_pvalue': float(t_pval),
        'wilcoxon_statistic': float(w_stat),
        'wilcoxon_pvalue': float(w_pval),
        'cohens_d': float(cohen_d),
        'significant_005': bool(t_pval < 0.05),
        'significant_001': bool(t_pval < 0.01),
    }

# 进行各种比较
print("\n" + "="*60)
print("MARGIN Comparisons")
print("="*60)

test_results = {}

test_results['baseline_vs_spatial_margin'] = paired_test(
    baseline_margins, spatial_margins, "Baseline", "Spatial", "Margin"
)

test_results['baseline_vs_random_margin'] = paired_test(
    baseline_margins, random_margins, "Baseline", "Random", "Margin"
)

test_results['spatial_vs_random_margin'] = paired_test(
    spatial_margins, random_margins, "Spatial", "Random", "Margin"
)

print("\n" + "="*60)
print("NLL Comparisons")
print("="*60)

test_results['baseline_vs_spatial_nll'] = paired_test(
    baseline_nlls, spatial_nlls, "Baseline", "Spatial", "NLL"
)

test_results['baseline_vs_random_nll'] = paired_test(
    baseline_nlls, random_nlls, "Baseline", "Random", "NLL"
)

test_results['spatial_vs_random_nll'] = paired_test(
    spatial_nlls, random_nlls, "Spatial", "Random", "NLL"
)

# =========================
# 汇总统计
# =========================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

conditions = [
    ("Baseline", baseline_results),
    ("Spatial ablation", spatial_results),
    ("Random ablation", random_results),
]

print(f"\n{'Condition':<20} {'Acc':<10} {'Margin':<12} {'NLL':<12}")
print("-" * 54)
for name, res in conditions:
    acc = np.mean([r['correct'] for r in res])
    margin = np.mean([r['margin'] for r in res])
    nll = np.mean([r['nll'] for r in res])
    print(f"{name:<20} {acc:<10.4f} {margin:<12.4f} {nll:<12.4f}")

# =========================
# 可视化
# =========================
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Distributions
# Plot 1: Margin distribution
axes[0, 0].hist(baseline_margins, bins=30, alpha=0.5, label='Baseline', color='steelblue')
axes[0, 0].hist(spatial_margins, bins=30, alpha=0.5, label='Spatial', color='coral')
axes[0, 0].hist(random_margins, bins=30, alpha=0.5, label='Random', color='mediumseagreen')
axes[0, 0].set_xlabel('Margin', fontsize=11)
axes[0, 0].set_ylabel('Count', fontsize=11)
axes[0, 0].set_title('Margin Distribution', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

# Plot 2: NLL distribution
axes[0, 1].hist(baseline_nlls, bins=30, alpha=0.5, label='Baseline', color='steelblue')
axes[0, 1].hist(spatial_nlls, bins=30, alpha=0.5, label='Spatial', color='coral')
axes[0, 1].hist(random_nlls, bins=30, alpha=0.5, label='Random', color='mediumseagreen')
axes[0, 1].set_xlabel('NLL', fontsize=11)
axes[0, 1].set_ylabel('Count', fontsize=11)
axes[0, 1].set_title('NLL Distribution', fontsize=12, fontweight='bold')
axes[0, 1].legend()

# Plot 3: Correctness
acc_values = [
    np.mean(baseline_correct),
    np.mean(spatial_correct),
    np.mean(random_correct),
]
axes[0, 2].bar(['Baseline', 'Spatial', 'Random'], acc_values, 
               color=['steelblue', 'coral', 'mediumseagreen'], alpha=0.8)
axes[0, 2].set_ylabel('Accuracy', fontsize=11)
axes[0, 2].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
axes[0, 2].set_ylim(0, 1)
axes[0, 2].axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random')
axes[0, 2].legend()

# Row 2: Paired differences
# Plot 4: Baseline vs Spatial (Margin)
diff_baseline_spatial = baseline_margins - spatial_margins
axes[1, 0].hist(diff_baseline_spatial, bins=30, color='coral', alpha=0.7)
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].axvline(x=np.mean(diff_baseline_spatial), color='darkred', linewidth=2, 
                   label=f'Mean: {np.mean(diff_baseline_spatial):.3f}')
axes[1, 0].set_xlabel('Margin Difference (Baseline - Spatial)', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title('Paired Diff: Baseline vs Spatial', fontsize=12, fontweight='bold')
axes[1, 0].legend()

# Plot 5: Baseline vs Random (Margin)
diff_baseline_random = baseline_margins - random_margins
axes[1, 1].hist(diff_baseline_random, bins=30, color='mediumseagreen', alpha=0.7)
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].axvline(x=np.mean(diff_baseline_random), color='darkgreen', linewidth=2,
                   label=f'Mean: {np.mean(diff_baseline_random):.3f}')
axes[1, 1].set_xlabel('Margin Difference (Baseline - Random)', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Paired Diff: Baseline vs Random', fontsize=12, fontweight='bold')
axes[1, 1].legend()

# Plot 6: Spatial vs Random (Margin)
diff_spatial_random = spatial_margins - random_margins
axes[1, 2].hist(diff_spatial_random, bins=30, color='purple', alpha=0.7)
axes[1, 2].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 2].axvline(x=np.mean(diff_spatial_random), color='darkviolet', linewidth=2,
                   label=f'Mean: {np.mean(diff_spatial_random):.3f}')
axes[1, 2].set_xlabel('Margin Difference (Spatial - Random)', fontsize=11)
axes[1, 2].set_ylabel('Count', fontsize=11)
axes[1, 2].set_title('Paired Diff: Spatial vs Random', fontsize=12, fontweight='bold')
axes[1, 2].legend()

plt.tight_layout()
plot_file = output_dir / f"statistical_tests_{timestamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved: {plot_file}")
plt.close()

# =========================
# 保存结果
# =========================
print("\n" + "="*60)
print("Saving results...")
print("="*60)

results = {
    'timestamp': timestamp,
    'config': {
        'model_name': MODEL_NAME,
        'layer': LAYER_IDX,
        'n_features': N_FEATURES,
        'n_spatial_features': len(spatial_feature_ids),
        'eval_samples': len(eval_data),
        'n_random_trials': args.n_random_trials,
    },
    'summary_statistics': {
        'baseline': {
            'accuracy': float(np.mean(baseline_correct)),
            'avg_margin': float(np.mean(baseline_margins)),
            'avg_nll': float(np.mean(baseline_nlls)),
        },
        'spatial': {
            'accuracy': float(np.mean(spatial_correct)),
            'avg_margin': float(np.mean(spatial_margins)),
            'avg_nll': float(np.mean(spatial_nlls)),
        },
        'random': {
            'accuracy': float(np.mean(random_correct)),
            'avg_margin': float(np.mean(random_margins)),
            'avg_nll': float(np.mean(random_nlls)),
        },
    },
    'statistical_tests': test_results,
}

results_file = output_dir / f"statistical_tests_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved: {results_file}")

# Summary report
summary_file = output_dir / f"statistical_summary_{timestamp}.txt"
with open(summary_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("Statistical Significance Experiment\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Layer: {LAYER_IDX}\n")
    f.write(f"Samples: {len(eval_data)}\n")
    f.write(f"Random trials: {args.n_random_trials}\n\n")
    
    f.write("="*60 + "\n")
    f.write("Summary Statistics\n")
    f.write("="*60 + "\n\n")
    for name, res in conditions:
        acc = np.mean([r['correct'] for r in res])
        margin = np.mean([r['margin'] for r in res])
        nll = np.mean([r['nll'] for r in res])
        f.write(f"{name}:\n")
        f.write(f"  Accuracy: {acc:.4f}\n")
        f.write(f"  Margin: {margin:.4f}\n")
        f.write(f"  NLL: {nll:.4f}\n\n")
    
    f.write("="*60 + "\n")
    f.write("Statistical Tests (Baseline vs Spatial)\n")
    f.write("="*60 + "\n\n")
    test = test_results['baseline_vs_spatial_margin']
    f.write(f"Margin:\n")
    f.write(f"  Mean difference: {test['mean_diff']:.4f}\n")
    f.write(f"  p-value (t-test): {test['t_pvalue']:.4e}\n")
    f.write(f"  Cohen's d: {test['cohens_d']:.4f}\n")
    f.write(f"  Significant: {test['significant_005']}\n\n")
    
    test = test_results['baseline_vs_spatial_nll']
    f.write(f"NLL:\n")
    f.write(f"  Mean difference: {test['mean_diff']:.4f}\n")
    f.write(f"  p-value (t-test): {test['t_pvalue']:.4e}\n")
    f.write(f"  Cohen's d: {test['cohens_d']:.4f}\n")
    f.write(f"  Significant: {test['significant_005']}\n\n")
    
    f.write("="*60 + "\n")
    f.write("Conclusion\n")
    f.write("="*60 + "\n\n")
    
    if test_results['baseline_vs_spatial_margin']['significant_001']:
        f.write("✓ STRONG evidence: Spatial ablation significantly hurts performance (p < 0.01)\n")
    elif test_results['baseline_vs_spatial_margin']['significant_005']:
        f.write("✓ Moderate evidence: Spatial ablation hurts performance (p < 0.05)\n")
    else:
        f.write("✗ Weak evidence: No significant difference\n")

print(f"✓ Summary saved: {summary_file}")

print("\n" + "="*60)
print("STATISTICAL SIGNIFICANCE EXPERIMENT COMPLETE")
print("="*60)




