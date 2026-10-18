"""
Matched Random Control Experiment: 按幅值/稀疏度匹配的随机对照
==================================================================

目的：普通 random 可能不公平：随机特征可能平均幅值更大/更小

做法：从非空间特征里抽样，使其
- activation 均值/方差分布
- |attribution| 分布（或 top-n）
尽量匹配空间特征，再 ablate

价值：一锤定音的特异性控制
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
parser = argparse.ArgumentParser(description='Matched Random Control Experiment')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True)
parser.add_argument("--eval_data_file", "-e", type=str, required=True)
parser.add_argument("--gradient_results", "-g", type=str, default=None,
                    help="Path to gradient attribution results (for attribution matching)")
parser.add_argument("--output_dir", "-o", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--n_random_samples", type=int, default=10,
                    help="Number of matched random samples to try")
parser.add_argument("--matching_method", type=str, default="activation", 
                    choices=["activation", "attribution", "both"],
                    help="What to match: activation stats, attribution, or both")
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
    output_dir = checkpoint_dir / "matched_random_results"
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
non_spatial_feature_ids = [i for i in range(N_FEATURES) if i not in spatial_feature_ids]

print(f"Spatial features: {len(spatial_feature_ids)}")
print(f"Non-spatial features: {len(non_spatial_feature_ids)}")

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
# SimpleSAE
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
# 计算特征统计（用于匹配）
# =========================
print("\n" + "="*60)
print("Computing feature statistics for matching...")
print("="*60)

# 收集所有特征的激活值
all_feature_activations = []

for sample in tqdm(eval_data[:min(200, len(eval_data))], desc="Collecting activations"):
    prompt = sample.get("prompt", sample.get("question", ""))
    tokens = model.to_tokens(prompt, truncate=True)
    
    cache = {}
    def capture_act(act, hook):
        cache['act'] = act[:, -1, :].detach().cpu()
        return act
    
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(HOOK_POINT, capture_act)]):
            _ = model(tokens)
    
    if 'act' in cache:
        # Encode to get feature activations
        with torch.no_grad():
            h = sae.encode(cache['act'].to(DEVICE))
            all_feature_activations.append(h.cpu())

# Stack all activations
all_feature_activations = torch.cat(all_feature_activations, dim=0)  # [n_samples, n_features]

print(f"Collected activations: {all_feature_activations.shape}")

# 计算每个特征的统计量
feature_stats = {}
for feat_id in range(N_FEATURES):
    feat_acts = all_feature_activations[:, feat_id].numpy()
    
    feature_stats[feat_id] = {
        'mean': float(np.mean(feat_acts)),
        'std': float(np.std(feat_acts)),
        'sparsity': float(np.mean(feat_acts > 0)),  # 非零比例
        'max': float(np.max(feat_acts)),
        'median': float(np.median(feat_acts)),
    }

print(f"✓ Computed statistics for {N_FEATURES} features")

# 加载 attribution (如果提供)
feature_attributions = None
if args.gradient_results:
    grad_results_path = Path(args.gradient_results)
    if grad_results_path.exists():
        print("\nLoading gradient attributions...")
        with open(grad_results_path, 'r') as f:
            grad_data = json.load(f)
        
        avg_attributions = np.array(grad_data['attribution_results']['average_attributions'])
        feature_attributions = {i: float(np.abs(avg_attributions[i])) for i in range(len(avg_attributions))}
        print(f"✓ Loaded attributions for {len(feature_attributions)} features")
    else:
        print(f"Warning: {grad_results_path} not found")

# =========================
# 匹配函数
# =========================
def compute_matching_score(spatial_ids: List[int], candidate_ids: List[int], 
                          feature_stats_dict: Dict, attributions: Optional[Dict] = None) -> float:
    """
    计算候选特征与空间特征的匹配度
    分数越低 = 越匹配
    """
    # 提取统计量
    spatial_means = np.array([feature_stats_dict[i]['mean'] for i in spatial_ids])
    spatial_stds = np.array([feature_stats_dict[i]['std'] for i in spatial_ids])
    spatial_sparsity = np.array([feature_stats_dict[i]['sparsity'] for i in spatial_ids])
    
    candidate_means = np.array([feature_stats_dict[i]['mean'] for i in candidate_ids])
    candidate_stds = np.array([feature_stats_dict[i]['std'] for i in candidate_ids])
    candidate_sparsity = np.array([feature_stats_dict[i]['sparsity'] for i in candidate_ids])
    
    # KS test for distribution similarity
    ks_mean = stats.ks_2samp(spatial_means, candidate_means).statistic
    ks_std = stats.ks_2samp(spatial_stds, candidate_stds).statistic
    ks_sparsity = stats.ks_2samp(spatial_sparsity, candidate_sparsity).statistic
    
    score = ks_mean + ks_std + ks_sparsity
    
    # 如果有 attribution，也匹配
    if attributions is not None and args.matching_method in ["attribution", "both"]:
        spatial_attr = np.array([attributions.get(i, 0) for i in spatial_ids])
        candidate_attr = np.array([attributions.get(i, 0) for i in candidate_ids])
        ks_attr = stats.ks_2samp(spatial_attr, candidate_attr).statistic
        score += ks_attr
    
    return score

# =========================
# 选择匹配的随机特征
# =========================
print("\n" + "="*60)
print(f"Selecting matched random features ({args.n_random_samples} samples)...")
print("="*60)

matched_random_sets = []
n_spatial = len(spatial_feature_ids)

for trial in range(args.n_random_samples):
    # 随机抽样
    candidates = random.sample(non_spatial_feature_ids, n_spatial)
    
    # 计算匹配分数
    score = compute_matching_score(spatial_feature_ids, candidates, feature_stats, feature_attributions)
    
    matched_random_sets.append({
        'ids': candidates,
        'score': score,
    })
    
    print(f"  Trial {trial+1}: score = {score:.4f}")

# 选择最佳匹配
matched_random_sets.sort(key=lambda x: x['score'])
best_matched_ids = matched_random_sets[0]['ids']
best_score = matched_random_sets[0]['score']

print(f"\n✓ Best matched random features (score = {best_score:.4f})")

# 也选择一个完全随机的对照（不匹配）
unmatched_random_ids = random.sample(non_spatial_feature_ids, n_spatial)

# 打印匹配质量
print("\n" + "="*60)
print("Matching Quality Check")
print("="*60)

def print_distribution_stats(ids, name):
    means = [feature_stats[i]['mean'] for i in ids]
    stds = [feature_stats[i]['std'] for i in ids]
    sparsities = [feature_stats[i]['sparsity'] for i in ids]
    
    print(f"\n{name}:")
    print(f"  Mean activation: {np.mean(means):.4f} ± {np.std(means):.4f}")
    print(f"  Std activation: {np.mean(stds):.4f} ± {np.std(stds):.4f}")
    print(f"  Sparsity: {np.mean(sparsities):.4f} ± {np.std(sparsities):.4f}")
    
    if feature_attributions:
        attrs = [feature_attributions.get(i, 0) for i in ids]
        print(f"  Attribution: {np.mean(attrs):.6f} ± {np.std(attrs):.6f}")

print_distribution_stats(spatial_feature_ids, "Spatial features")
print_distribution_stats(best_matched_ids, "Matched random features")
print_distribution_stats(unmatched_random_ids, "Unmatched random features")

# =========================
# 评估函数
# =========================
def evaluate_with_metrics(
    model: HookedTransformer,
    data: List[Dict],
    ablator: Optional[SAEAblator] = None,
    desc: str = "Evaluating"
) -> Dict:
    """评估并返回详细指标"""
    results = []
    correct = 0
    total = 0
    
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
        
        correct_logit = option_logits[correct_option]
        other_logits = [v for k, v in option_logits.items() if k != correct_option]
        max_other_logit = max(other_logits)
        margin = correct_logit - max_other_logit
        
        all_logits = torch.tensor([option_logits[opt] for opt in sorted(OPTION_TOKENS.keys())])
        probs = torch.softmax(all_logits, dim=0)
        correct_idx = sorted(OPTION_TOKENS.keys()).index(correct_option)
        nll = -torch.log(probs[correct_idx] + 1e-10).item()
        
        results.append({
            'correct': is_correct,
            'margin': margin,
            'nll': nll,
        })
        
        if is_correct:
            correct += 1
        total += 1
    
    return {
        'accuracy': correct / total if total > 0 else 0.0,
        'correct': correct,
        'total': total,
        'avg_margin': np.mean([r['margin'] for r in results]),
        'avg_nll': np.mean([r['nll'] for r in results]),
    }

# =========================
# 主实验
# =========================
print("\n" + "="*60)
print("Running Matched Random Control Experiment")
print("="*60)

# 1. Baseline
print("\n[1/4] Baseline: No ablation...")
baseline_results = evaluate_with_metrics(model, eval_data, ablator=None, desc="Baseline")
print(f"✓ Acc: {baseline_results['accuracy']:.4f}, "
      f"Margin: {baseline_results['avg_margin']:.4f}, "
      f"NLL: {baseline_results['avg_nll']:.4f}")

# 2. Spatial ablation
print("\n[2/4] Spatial ablation...")
spatial_ablator = SAEAblator(sae, spatial_feature_ids)
spatial_results = evaluate_with_metrics(model, eval_data, ablator=spatial_ablator, desc="Spatial")
print(f"✓ Acc: {spatial_results['accuracy']:.4f}, "
      f"Margin: {spatial_results['avg_margin']:.4f}, "
      f"NLL: {spatial_results['avg_nll']:.4f}")

# 3. Matched random ablation
print("\n[3/4] Matched random ablation...")
matched_ablator = SAEAblator(sae, best_matched_ids)
matched_results = evaluate_with_metrics(model, eval_data, ablator=matched_ablator, desc="Matched random")
print(f"✓ Acc: {matched_results['accuracy']:.4f}, "
      f"Margin: {matched_results['avg_margin']:.4f}, "
      f"NLL: {matched_results['avg_nll']:.4f}")

# 4. Unmatched random ablation
print("\n[4/4] Unmatched random ablation...")
unmatched_ablator = SAEAblator(sae, unmatched_random_ids)
unmatched_results = evaluate_with_metrics(model, eval_data, ablator=unmatched_ablator, desc="Unmatched random")
print(f"✓ Acc: {unmatched_results['accuracy']:.4f}, "
      f"Margin: {unmatched_results['avg_margin']:.4f}, "
      f"NLL: {unmatched_results['avg_nll']:.4f}")

# =========================
# 结果分析
# =========================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

baseline_acc = baseline_results['accuracy']

conditions = [
    ("Baseline", baseline_results),
    ("Spatial ablation", spatial_results),
    ("Matched random", matched_results),
    ("Unmatched random", unmatched_results),
]

print(f"\n{'Condition':<20} {'Accuracy':<12} {'Δ Acc':<12} {'Margin':<12} {'NLL':<12}")
print("-" * 68)

for name, res in conditions:
    acc = res['accuracy']
    delta = acc - baseline_acc
    margin = res['avg_margin']
    nll = res['avg_nll']
    print(f"{name:<20} {acc:<12.4f} {delta:<+12.4f} {margin:<12.4f} {nll:<12.4f}")

# 关键对比
print("\n" + "="*60)
print("KEY COMPARISONS")
print("="*60)

spatial_drop = baseline_acc - spatial_results['accuracy']
matched_drop = baseline_acc - matched_results['accuracy']
unmatched_drop = baseline_acc - unmatched_results['accuracy']

print(f"\nAccuracy drops:")
print(f"  Spatial: {spatial_drop:.4f}")
print(f"  Matched random: {matched_drop:.4f}")
print(f"  Unmatched random: {unmatched_drop:.4f}")

print(f"\nRatios:")
print(f"  Spatial / Matched: {spatial_drop / matched_drop if matched_drop > 0 else float('inf'):.2f}x")
print(f"  Spatial / Unmatched: {spatial_drop / unmatched_drop if unmatched_drop > 0 else float('inf'):.2f}x")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if spatial_drop > matched_drop * 1.5:
    print("✓ STRONG EVIDENCE: Even with matched controls, spatial features are more important")
    print("  → Result is NOT due to differences in activation statistics")
elif spatial_drop > matched_drop * 1.2:
    print("✓ MODERATE EVIDENCE: Spatial features show specific importance")
else:
    print("✗ WEAK EVIDENCE: Spatial features not more important than matched controls")
    print("  → Effect may be confounded by activation statistics")

# =========================
# 可视化
# =========================
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Accuracy comparison
condition_names = ["Baseline", "Spatial", "Matched\nRandom", "Unmatched\nRandom"]
accuracies = [r[1]['accuracy'] for r in conditions]
colors = ['steelblue', 'coral', 'mediumseagreen', 'gray']

axes[0, 0].bar(condition_names, accuracies, color=colors, alpha=0.8)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Accuracy drops
drops = [0, spatial_drop, matched_drop, unmatched_drop]
axes[0, 1].bar(condition_names, drops, color=colors, alpha=0.8)
axes[0, 1].set_ylabel('Accuracy Drop', fontsize=12)
axes[0, 1].set_title('Accuracy Drop Comparison', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Feature statistics distributions
spatial_means = [feature_stats[i]['mean'] for i in spatial_feature_ids]
matched_means = [feature_stats[i]['mean'] for i in best_matched_ids]
unmatched_means = [feature_stats[i]['mean'] for i in unmatched_random_ids]

axes[1, 0].hist(spatial_means, bins=20, alpha=0.5, label='Spatial', color='coral')
axes[1, 0].hist(matched_means, bins=20, alpha=0.5, label='Matched', color='mediumseagreen')
axes[1, 0].hist(unmatched_means, bins=20, alpha=0.5, label='Unmatched', color='gray')
axes[1, 0].set_xlabel('Mean Activation', fontsize=11)
axes[1, 0].set_ylabel('Count', fontsize=11)
axes[1, 0].set_title('Feature Activation Distribution', fontsize=12, fontweight='bold')
axes[1, 0].legend()

# Plot 4: Sparsity distribution
spatial_sparsity = [feature_stats[i]['sparsity'] for i in spatial_feature_ids]
matched_sparsity = [feature_stats[i]['sparsity'] for i in best_matched_ids]
unmatched_sparsity = [feature_stats[i]['sparsity'] for i in unmatched_random_ids]

axes[1, 1].hist(spatial_sparsity, bins=20, alpha=0.5, label='Spatial', color='coral')
axes[1, 1].hist(matched_sparsity, bins=20, alpha=0.5, label='Matched', color='mediumseagreen')
axes[1, 1].hist(unmatched_sparsity, bins=20, alpha=0.5, label='Unmatched', color='gray')
axes[1, 1].set_xlabel('Sparsity', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Feature Sparsity Distribution', fontsize=12, fontweight='bold')
axes[1, 1].legend()

plt.tight_layout()
plot_file = output_dir / f"matched_random_control_{timestamp}.png"
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
        'matching_method': args.matching_method,
        'n_random_samples': args.n_random_samples,
        'best_matching_score': float(best_score),
    },
    'results': {
        'baseline': {
            'accuracy': float(baseline_results['accuracy']),
            'avg_margin': float(baseline_results['avg_margin']),
            'avg_nll': float(baseline_results['avg_nll']),
        },
        'spatial': {
            'accuracy': float(spatial_results['accuracy']),
            'avg_margin': float(spatial_results['avg_margin']),
            'avg_nll': float(spatial_results['avg_nll']),
            'drop': float(spatial_drop),
        },
        'matched_random': {
            'accuracy': float(matched_results['accuracy']),
            'avg_margin': float(matched_results['avg_margin']),
            'avg_nll': float(matched_results['avg_nll']),
            'drop': float(matched_drop),
        },
        'unmatched_random': {
            'accuracy': float(unmatched_results['accuracy']),
            'avg_margin': float(unmatched_results['avg_margin']),
            'avg_nll': float(unmatched_results['avg_nll']),
            'drop': float(unmatched_drop),
        },
    },
    'comparisons': {
        'spatial_vs_matched_ratio': float(spatial_drop / matched_drop if matched_drop > 0 else 0),
        'spatial_vs_unmatched_ratio': float(spatial_drop / unmatched_drop if unmatched_drop > 0 else 0),
    }
}

results_file = output_dir / f"matched_random_control_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved: {results_file}")

print("\n" + "="*60)
print("MATCHED RANDOM CONTROL EXPERIMENT COMPLETE")
print("="*60)

if spatial_drop > matched_drop * 1.5:
    print("\n✓ Conclusion: Spatial features show SPECIFIC importance")
    print("  even when controlling for activation statistics")

