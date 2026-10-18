"""
Dose-Response Experiment: Top-k 空间特征逐步 Ablate
======================================================

目的：证明"越多、越重要的空间特征被移除 → 性能单调下降"

做法：按 grad_x_act（或|attribution|）对空间特征排序，
     做 k = 5/10/20/40/80/136 的 mean-ablation（或 shuffle）

报告：Acc、margin、NLL 随 k 的曲线；最好单调、最好有拐点（bottleneck）
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

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Dose-Response Ablation Experiment')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True,
                    help="Path to SAE checkpoint directory")
parser.add_argument("--gradient_results", "-g", type=str, default=None,
                    help="Path to gradient attribution results (optional)")
parser.add_argument("--eval_data_file", "-e", type=str, required=True,
                    help="Path to evaluation data file")
parser.add_argument("--output_dir", "-o", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--k_values", type=str, default="5,10,20,40,80,136",
                    help="Comma-separated list of k values")
parser.add_argument("--ablation_type", type=str, default="mean", 
                    choices=["zero", "mean", "shuffle"],
                    help="Type of ablation: zero, mean, or shuffle")
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

# 解析 k values
K_VALUES = [int(k) for k in args.k_values.split(',')]
print(f"K values: {K_VALUES}")

# 输出目录
checkpoint_dir = Path(args.sae_checkpoint)
if args.output_dir is None:
    output_dir = checkpoint_dir / "dose_response_results"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime('%Y%m%d_%H%M%S')
print(f"Timestamp: {timestamp}")

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

if not dimension_features_file.exists():
    print(f"Error: {dimension_features_file} not found!")
    sys.exit(1)

with open(dimension_features_file, 'r') as f:
    dimension_features = json.load(f)

# 收集所有 spatial feature IDs
spatial_feature_ids = set()
for key in dimension_features:
    for feat in dimension_features[key]:
        spatial_feature_ids.add(feat['feature_idx'])

spatial_feature_ids = sorted(list(spatial_feature_ids))
print(f"Total spatial features: {len(spatial_feature_ids)}")

# =========================
# 加载 Gradient Attribution (如果提供)
# =========================
feature_importance_scores = None

if args.gradient_results:
    print("\n" + "="*60)
    print("Loading gradient attribution results...")
    print("="*60)
    
    grad_results_path = Path(args.gradient_results)
    if grad_results_path.exists():
        with open(grad_results_path, 'r') as f:
            grad_data = json.load(f)
        
        # 提取特征重要性分数
        avg_attributions = np.array(grad_data['attribution_results']['average_attributions'])
        feature_importance_scores = {i: float(np.abs(avg_attributions[i])) 
                                     for i in spatial_feature_ids}
        print(f"Loaded importance scores for {len(feature_importance_scores)} features")
    else:
        print(f"Warning: {grad_results_path} not found, will use random ordering")

# 按重要性排序 spatial features
if feature_importance_scores:
    # 按 importance score 降序排序
    sorted_spatial_features = sorted(spatial_feature_ids, 
                                     key=lambda x: feature_importance_scores.get(x, 0), 
                                     reverse=True)
    print("Spatial features sorted by importance (gradient attribution)")
else:
    # 随机排序
    sorted_spatial_features = spatial_feature_ids.copy()
    random.shuffle(sorted_spatial_features)
    print("Spatial features randomly ordered (no gradient data)")

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

print(f"Model loaded: {model.cfg.n_layers} layers")

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
        # 用于 mean ablation 的缓存
        self.feature_means = None
    
    def encode(self, x):
        x_centered = x - self.b_dec
        pre_relu = x_centered @ self.W_enc + self.b_enc
        return torch.nn.functional.relu(pre_relu)
    
    def decode(self, h):
        return h @ self.W_dec + self.b_dec
    
    def compute_feature_means(self, activations_list):
        """计算特征的平均激活值"""
        all_features = []
        for act in activations_list:
            # 确保 act 在正确的设备上
            act = act.to(self.device)
            h = self.encode(act)
            all_features.append(h.detach().cpu())
        
        all_features = torch.cat(all_features, dim=0)
        self.feature_means = torch.mean(all_features, dim=0).to(self.device)
        return self.feature_means

sae = SimpleSAE(W_enc, b_enc, W_dec, b_dec, DEVICE)

# =========================
# Dose-Response Ablator
# =========================
class DoseResponseAblator:
    """
    Ablate top-k spatial features with different ablation types
    """
    def __init__(self, sae: SimpleSAE, feature_ids: List[int], 
                 ablation_type: str = "mean", n_features: int = None):
        self.sae = sae
        self.feature_ids = feature_ids  # Features to ablate (already sorted)
        self.ablation_type = ablation_type
        self.n_features = n_features
        
    def __call__(self, act, hook):
        original_shape = act.shape
        
        if len(original_shape) == 3:
            batch_size, seq_len, d = original_shape
            act_flat = act.reshape(-1, d)
        else:
            act_flat = act
        
        with torch.no_grad():
            # Encode
            h = self.sae.encode(act_flat)
            
            # Ablate
            if self.ablation_type == "zero":
                h[:, self.feature_ids] = 0.0
            elif self.ablation_type == "mean":
                if self.sae.feature_means is not None:
                    h[:, self.feature_ids] = self.sae.feature_means[self.feature_ids]
                else:
                    h[:, self.feature_ids] = 0.0  # Fallback
            elif self.ablation_type == "shuffle":
                # Shuffle within batch
                for feat_id in self.feature_ids:
                    perm = torch.randperm(h.shape[0], device=h.device)
                    h[:, feat_id] = h[perm, feat_id]
            
            # Decode
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

# 编码选项
OPTION_TOKENS = {}
for option in ['A', 'B', 'C', 'D']:
    tokens = tokenizer.encode(option, add_special_tokens=False)
    if len(tokens) > 0:
        OPTION_TOKENS[option] = tokens[0]

print(f"Option tokens: {OPTION_TOKENS}")

# =========================
# 预计算 feature means (for mean ablation)
# =========================
if args.ablation_type == "mean":
    print("\n" + "="*60)
    print("Computing feature means...")
    print("="*60)
    
    sample_acts = []
    for i, sample in enumerate(tqdm(eval_data[:min(100, len(eval_data))], desc="Collecting activations")):
        prompt = sample.get("prompt", sample.get("question", ""))
        tokens = model.to_tokens(prompt, truncate=True)
        
        # Get activation
        cache = {}
        def capture_act(act, hook):
            cache['act'] = act[:, -1, :].detach().cpu()
            return act
        
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(HOOK_POINT, capture_act)]):
                _ = model(tokens)
        
        if 'act' in cache:
            sample_acts.append(cache['act'])
    
    # Compute means
    sae.compute_feature_means(sample_acts)
    print(f"✓ Computed feature means from {len(sample_acts)} samples")

# =========================
# 评估函数
# =========================
def evaluate_with_metrics(
    model: HookedTransformer,
    data: List[Dict],
    ablator: Optional[DoseResponseAblator] = None,
    desc: str = "Evaluating"
) -> Dict:
    """
    评估模型，计算 accuracy, margin, NLL
    """
    results = []
    correct = 0
    total = 0
    
    for sample in tqdm(data, desc=desc):
        prompt = sample.get("prompt", sample.get("question", ""))
        correct_option = sample.get("correct_option", sample.get("answer", "A"))
        
        # Tokenize
        tokens = model.to_tokens(prompt, truncate=True)
        
        # Forward pass
        if ablator is not None:
            with model.hooks(fwd_hooks=[(HOOK_POINT, ablator)]):
                with torch.no_grad():
                    logits = model(tokens)
        else:
            with torch.no_grad():
                logits = model(tokens)
        
        last_logits = logits[0, -1, :]
        
        # Get option logits
        option_logits = {opt: last_logits[OPTION_TOKENS[opt]].item() 
                        for opt in OPTION_TOKENS}
        
        # Prediction
        pred_option = max(option_logits, key=option_logits.get)
        is_correct = (pred_option == correct_option)
        
        # Metrics
        correct_logit = option_logits[correct_option]
        other_logits = [v for k, v in option_logits.items() if k != correct_option]
        max_other_logit = max(other_logits)
        
        margin = correct_logit - max_other_logit
        
        # NLL (negative log-likelihood)
        all_logits = torch.tensor([option_logits[opt] for opt in sorted(OPTION_TOKENS.keys())])
        probs = torch.softmax(all_logits, dim=0)
        correct_idx = sorted(OPTION_TOKENS.keys()).index(correct_option)
        nll = -torch.log(probs[correct_idx] + 1e-10).item()
        
        results.append({
            'prompt': prompt,
            'correct_option': correct_option,
            'predicted_option': pred_option,
            'correct': is_correct,
            'margin': margin,
            'nll': nll,
            'option_logits': option_logits,
        })
        
        if is_correct:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    avg_margin = np.mean([r['margin'] for r in results])
    avg_nll = np.mean([r['nll'] for r in results])
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_margin': avg_margin,
        'avg_nll': avg_nll,
        'results': results,
    }

# =========================
# 主实验：Dose-Response
# =========================
print("\n" + "="*60)
print(f"Running Dose-Response Experiment ({args.ablation_type} ablation)")
print("="*60)

all_results = {}

# Baseline (no ablation)
print("\n[Baseline] No ablation...")
baseline_results = evaluate_with_metrics(model, eval_data, ablator=None, desc="Baseline")
print(f"✓ Accuracy: {baseline_results['accuracy']:.4f}, "
      f"Margin: {baseline_results['avg_margin']:.4f}, "
      f"NLL: {baseline_results['avg_nll']:.4f}")

all_results['baseline'] = baseline_results

# Test different k values
for k in K_VALUES:
    if k > len(sorted_spatial_features):
        print(f"\nSkipping k={k} (exceeds number of spatial features)")
        continue
    
    print(f"\n[k={k}] Ablating top {k} spatial features...")
    
    # Select top-k features
    features_to_ablate = sorted_spatial_features[:k]
    
    # Create ablator
    ablator = DoseResponseAblator(
        sae=sae,
        feature_ids=features_to_ablate,
        ablation_type=args.ablation_type,
        n_features=N_FEATURES
    )
    
    # Evaluate
    k_results = evaluate_with_metrics(model, eval_data, ablator=ablator, desc=f"k={k}")
    print(f"✓ Accuracy: {k_results['accuracy']:.4f}, "
          f"Margin: {k_results['avg_margin']:.4f}, "
          f"NLL: {k_results['avg_nll']:.4f}")
    
    all_results[f'k_{k}'] = k_results

# =========================
# 结果分析
# =========================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"\n{'k':<10} {'Accuracy':<12} {'Δ Acc':<12} {'Margin':<12} {'NLL':<12}")
print("-" * 60)

baseline_acc = all_results['baseline']['accuracy']
baseline_margin = all_results['baseline']['avg_margin']
baseline_nll = all_results['baseline']['avg_nll']

print(f"{'0 (base)':<10} {baseline_acc:<12.4f} {'-':<12} "
      f"{baseline_margin:<12.4f} {baseline_nll:<12.4f}")

for k in K_VALUES:
    if f'k_{k}' in all_results:
        res = all_results[f'k_{k}']
        acc = res['accuracy']
        margin = res['avg_margin']
        nll = res['avg_nll']
        delta_acc = acc - baseline_acc
        print(f"{k:<10} {acc:<12.4f} {delta_acc:<+12.4f} "
              f"{margin:<12.4f} {nll:<12.4f}")

# =========================
# 可视化
# =========================
print("\n" + "="*60)
print("Generating plots...")
print("="*60)

# 准备数据
k_list = [0] + [k for k in K_VALUES if f'k_{k}' in all_results]
acc_list = [baseline_acc] + [all_results[f'k_{k}']['accuracy'] for k in K_VALUES if f'k_{k}' in all_results]
margin_list = [baseline_margin] + [all_results[f'k_{k}']['avg_margin'] for k in K_VALUES if f'k_{k}' in all_results]
nll_list = [baseline_nll] + [all_results[f'k_{k}']['avg_nll'] for k in K_VALUES if f'k_{k}' in all_results]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Accuracy vs k
axes[0].plot(k_list, acc_list, marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0].set_xlabel('Number of Ablated Features (k)', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Accuracy vs. Number of Ablated Features', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random chance (4 options)')
axes[0].legend()

# Plot 2: Margin vs k
axes[1].plot(k_list, margin_list, marker='s', linewidth=2, markersize=8, color='coral')
axes[1].set_xlabel('Number of Ablated Features (k)', fontsize=12)
axes[1].set_ylabel('Average Margin (correct - max_other)', fontsize=12)
axes[1].set_title('Margin vs. Number of Ablated Features', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Decision boundary')
axes[1].legend()

# Plot 3: NLL vs k
axes[2].plot(k_list, nll_list, marker='^', linewidth=2, markersize=8, color='mediumseagreen')
axes[2].set_xlabel('Number of Ablated Features (k)', fontsize=12)
axes[2].set_ylabel('Negative Log-Likelihood (NLL)', fontsize=12)
axes[2].set_title('NLL vs. Number of Ablated Features', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_file = output_dir / f"dose_response_{args.ablation_type}_{timestamp}.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {plot_file}")
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
        'ablation_type': args.ablation_type,
        'k_values': K_VALUES,
        'n_spatial_features': len(spatial_feature_ids),
        'eval_samples': len(eval_data),
        'sorted_by_importance': feature_importance_scores is not None,
    },
    'sorted_spatial_features': sorted_spatial_features,
    'all_results': {
        k: {
            'accuracy': float(v['accuracy']),
            'avg_margin': float(v['avg_margin']),
            'avg_nll': float(v['avg_nll']),
            'correct': v['correct'],
            'total': v['total'],
        }
        for k, v in all_results.items()
    },
}

results_file = output_dir / f"dose_response_{args.ablation_type}_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved to: {results_file}")

# Summary
summary_file = output_dir / f"dose_response_summary_{timestamp}.txt"
with open(summary_file, 'w') as f:
    f.write("="*60 + "\n")
    f.write("Dose-Response Ablation Experiment\n")
    f.write("="*60 + "\n\n")
    f.write(f"Ablation type: {args.ablation_type}\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Layer: {LAYER_IDX}\n")
    f.write(f"Spatial features: {len(spatial_feature_ids)}\n\n")
    f.write("="*60 + "\n")
    f.write("Results:\n")
    f.write("="*60 + "\n\n")
    f.write(f"{'k':<10} {'Accuracy':<12} {'Δ Acc':<12} {'Margin':<12} {'NLL':<12}\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'0 (base)':<10} {baseline_acc:<12.4f} {'-':<12} "
            f"{baseline_margin:<12.4f} {baseline_nll:<12.4f}\n")
    for k in K_VALUES:
        if f'k_{k}' in all_results:
            res = all_results[f'k_{k}']
            acc = res['accuracy']
            margin = res['avg_margin']
            nll = res['avg_nll']
            delta_acc = acc - baseline_acc
            f.write(f"{k:<10} {acc:<12.4f} {delta_acc:<+12.4f} "
                    f"{margin:<12.4f} {nll:<12.4f}\n")
    
    # Interpretation
    f.write("\n" + "="*60 + "\n")
    f.write("Interpretation:\n")
    f.write("="*60 + "\n")
    
    # Check monotonicity
    is_monotonic = all(acc_list[i] >= acc_list[i+1] for i in range(len(acc_list)-1))
    if is_monotonic:
        f.write("✓ Performance monotonically decreases with more ablations\n")
    else:
        f.write("~ Performance does not monotonically decrease\n")
    
    # Check if there's a significant drop
    final_drop = baseline_acc - acc_list[-1]
    if final_drop > 0.15:
        f.write(f"✓ Strong dose-response effect: {final_drop:.2%} accuracy drop\n")
    elif final_drop > 0.05:
        f.write(f"~ Moderate dose-response effect: {final_drop:.2%} accuracy drop\n")
    else:
        f.write(f"✗ Weak dose-response effect: {final_drop:.2%} accuracy drop\n")

print(f"✓ Summary saved to: {summary_file}")

print("\n" + "="*60)
print("DOSE-RESPONSE EXPERIMENT COMPLETE")
print("="*60)

