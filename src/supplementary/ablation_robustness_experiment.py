"""
Ablation Robustness Experiment: zero vs mean vs shuffle
=========================================================

目的：排除"置零导致分布外（OOD）所以崩"的反驳

期待模式：
- mean / shuffle 也能复现"spatial 更伤" → 结论更稳
- 只有置零有效 → 可能是 OOD artifact
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
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 命令行参数
# =========================
parser = argparse.ArgumentParser(description='Ablation Robustness Experiment')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True)
parser.add_argument("--eval_data_file", "-e", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=None)
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
    output_dir = checkpoint_dir / "ablation_robustness_results"
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
# SimpleSAE
# =========================
class SimpleSAE:
    def __init__(self, W_enc, b_enc, W_dec, b_dec, device):
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec
        self.b_dec = b_dec
        self.device = device
        self.feature_means = None
    
    def encode(self, x):
        x_centered = x - self.b_dec
        pre_relu = x_centered @ self.W_enc + self.b_enc
        return torch.nn.functional.relu(pre_relu)
    
    def decode(self, h):
        return h @ self.W_dec + self.b_dec
    
    def compute_feature_means(self, activations_list):
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
# Multi-Mode Ablator
# =========================
class MultiModeAblator:
    """
    Ablator that supports: zero, mean, shuffle
    """
    def __init__(self, sae: SimpleSAE, feature_ids: List[int], 
                 ablation_type: str = "zero"):
        self.sae = sae
        self.feature_ids = feature_ids
        self.ablation_type = ablation_type
        
    def __call__(self, act, hook):
        original_shape = act.shape
        
        if len(original_shape) == 3:
            batch_size, seq_len, d = original_shape
            act_flat = act.reshape(-1, d)
        else:
            act_flat = act
        
        with torch.no_grad():
            h = self.sae.encode(act_flat)
            
            if self.ablation_type == "zero":
                h[:, self.feature_ids] = 0.0
            
            elif self.ablation_type == "mean":
                if self.sae.feature_means is not None:
                    h[:, self.feature_ids] = self.sae.feature_means[self.feature_ids]
                else:
                    h[:, self.feature_ids] = 0.0  # fallback
            
            elif self.ablation_type == "shuffle":
                for feat_id in self.feature_ids:
                    perm = torch.randperm(h.shape[0], device=h.device)
                    h[:, feat_id] = h[perm, feat_id]
            
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
# 预计算 feature means (for mean ablation)
# =========================
print("\n" + "="*60)
print("Computing feature means...")
print("="*60)

sample_acts = []
for sample in tqdm(eval_data[:min(100, len(eval_data))], desc="Collecting"):
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
        sample_acts.append(cache['act'])

sae.compute_feature_means(sample_acts)
print(f"✓ Computed feature means")

# =========================
# 评估函数
# =========================
def evaluate_with_metrics(
    model: HookedTransformer,
    data: List[Dict],
    ablator: Optional[MultiModeAblator] = None,
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
# 主实验：测试不同 ablation 方法
# =========================
print("\n" + "="*60)
print("Running Ablation Robustness Experiment")
print("="*60)

ABLATION_TYPES = ["zero", "mean", "shuffle"]

all_results = {}

# Baseline
print("\n[Baseline] No ablation...")
baseline_results = evaluate_with_metrics(model, eval_data, ablator=None, desc="Baseline")
print(f"✓ Acc: {baseline_results['accuracy']:.4f}, "
      f"Margin: {baseline_results['avg_margin']:.4f}, "
      f"NLL: {baseline_results['avg_nll']:.4f}")
all_results['baseline'] = baseline_results

# For each ablation type, test spatial and random
for ablation_type in ABLATION_TYPES:
    print(f"\n" + "="*60)
    print(f"Testing {ablation_type.upper()} ablation")
    print("="*60)
    
    # Spatial ablation
    print(f"\n[{ablation_type}] Spatial ablation...")
    spatial_ablator = MultiModeAblator(sae, spatial_feature_ids, ablation_type)
    spatial_results = evaluate_with_metrics(
        model, eval_data, ablator=spatial_ablator, desc=f"{ablation_type} spatial"
    )
    print(f"✓ Acc: {spatial_results['accuracy']:.4f}, "
          f"Margin: {spatial_results['avg_margin']:.4f}, "
          f"NLL: {spatial_results['avg_nll']:.4f}")
    all_results[f'spatial_{ablation_type}'] = spatial_results
    
    # Random ablation
    print(f"\n[{ablation_type}] Random ablation...")
    available_ids = [i for i in range(N_FEATURES) if i not in spatial_feature_ids]
    random_ids = random.sample(available_ids, min(len(spatial_feature_ids), len(available_ids)))
    
    random_ablator = MultiModeAblator(sae, random_ids, ablation_type)
    random_results = evaluate_with_metrics(
        model, eval_data, ablator=random_ablator, desc=f"{ablation_type} random"
    )
    print(f"✓ Acc: {random_results['accuracy']:.4f}, "
          f"Margin: {random_results['avg_margin']:.4f}, "
          f"NLL: {random_results['avg_nll']:.4f}")
    all_results[f'random_{ablation_type}'] = random_results

# =========================
# 结果分析
# =========================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

baseline_acc = baseline_results['accuracy']

print(f"\n{'Condition':<25} {'Accuracy':<12} {'Δ Acc':<12} {'Margin':<12} {'NLL':<12}")
print("-" * 73)

print(f"{'Baseline':<25} {baseline_acc:<12.4f} {'-':<12} "
      f"{baseline_results['avg_margin']:<12.4f} {baseline_results['avg_nll']:<12.4f}")

for ablation_type in ABLATION_TYPES:
    spatial_key = f'spatial_{ablation_type}'
    random_key = f'random_{ablation_type}'
    
    spatial_acc = all_results[spatial_key]['accuracy']
    random_acc = all_results[random_key]['accuracy']
    
    spatial_delta = spatial_acc - baseline_acc
    random_delta = random_acc - baseline_acc
    
    print(f"\n{ablation_type.upper()}:")
    print(f"  {'Spatial':<23} {spatial_acc:<12.4f} {spatial_delta:<+12.4f} "
          f"{all_results[spatial_key]['avg_margin']:<12.4f} "
          f"{all_results[spatial_key]['avg_nll']:<12.4f}")
    print(f"  {'Random':<23} {random_acc:<12.4f} {random_delta:<+12.4f} "
          f"{all_results[random_key]['avg_margin']:<12.4f} "
          f"{all_results[random_key]['avg_nll']:<12.4f}")

# =========================
# 稳健性分析
# =========================
print("\n" + "="*60)
print("ROBUSTNESS ANALYSIS")
print("="*60)

print(f"\n{'Ablation Type':<15} {'Spatial Drop':<15} {'Random Drop':<15} {'Ratio':<10}")
print("-" * 55)

robustness_data = []
for ablation_type in ABLATION_TYPES:
    spatial_acc = all_results[f'spatial_{ablation_type}']['accuracy']
    random_acc = all_results[f'random_{ablation_type}']['accuracy']
    
    spatial_drop = baseline_acc - spatial_acc
    random_drop = baseline_acc - random_acc
    ratio = spatial_drop / random_drop if random_drop > 0 else float('inf')
    
    print(f"{ablation_type:<15} {spatial_drop:<15.4f} {random_drop:<15.4f} {ratio:<10.2f}")
    
    robustness_data.append({
        'type': ablation_type,
        'spatial_drop': spatial_drop,
        'random_drop': random_drop,
        'ratio': ratio,
    })

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

consistent_types = []
for data in robustness_data:
    if data['ratio'] > 1.5:  # Spatial hurts more than random
        consistent_types.append(data['type'])

if len(consistent_types) == len(ABLATION_TYPES):
    print("✓ STRONG ROBUSTNESS: All ablation types show spatial features are more important")
    print("  → Result is NOT due to OOD artifact from zero ablation")
elif len(consistent_types) >= 2:
    print("✓ MODERATE ROBUSTNESS: Most ablation types confirm importance")
    print(f"  → Consistent types: {consistent_types}")
else:
    print("✗ WEAK ROBUSTNESS: Only some ablation types show effect")
    print("  → Result may be specific to ablation method")

# =========================
# 可视化
# =========================
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Accuracy comparison
x_labels = ['Baseline'] + [f'Spatial\n{t}' for t in ABLATION_TYPES] + [f'Random\n{t}' for t in ABLATION_TYPES]
y_values = [baseline_acc]
colors = ['steelblue']

for t in ABLATION_TYPES:
    y_values.append(all_results[f'spatial_{t}']['accuracy'])
    colors.append('coral')

for t in ABLATION_TYPES:
    y_values.append(all_results[f'random_{t}']['accuracy'])
    colors.append('mediumseagreen')

axes[0].bar(range(len(x_labels)), y_values, color=colors, alpha=0.8)
axes[0].set_xticks(range(len(x_labels)))
axes[0].set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Accuracy Across Ablation Types', fontsize=14, fontweight='bold')
axes[0].axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Random')
axes[0].grid(axis='y', alpha=0.3)
axes[0].legend()

# Plot 2: Accuracy drop comparison
spatial_drops = [baseline_acc - all_results[f'spatial_{t}']['accuracy'] for t in ABLATION_TYPES]
random_drops = [baseline_acc - all_results[f'random_{t}']['accuracy'] for t in ABLATION_TYPES]

x = np.arange(len(ABLATION_TYPES))
width = 0.35

axes[1].bar(x - width/2, spatial_drops, width, label='Spatial', color='coral', alpha=0.8)
axes[1].bar(x + width/2, random_drops, width, label='Random', color='mediumseagreen', alpha=0.8)
axes[1].set_xlabel('Ablation Type', fontsize=12)
axes[1].set_ylabel('Accuracy Drop', fontsize=12)
axes[1].set_title('Accuracy Drop Comparison', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels([t.capitalize() for t in ABLATION_TYPES])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Plot 3: Ratio (Spatial / Random)
ratios = [d['ratio'] for d in robustness_data]
axes[2].bar(range(len(ABLATION_TYPES)), ratios, color='purple', alpha=0.8)
axes[2].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Equal effect')
axes[2].set_xlabel('Ablation Type', fontsize=12)
axes[2].set_ylabel('Ratio (Spatial Drop / Random Drop)', fontsize=12)
axes[2].set_title('Effect Ratio Across Ablation Types', fontsize=14, fontweight='bold')
axes[2].set_xticks(range(len(ABLATION_TYPES)))
axes[2].set_xticklabels([t.capitalize() for t in ABLATION_TYPES])
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_file = output_dir / f"ablation_robustness_{timestamp}.png"
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
        'ablation_types': ABLATION_TYPES,
    },
    'all_results': {
        k: {
            'accuracy': float(v['accuracy']),
            'avg_margin': float(v['avg_margin']),
            'avg_nll': float(v['avg_nll']),
        }
        for k, v in all_results.items()
    },
    'robustness_analysis': robustness_data,
    'consistent_types': consistent_types,
}

results_file = output_dir / f"ablation_robustness_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved: {results_file}")

print("\n" + "="*60)
print("ABLATION ROBUSTNESS EXPERIMENT COMPLETE")
print("="*60)

