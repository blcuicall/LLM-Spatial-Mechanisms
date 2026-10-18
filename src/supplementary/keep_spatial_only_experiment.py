"""
Keep-Spatial-Only Experiment: 充分性测试
=========================================

目的：回答"空间特征是否足以解决任务 / 是否 shortcut"

做法：非空间特征全部 mean-ablate（或置换），只保留空间特征

解读：
- 若接近随机（≈25%）：空间特征"必要但不充分"（像门控/路由/辅助通道）
- 若明显高于随机：空间特征承载主要信息
- 若接近基线：空间特征几乎决定性（强 shortcut 风险）
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
parser = argparse.ArgumentParser(description='Keep-Spatial-Only Experiment')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--sae_checkpoint", "-c", type=str, required=True)
parser.add_argument("--eval_data_file", "-e", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--ablation_type", type=str, default="mean", 
                    choices=["zero", "mean", "shuffle"])
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
    output_dir = checkpoint_dir / "keep_spatial_only_results"
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

if not dimension_features_file.exists():
    print(f"Error: {dimension_features_file} not found!")
    sys.exit(1)

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
# Keep-Spatial-Only Ablator
# =========================
class KeepSpatialOnlyAblator:
    """
    Ablate non-spatial features, keep only spatial features
    """
    def __init__(self, sae: SimpleSAE, spatial_ids: List[int], 
                 non_spatial_ids: List[int], ablation_type: str = "mean"):
        self.sae = sae
        self.spatial_ids = spatial_ids
        self.non_spatial_ids = non_spatial_ids
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
            
            # Ablate non-spatial features
            if self.ablation_type == "zero":
                h[:, self.non_spatial_ids] = 0.0
            elif self.ablation_type == "mean":
                if self.sae.feature_means is not None:
                    h[:, self.non_spatial_ids] = self.sae.feature_means[self.non_spatial_ids]
                else:
                    h[:, self.non_spatial_ids] = 0.0
            elif self.ablation_type == "shuffle":
                for feat_id in self.non_spatial_ids:
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
# 预计算 feature means
# =========================
if args.ablation_type == "mean":
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
    ablator: Optional[KeepSpatialOnlyAblator] = None,
    desc: str = "Evaluating"
) -> Dict:
    """评估并计算详细指标"""
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
        
        # Metrics
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
        'results': results,
    }

# =========================
# 主实验
# =========================
print("\n" + "="*60)
print("Running Keep-Spatial-Only Experiment")
print("="*60)

# Condition 1: Baseline (no ablation)
print("\n[1/4] Baseline: No ablation...")
baseline_results = evaluate_with_metrics(model, eval_data, ablator=None, desc="Baseline")
print(f"✓ Acc: {baseline_results['accuracy']:.4f}, "
      f"Margin: {baseline_results['avg_margin']:.4f}, "
      f"NLL: {baseline_results['avg_nll']:.4f}")

# Condition 2: Keep spatial only (ablate non-spatial)
print("\n[2/4] Keep spatial only (ablate all non-spatial)...")
keep_spatial_ablator = KeepSpatialOnlyAblator(
    sae=sae,
    spatial_ids=spatial_feature_ids,
    non_spatial_ids=non_spatial_feature_ids,
    ablation_type=args.ablation_type
)
keep_spatial_results = evaluate_with_metrics(
    model, eval_data, ablator=keep_spatial_ablator, desc="Keep spatial"
)
print(f"✓ Acc: {keep_spatial_results['accuracy']:.4f}, "
      f"Margin: {keep_spatial_results['avg_margin']:.4f}, "
      f"NLL: {keep_spatial_results['avg_nll']:.4f}")

# Condition 3: Ablate spatial (keep non-spatial) - for comparison
print("\n[3/4] Ablate spatial (keep non-spatial)...")
ablate_spatial_ablator = KeepSpatialOnlyAblator(
    sae=sae,
    spatial_ids=non_spatial_feature_ids,  # Keep non-spatial
    non_spatial_ids=spatial_feature_ids,  # Ablate spatial
    ablation_type=args.ablation_type
)
ablate_spatial_results = evaluate_with_metrics(
    model, eval_data, ablator=ablate_spatial_ablator, desc="Ablate spatial"
)
print(f"✓ Acc: {ablate_spatial_results['accuracy']:.4f}, "
      f"Margin: {ablate_spatial_results['avg_margin']:.4f}, "
      f"NLL: {ablate_spatial_results['avg_nll']:.4f}")

# Condition 4: Ablate all features (sanity check)
print("\n[4/4] Ablate all features (sanity check)...")
ablate_all_ablator = KeepSpatialOnlyAblator(
    sae=sae,
    spatial_ids=[],  # Keep nothing
    non_spatial_ids=list(range(N_FEATURES)),  # Ablate all
    ablation_type=args.ablation_type
)
ablate_all_results = evaluate_with_metrics(
    model, eval_data, ablator=ablate_all_ablator, desc="Ablate all"
)
print(f"✓ Acc: {ablate_all_results['accuracy']:.4f}, "
      f"Margin: {ablate_all_results['avg_margin']:.4f}, "
      f"NLL: {ablate_all_results['avg_nll']:.4f}")

# =========================
# 结果分析
# =========================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

random_chance = 0.25  # 4 options

conditions = [
    ("Baseline (no ablation)", baseline_results),
    ("Keep spatial only", keep_spatial_results),
    ("Ablate spatial", ablate_spatial_results),
    ("Ablate all", ablate_all_results),
]

print(f"\n{'Condition':<30} {'Accuracy':<12} {'Margin':<12} {'NLL':<12}")
print("-" * 66)
for name, res in conditions:
    print(f"{name:<30} {res['accuracy']:<12.4f} {res['avg_margin']:<12.4f} {res['avg_nll']:<12.4f}")

print(f"\n{'Random chance':<30} {random_chance:<12.4f}")

# Interpretation
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

baseline_acc = baseline_results['accuracy']
keep_spatial_acc = keep_spatial_results['accuracy']
ablate_spatial_acc = ablate_spatial_results['accuracy']

retention_rate = keep_spatial_acc / baseline_acc if baseline_acc > 0 else 0

print(f"\nRetention rate (keep spatial / baseline): {retention_rate:.2%}")
print(f"Above random chance: {keep_spatial_acc > random_chance * 1.5}")
print(f"Gap to baseline: {baseline_acc - keep_spatial_acc:.4f} ({(baseline_acc - keep_spatial_acc)/baseline_acc:.1%})")

if keep_spatial_acc < random_chance * 1.2:
    print("\n✓ Spatial features are NECESSARY but NOT SUFFICIENT")
    print("  → They act like gating/routing/auxiliary channels")
elif keep_spatial_acc > baseline_acc * 0.8:
    print("\n✓ Spatial features are nearly SUFFICIENT")
    print("  → Strong shortcut risk (they almost determine the answer)")
else:
    print("\n✓ Spatial features carry SIGNIFICANT information")
    print("  → They are important but not the only component")

# =========================
# 可视化
# =========================
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Accuracy comparison
condition_names = ["Baseline", "Keep\nSpatial", "Ablate\nSpatial", "Ablate\nAll"]
accuracies = [r[1]['accuracy'] for r in conditions]
colors = ['steelblue', 'mediumseagreen', 'coral', 'gray']

axes[0].bar(condition_names, accuracies, color=colors, alpha=0.8)
axes[0].axhline(y=random_chance, color='red', linestyle='--', linewidth=2, label='Random (25%)')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Margin comparison
margins = [r[1]['avg_margin'] for r in conditions]
axes[1].bar(condition_names, margins, color=colors, alpha=0.8)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Decision boundary')
axes[1].set_ylabel('Average Margin', fontsize=12)
axes[1].set_title('Margin Comparison', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# Plot 3: NLL comparison
nlls = [r[1]['avg_nll'] for r in conditions]
axes[2].bar(condition_names, nlls, color=colors, alpha=0.8)
axes[2].set_ylabel('Negative Log-Likelihood', fontsize=12)
axes[2].set_title('NLL Comparison', fontsize=14, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_file = output_dir / f"keep_spatial_only_{args.ablation_type}_{timestamp}.png"
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
        'ablation_type': args.ablation_type,
        'n_spatial_features': len(spatial_feature_ids),
        'n_non_spatial_features': len(non_spatial_feature_ids),
        'eval_samples': len(eval_data),
    },
    'results': {
        'baseline': {
            'accuracy': float(baseline_results['accuracy']),
            'avg_margin': float(baseline_results['avg_margin']),
            'avg_nll': float(baseline_results['avg_nll']),
        },
        'keep_spatial_only': {
            'accuracy': float(keep_spatial_results['accuracy']),
            'avg_margin': float(keep_spatial_results['avg_margin']),
            'avg_nll': float(keep_spatial_results['avg_nll']),
        },
        'ablate_spatial': {
            'accuracy': float(ablate_spatial_results['accuracy']),
            'avg_margin': float(ablate_spatial_results['avg_margin']),
            'avg_nll': float(ablate_spatial_results['avg_nll']),
        },
        'ablate_all': {
            'accuracy': float(ablate_all_results['accuracy']),
            'avg_margin': float(ablate_all_results['avg_margin']),
            'avg_nll': float(ablate_all_results['avg_nll']),
        },
    },
    'interpretation': {
        'retention_rate': float(retention_rate),
        'above_random': bool(keep_spatial_acc > random_chance * 1.5),
        'gap_to_baseline': float(baseline_acc - keep_spatial_acc),
    }
}

results_file = output_dir / f"keep_spatial_only_{args.ablation_type}_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved: {results_file}")

print("\n" + "="*60)
print("KEEP-SPATIAL-ONLY EXPERIMENT COMPLETE")
print("="*60)

