"""
Layer Localization Experiment: 分层/分模块定位
================================================

目的：在哪一层的空间特征最关键？

做法：对不同层（或不同 SAE 字典）分别做同样 ablation

产出：一张"layer → ΔAcc/Δmargin"的表或条形图

价值：直接给出"关键瓶颈层"
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
parser = argparse.ArgumentParser(description='Layer Localization Experiment')
parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--sae_results_dir", "-s", type=str, required=True,
                    help="Directory containing multiple SAE results for different layers")
parser.add_argument("--eval_data_file", "-e", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default=None)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--ablation_type", type=str, default="zero", 
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

sae_results_dir = Path(args.sae_results_dir)
if args.output_dir is None:
    output_dir = sae_results_dir / "layer_localization_results"
else:
    output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime('%Y%m%d_%H%M%S')

# =========================
# 发现所有层的 SAE checkpoints
# =========================
print("\n" + "="*60)
print("Discovering SAE checkpoints for different layers...")
print("="*60)

# 查找所有 SAE checkpoint 目录
# 假设目录名格式为: L{layer}_F{features}_...
sae_checkpoints = {}

for subdir in sae_results_dir.iterdir():
    if subdir.is_dir():
        # 尝试从目录名中提取层号
        dir_name = subdir.name
        if dir_name.startswith('L') and '_F' in dir_name:
            try:
                layer_str = dir_name.split('_')[0][1:]  # 去掉 'L'
                layer_idx = int(layer_str)
                
                # 检查是否有 checkpoint 文件
                checkpoint_file = subdir / 'sae_checkpoint.pt'
                if checkpoint_file.exists():
                    sae_checkpoints[layer_idx] = subdir
                    print(f"Found SAE for Layer {layer_idx}: {subdir.name}")
            except ValueError:
                continue

if not sae_checkpoints:
    print("Error: No SAE checkpoints found!")
    print(f"Please ensure SAE results are in {sae_results_dir}")
    print("Expected directory format: L{layer}_F{features}_...")
    sys.exit(1)

# 按层号排序
layers = sorted(sae_checkpoints.keys())
print(f"\nTotal layers found: {len(layers)}")
print(f"Layers: {layers}")

# =========================
# 加载模型（一次性加载）
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
# SimpleSAE + Ablator
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

class LayerAblator:
    def __init__(self, sae: SimpleSAE, feature_ids: List[int], ablation_type: str = "zero"):
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
                    h[:, self.feature_ids] = 0.0
            elif self.ablation_type == "shuffle":
                for feat_id in self.feature_ids:
                    perm = torch.randperm(h.shape[0], device=h.device)
                    h[:, feat_id] = h[perm, feat_id]
            
            act_hat = self.sae.decode(h)
        
        if len(original_shape) == 3:
            act_hat = act_hat.reshape(batch_size, seq_len, d)
        
        return act_hat

# =========================
# 评估函数
# =========================
def evaluate_with_metrics(
    model: HookedTransformer,
    data: List[Dict],
    ablator: Optional[LayerAblator] = None,
    hook_point: str = None,
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
        
        if ablator is not None and hook_point is not None:
            with model.hooks(fwd_hooks=[(hook_point, ablator)]):
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
# 主实验：逐层测试
# =========================
print("\n" + "="*60)
print("Running Layer Localization Experiment")
print("="*60)

# Baseline (no ablation)
print("\n[Baseline] No ablation...")
baseline_results = evaluate_with_metrics(model, eval_data, ablator=None, desc="Baseline")
print(f"✓ Acc: {baseline_results['accuracy']:.4f}, "
      f"Margin: {baseline_results['avg_margin']:.4f}, "
      f"NLL: {baseline_results['avg_nll']:.4f}")

# 逐层测试
layer_results = {}

for layer_idx in layers:
    print(f"\n" + "="*60)
    print(f"Testing Layer {layer_idx}")
    print("="*60)
    
    checkpoint_dir = sae_checkpoints[layer_idx]
    
    # 加载 SAE
    checkpoint_file = checkpoint_dir / 'sae_checkpoint.pt'
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    sae_config = checkpoint['config']
    sae_state = checkpoint['model_state_dict']
    
    N_FEATURES = sae_config['n_features']
    D_IN = sae_config['d_in']
    
    W_enc = sae_state['W_enc'].to(DEVICE)
    b_enc = sae_state['b_enc'].to(DEVICE)
    W_dec = sae_state['W_dec'].to(DEVICE)
    b_dec = sae_state.get('b_dec', torch.zeros(D_IN)).to(DEVICE)
    
    sae = SimpleSAE(W_enc, b_enc, W_dec, b_dec, DEVICE)
    
    # 加载 spatial features
    analysis_dir = checkpoint_dir / "analysis"
    dimension_features_file = analysis_dir / "dimension_features.json"
    
    if not dimension_features_file.exists():
        print(f"Warning: No spatial features found for layer {layer_idx}, skipping...")
        continue
    
    with open(dimension_features_file, 'r') as f:
        dimension_features = json.load(f)
    
    spatial_feature_ids = set()
    for key in dimension_features:
        for feat in dimension_features[key]:
            spatial_feature_ids.add(feat['feature_idx'])
    
    spatial_feature_ids = sorted(list(spatial_feature_ids))
    print(f"Spatial features: {len(spatial_feature_ids)}")
    
    if len(spatial_feature_ids) == 0:
        print(f"Warning: No spatial features for layer {layer_idx}, skipping...")
        continue
    
    # 预计算 feature means (if needed)
    if args.ablation_type == "mean":
        print("Computing feature means...")
        sample_acts = []
        hook_point = f"blocks.{layer_idx}.hook_mlp_out"
        
        for sample in tqdm(eval_data[:min(50, len(eval_data))], desc="Collecting", leave=False):
            prompt = sample.get("prompt", sample.get("question", ""))
            tokens = model.to_tokens(prompt, truncate=True)
            
            cache = {}
            def capture_act(act, hook):
                cache['act'] = act[:, -1, :].detach().cpu()
                return act
            
            with torch.no_grad():
                with model.hooks(fwd_hooks=[(hook_point, capture_act)]):
                    _ = model(tokens)
            
            if 'act' in cache:
                sample_acts.append(cache['act'])
        
        sae.compute_feature_means(sample_acts)
    
    # Hook point
    hook_point = f"blocks.{layer_idx}.hook_mlp_out"
    
    # Spatial ablation
    print(f"[Layer {layer_idx}] Spatial ablation...")
    spatial_ablator = LayerAblator(sae, spatial_feature_ids, args.ablation_type)
    spatial_results = evaluate_with_metrics(
        model, eval_data, ablator=spatial_ablator, hook_point=hook_point,
        desc=f"Layer {layer_idx} spatial"
    )
    print(f"✓ Acc: {spatial_results['accuracy']:.4f}, "
          f"Margin: {spatial_results['avg_margin']:.4f}, "
          f"NLL: {spatial_results['avg_nll']:.4f}")
    
    # Random ablation
    print(f"[Layer {layer_idx}] Random ablation...")
    available_ids = [i for i in range(N_FEATURES) if i not in spatial_feature_ids]
    random_ids = random.sample(available_ids, min(len(spatial_feature_ids), len(available_ids)))
    
    random_ablator = LayerAblator(sae, random_ids, args.ablation_type)
    random_results = evaluate_with_metrics(
        model, eval_data, ablator=random_ablator, hook_point=hook_point,
        desc=f"Layer {layer_idx} random"
    )
    print(f"✓ Acc: {random_results['accuracy']:.4f}, "
          f"Margin: {random_results['avg_margin']:.4f}, "
          f"NLL: {random_results['avg_nll']:.4f}")
    
    # Store results
    layer_results[layer_idx] = {
        'n_spatial_features': len(spatial_feature_ids),
        'spatial': spatial_results,
        'random': random_results,
    }

# =========================
# 结果分析
# =========================
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

baseline_acc = baseline_results['accuracy']
baseline_margin = baseline_results['avg_margin']

print(f"\nBaseline: Acc={baseline_acc:.4f}, Margin={baseline_margin:.4f}")
print(f"\n{'Layer':<8} {'#Spatial':<10} {'Spatial Acc':<13} {'Random Acc':<13} "
      f"{'Δ Acc (S-R)':<15} {'Effect Size':<12}")
print("-" * 81)

for layer_idx in layers:
    if layer_idx not in layer_results:
        continue
    
    res = layer_results[layer_idx]
    spatial_acc = res['spatial']['accuracy']
    random_acc = res['random']['accuracy']
    delta_acc = spatial_acc - random_acc
    
    # Effect size: (baseline - spatial) - (baseline - random)
    spatial_drop = baseline_acc - spatial_acc
    random_drop = baseline_acc - random_acc
    effect_size = spatial_drop - random_drop
    
    print(f"{layer_idx:<8} {res['n_spatial_features']:<10} {spatial_acc:<13.4f} "
          f"{random_acc:<13.4f} {delta_acc:<+15.4f} {effect_size:<+12.4f}")

# 找出关键层
print("\n" + "="*60)
print("KEY BOTTLENECK LAYERS")
print("="*60)

effect_sizes = {}
for layer_idx in layers:
    if layer_idx not in layer_results:
        continue
    
    res = layer_results[layer_idx]
    spatial_drop = baseline_acc - res['spatial']['accuracy']
    random_drop = baseline_acc - res['random']['accuracy']
    effect_size = spatial_drop - random_drop
    effect_sizes[layer_idx] = effect_size

# 按 effect size 排序
sorted_layers = sorted(effect_sizes.items(), key=lambda x: x[1], reverse=True)

print("\nLayers ranked by effect size (spatial drop - random drop):")
for rank, (layer_idx, effect) in enumerate(sorted_layers, 1):
    print(f"{rank}. Layer {layer_idx}: {effect:+.4f}")

if sorted_layers:
    top_layer = sorted_layers[0][0]
    print(f"\n✓ Most critical layer: Layer {top_layer}")

# =========================
# 可视化
# =========================
print("\n" + "="*60)
print("Generating visualizations...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

layer_list = [l for l in layers if l in layer_results]
spatial_accs = [layer_results[l]['spatial']['accuracy'] for l in layer_list]
random_accs = [layer_results[l]['random']['accuracy'] for l in layer_list]
spatial_drops = [baseline_acc - layer_results[l]['spatial']['accuracy'] for l in layer_list]
random_drops = [baseline_acc - layer_results[l]['random']['accuracy'] for l in layer_list]
effect_sizes_list = [spatial_drops[i] - random_drops[i] for i in range(len(layer_list))]

# Plot 1: Accuracy by layer
axes[0, 0].plot(layer_list, [baseline_acc]*len(layer_list), 'k--', 
                linewidth=2, label='Baseline', alpha=0.7)
axes[0, 0].plot(layer_list, spatial_accs, 'o-', linewidth=2, 
                markersize=8, color='coral', label='Spatial ablation')
axes[0, 0].plot(layer_list, random_accs, 's-', linewidth=2, 
                markersize=8, color='mediumseagreen', label='Random ablation')
axes[0, 0].set_xlabel('Layer', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Accuracy Across Layers', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy drop by layer
x = np.arange(len(layer_list))
width = 0.35

axes[0, 1].bar(x - width/2, spatial_drops, width, label='Spatial', 
               color='coral', alpha=0.8)
axes[0, 1].bar(x + width/2, random_drops, width, label='Random', 
               color='mediumseagreen', alpha=0.8)
axes[0, 1].set_xlabel('Layer', fontsize=12)
axes[0, 1].set_ylabel('Accuracy Drop', fontsize=12)
axes[0, 1].set_title('Accuracy Drop by Layer', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels([f'L{l}' for l in layer_list])
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# Plot 3: Effect size by layer
axes[1, 0].bar(range(len(layer_list)), effect_sizes_list, color='purple', alpha=0.8)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Layer', fontsize=12)
axes[1, 0].set_ylabel('Effect Size (Spatial Drop - Random Drop)', fontsize=12)
axes[1, 0].set_title('Effect Size by Layer', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(range(len(layer_list)))
axes[1, 0].set_xticklabels([f'L{l}' for l in layer_list])
axes[1, 0].grid(axis='y', alpha=0.3)

# Plot 4: Number of spatial features by layer
n_spatial_list = [layer_results[l]['n_spatial_features'] for l in layer_list]
axes[1, 1].bar(range(len(layer_list)), n_spatial_list, color='steelblue', alpha=0.8)
axes[1, 1].set_xlabel('Layer', fontsize=12)
axes[1, 1].set_ylabel('Number of Spatial Features', fontsize=12)
axes[1, 1].set_title('Spatial Features Count by Layer', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(range(len(layer_list)))
axes[1, 1].set_xticklabels([f'L{l}' for l in layer_list])
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plot_file = output_dir / f"layer_localization_{timestamp}.png"
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
        'layers_tested': layer_list,
        'ablation_type': args.ablation_type,
        'eval_samples': len(eval_data),
    },
    'baseline': {
        'accuracy': float(baseline_acc),
        'avg_margin': float(baseline_margin),
        'avg_nll': float(baseline_results['avg_nll']),
    },
    'layer_results': {
        str(layer): {
            'n_spatial_features': res['n_spatial_features'],
            'spatial_accuracy': float(res['spatial']['accuracy']),
            'random_accuracy': float(res['random']['accuracy']),
            'spatial_drop': float(baseline_acc - res['spatial']['accuracy']),
            'random_drop': float(baseline_acc - res['random']['accuracy']),
            'effect_size': float((baseline_acc - res['spatial']['accuracy']) - 
                                (baseline_acc - res['random']['accuracy'])),
        }
        for layer, res in layer_results.items()
    },
    'ranking': [
        {'layer': layer, 'effect_size': float(effect)}
        for layer, effect in sorted_layers
    ],
}

results_file = output_dir / f"layer_localization_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Results saved: {results_file}")

print("\n" + "="*60)
print("LAYER LOCALIZATION EXPERIMENT COMPLETE")
print("="*60)

if sorted_layers:
    print(f"\n✓ Most critical layer: Layer {sorted_layers[0][0]} "
          f"(effect size: {sorted_layers[0][1]:+.4f})")

