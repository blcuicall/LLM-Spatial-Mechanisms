"""
SAE特征梯度归因分析 - 空间过程执行任务
使用梯度×激活值来分析哪些SAE特征对空间坐标变换任务最重要
"""
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from config import PATHS

import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# SAELens imports
try:
    from sae_lens import SAE
    from sae_lens.config import LanguageModelSAERunnerConfig, LoggingConfig
    from sae_lens.saes import TrainingSAE, StandardTrainingSAE, StandardTrainingSAEConfig
    SAELENS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAELens not installed or import failed: {e}")
    print("Please run: pip install sae-lens")
    SAELENS_AVAILABLE = False
    sys.exit(1)

# =========================
# 参数解析
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, required=True)
parser.add_argument("--sae_path", "-s", type=str, required=True, help="Path to trained SAE")
parser.add_argument("--test_data_file", "-te", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default="./gradient_attribution")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_samples", "-n", type=int, default=200, help="Number of test samples")
parser.add_argument("--top_k", type=int, default=50, help="Top K features to analyze")
args = parser.parse_args()

# =========================
# 配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = args.device
DTYPE = torch.float32
SAE_PATH = args.sae_path
TEST_DATA_FILE = args.test_data_file
OUTPUT_DIR = args.output_dir
NUM_SAMPLES = args.num_samples
TOP_K = args.top_k

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*50)
print("SAE Gradient Attribution Analysis (Spatial Procedure)")
print("="*50)
print(f"Model: {MODEL_NAME}")
print(f"SAE Path: {SAE_PATH}")
print(f"Output Dir: {OUTPUT_DIR}")
print(f"Num Samples: {NUM_SAMPLES}")
print(f"Top K: {TOP_K}")
print("="*50)

# =========================
# 加载模型
# =========================
print("\nLoading model...")
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
# 加载SAE
# =========================
print("\nLoading SAE...")
checkpoint = torch.load(SAE_PATH, map_location=DEVICE)
sae_config = checkpoint['config']
LAYER = sae_config['layer']

print(f"SAE Config: {sae_config}")

# 重建SAE
sae_cfg = StandardTrainingSAEConfig(
    d_in=sae_config['d_model'],
    d_sae=sae_config['d_sae'],
    l1_coefficient=sae_config.get('l1_coefficient', 0.001),
    dtype="float32",
    device=str(DEVICE),
    apply_b_dec_to_input=True,
    normalize_activations="none",
)

sae = StandardTrainingSAE(sae_cfg)
sae.load_state_dict(checkpoint['sae_state_dict'])
sae = sae.to(DEVICE)
sae.eval()

print(f"SAE loaded: layer={LAYER}, d_model={sae_config['d_model']}, d_sae={sae_config['d_sae']}")

# =========================
# 加载数据
# =========================
print("\nLoading test data...")
with open(TEST_DATA_FILE, "r") as f:
    test_data = json.load(f)

# 随机采样
np.random.seed(42)
if len(test_data) > NUM_SAMPLES:
    indices = np.random.choice(len(test_data), NUM_SAMPLES, replace=False)
    test_data = [test_data[i] for i in indices]

print(f"Using {len(test_data)} test samples")

# =========================
# 梯度归因计算
# =========================
print("\n" + "="*50)
print("Computing Gradient Attribution")
print("="*50)

all_feature_acts = []
all_gradients = []
all_attributions = []
all_targets = []

for sample in tqdm(test_data, desc="Computing gradient attribution"):
    prompt = sample['prompt']
    target = np.array(sample['target'])  # 最终坐标位置
    
    try:
        # 清除之前的梯度
        model.zero_grad()
        if hasattr(sae, 'zero_grad'):
            sae.zero_grad()
        
        # 使用更稳健的梯度计算方法
        tokens = model.to_tokens(prompt, truncate=True)
        
        # 前向传播
        with torch.enable_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{LAYER}.hook_resid_post"
            )
            
            hidden = cache[f"blocks.{LAYER}.hook_resid_post"][0, -1].detach()
            hidden.requires_grad_(True)
            
            # SAE编码
            feature_acts = sae.encode(hidden.unsqueeze(0))[0]
            
            # 重要：保留梯度以便访问
            feature_acts.retain_grad()
            
            # 计算一个简单的标量损失
            # 使用特征激活的总和作为损失
            loss = feature_acts.sum()
            
            # 反向传播
            loss.backward()
            
            # 获取梯度
            gradients = feature_acts.grad if feature_acts.grad is not None else torch.zeros_like(feature_acts)
            
            # 计算归因
            attribution = gradients * feature_acts.detach()
        
        all_feature_acts.append(feature_acts.detach().cpu().numpy())
        all_gradients.append(gradients.detach().cpu().numpy())
        all_attributions.append(attribution.detach().cpu().numpy())
        all_targets.append(target)
        
    except Exception as e:
        print(f"\nError processing sample: {e}")
        continue

# 转换为numpy数组
all_feature_acts = np.array(all_feature_acts)  # [n_samples, d_sae]
all_gradients = np.array(all_gradients)  # [n_samples, d_sae]
all_attributions = np.array(all_attributions)  # [n_samples, d_sae]
all_targets = np.array(all_targets)  # [n_samples, 3]

print(f"\nShape of attribution matrix: {all_attributions.shape}")

# =========================
# 分析1: 特征归因统计
# =========================
print("\n" + "="*50)
print("Analysis 1: Feature Attribution Statistics")
print("="*50)

# 计算每个特征的平均归因（绝对值）
mean_attribution = np.abs(all_attributions).mean(axis=0)  # [d_sae]
std_attribution = np.abs(all_attributions).std(axis=0)  # [d_sae]

# 计算激活频率
activation_freq = (all_feature_acts > 1e-6).mean(axis=0)  # [d_sae]

# 找出Top K特征
top_k_indices = np.argsort(mean_attribution)[-TOP_K:][::-1]

print(f"\nTop {min(20, TOP_K)} features by gradient attribution:")
for rank, idx in enumerate(top_k_indices[:20], 1):
    print(f"  {rank:2d}. Feature {idx:4d}: "
          f"Attribution={mean_attribution[idx]:.6f}, "
          f"Std={std_attribution[idx]:.6f}, "
          f"Freq={activation_freq[idx]:.4f}")

# =========================
# 分析2: 每个维度的归因
# =========================
print("\n" + "="*50)
print("Analysis 2: Per-Dimension Attribution")
print("="*50)

# 为每个坐标维度计算相关性
attribution_per_dim = {}

for dim_idx, dim_name in enumerate(['x', 'y', 'z']):
    # 计算每个特征与该维度的相关性
    correlations = []
    for feat_idx in range(all_attributions.shape[1]):
        # 使用皮尔逊相关系数
        # 检查标准差是否为零
        if np.std(all_attributions[:, feat_idx]) < 1e-10 or np.std(all_targets[:, dim_idx]) < 1e-10:
            corr = 0
        else:
            corr = np.corrcoef(all_attributions[:, feat_idx], all_targets[:, dim_idx])[0, 1]
            if np.isnan(corr):
                corr = 0
        correlations.append(abs(corr))
    
    correlations = np.array(correlations)
    top_indices = np.argsort(correlations)[-10:][::-1]
    
    attribution_per_dim[dim_name] = {
        'correlations': correlations,
        'top_indices': top_indices.tolist(),
        'top_values': [float(correlations[i]) for i in top_indices]
    }
    
    print(f"\n{dim_name.upper()}-axis (Top 10 features):")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. Feature {idx:4d}: Correlation={correlations[idx]:.6f}")

# =========================
# 分析3: 特征重叠分析
# =========================
print("\n" + "="*50)
print("Analysis 3: Feature Overlap Analysis")
print("="*50)

# 分析top features在不同维度间的重叠
top_x = set(attribution_per_dim['x']['top_indices'][:10])
top_y = set(attribution_per_dim['y']['top_indices'][:10])
top_z = set(attribution_per_dim['z']['top_indices'][:10])

overlap_xy = top_x & top_y
overlap_xz = top_x & top_z
overlap_yz = top_y & top_z
overlap_xyz = top_x & top_y & top_z

print(f"X-Y overlap: {len(overlap_xy)} features: {sorted(overlap_xy)}")
print(f"X-Z overlap: {len(overlap_xz)} features: {sorted(overlap_xz)}")
print(f"Y-Z overlap: {len(overlap_yz)} features: {sorted(overlap_yz)}")
print(f"X-Y-Z overlap: {len(overlap_xyz)} features: {sorted(overlap_xyz)}")

# =========================
# 保存结果
# =========================
print("\n" + "="*50)
print("Saving Results")
print("="*50)

results = {
    'model_name': MODEL_NAME,
    'layer': LAYER,
    'd_sae': sae_config['d_sae'],
    'num_samples': len(all_attributions),
    'top_features': {
        'indices': top_k_indices.tolist(),
        'mean_attribution': [float(mean_attribution[i]) for i in top_k_indices],
        'std_attribution': [float(std_attribution[i]) for i in top_k_indices],
        'activation_freq': [float(activation_freq[i]) for i in top_k_indices],
    },
    'per_dimension': {
        'x': {
            'top_indices': attribution_per_dim['x']['top_indices'],
            'correlations': attribution_per_dim['x']['top_values'],
        },
        'y': {
            'top_indices': attribution_per_dim['y']['top_indices'],
            'correlations': attribution_per_dim['y']['top_values'],
        },
        'z': {
            'top_indices': attribution_per_dim['z']['top_indices'],
            'correlations': attribution_per_dim['z']['top_values'],
        },
    },
    'feature_overlap': {
        'xy': list(overlap_xy),
        'xz': list(overlap_xz),
        'yz': list(overlap_yz),
        'xyz': list(overlap_xyz),
    },
}

output_file = os.path.join(OUTPUT_DIR, f"gradient_attribution_layer{LAYER}.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")

# =========================
# 可视化
# =========================
print("\nGenerating visualizations...")

# 1. Top features 柱状图
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
colors = plt.cm.viridis(np.linspace(0, 1, min(20, TOP_K)))
bars = plt.barh(range(min(20, TOP_K)), 
                [mean_attribution[i] for i in top_k_indices[:20][::-1]],
                color=colors)
plt.yticks(range(min(20, TOP_K)), [f'F{i}' for i in top_k_indices[:20][::-1]])
plt.xlabel('Mean Absolute Attribution')
plt.title(f'Top {min(20, TOP_K)} Features by Gradient Attribution')
plt.grid(axis='x', alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(activation_freq[top_k_indices], 
           mean_attribution[top_k_indices],
           c=range(len(top_k_indices)), cmap='viridis',
           s=100, alpha=0.6, edgecolors='black')
plt.xlabel('Activation Frequency')
plt.ylabel('Mean Attribution')
plt.title('Attribution vs Activation Frequency')
plt.colorbar(label='Rank')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'top_features_grad_x_act.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. 归因分布
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(mean_attribution, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Mean Attribution')
plt.ylabel('Count')
plt.title('Distribution of Feature Attribution')
plt.yscale('log')

plt.subplot(1, 3, 2)
plt.hist(activation_freq, bins=50, edgecolor='black', alpha=0.7, color='coral')
plt.xlabel('Activation Frequency')
plt.ylabel('Count')
plt.title('Distribution of Activation Frequency')

plt.subplot(1, 3, 3)
plt.scatter(mean_attribution, activation_freq, alpha=0.3, s=10)
top_20_indices = top_k_indices[:20]
plt.scatter(mean_attribution[top_20_indices], 
           activation_freq[top_20_indices],
           c='red', s=100, alpha=0.8, label=f'Top {20}')
plt.xlabel('Mean Attribution')
plt.ylabel('Activation Frequency')
plt.title('Attribution vs Frequency (All Features)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'attribution_distribution_grad_x_act.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Per-dimension heatmap
plt.figure(figsize=(15, 5))

# 收集每个维度的top features
top_features_matrix = np.zeros((3, TOP_K))
for dim_idx, dim_name in enumerate(['x', 'y', 'z']):
    top_dim_indices = attribution_per_dim[dim_name]['top_indices'][:TOP_K]
    top_dim_values = [attribution_per_dim[dim_name]['correlations'][i] for i in top_dim_indices]
    top_features_matrix[dim_idx, :len(top_dim_values)] = top_dim_values

plt.subplot(1, 2, 1)
sns.heatmap(top_features_matrix, cmap='YlOrRd', 
            yticklabels=['X', 'Y', 'Z'],
            xticklabels=[f'F{i}' for i in range(TOP_K)],
            cbar_kws={'label': 'Correlation'},
            vmin=0, vmax=1)
plt.title(f'Top {TOP_K} Features Correlation per Dimension')
plt.xlabel('Feature Rank')
plt.ylabel('Dimension')

# 画维度重叠的韦恩图（简化版）
plt.subplot(1, 2, 2)
overlap_data = [
    len(top_x - top_y - top_z),  # Only X
    len(top_y - top_x - top_z),  # Only Y
    len(top_z - top_x - top_y),  # Only Z
    len(overlap_xy - overlap_xyz),  # X-Y only
    len(overlap_xz - overlap_xyz),  # X-Z only
    len(overlap_yz - overlap_xyz),  # Y-Z only
    len(overlap_xyz)  # All three
]
labels = ['X only', 'Y only', 'Z only', 'X∩Y', 'X∩Z', 'Y∩Z', 'X∩Y∩Z']
colors_venn = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#6FCAC8', '#95E1D3']
plt.bar(range(len(overlap_data)), overlap_data, color=colors_venn, alpha=0.7)
plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
plt.ylabel('Number of Features')
plt.title('Feature Overlap between Dimensions')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'attribution_heatmap_grad_x_act.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. 特征重叠详细可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 每个维度的top特征（动态数量）
for idx, (dim_name, color) in enumerate([('x', '#FF6B6B'), ('y', '#4ECDC4'), ('z', '#45B7D1')]):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # 获取该维度的top特征，确保数量一致
    top_indices = attribution_per_dim[dim_name]['top_indices']
    top_values = [attribution_per_dim[dim_name]['correlations'][i] for i in top_indices]
    
    # 限制显示数量
    n_features = min(len(top_indices), 15)
    top_indices_plot = top_indices[:n_features]
    top_values_plot = top_values[:n_features]
    
    if len(top_values_plot) > 0:
        bars = ax.barh(range(n_features), top_values_plot[::-1], color=color, alpha=0.7)
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([f'F{i}' for i in top_indices_plot[::-1]])
        ax.set_xlabel('Correlation with Target')
        ax.set_title(f'{dim_name.upper()}-axis Top {n_features} Features', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No significant features', 
                ha='center', va='center', transform=ax.transAxes)

# 重叠分析
ax = axes[1, 1]
overlap_counts = {
    'X only': len(top_x - top_y - top_z),
    'Y only': len(top_y - top_x - top_z),
    'Z only': len(top_z - top_x - top_y),
    'X∩Y': len(overlap_xy - overlap_xyz),
    'X∩Z': len(overlap_xz - overlap_xyz),
    'Y∩Z': len(overlap_yz - overlap_xyz),
    'All': len(overlap_xyz)
}
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#6FCAC8', '#95E1D3']
bars = ax.bar(range(len(overlap_counts)), list(overlap_counts.values()), 
              color=colors_bar, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(overlap_counts)))
ax.set_xticklabels(list(overlap_counts.keys()), rotation=45, ha='right')
ax.set_ylabel('Number of Features')
ax.set_title('Feature Overlap Analysis', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 在柱状图上标注数值
for i, (bar, count) in enumerate(zip(bars, overlap_counts.values())):
    if count > 0:
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

plt.suptitle(f'Gradient Attribution Analysis - Layer {LAYER}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_overlap_grad_x_act.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"All visualizations saved to {OUTPUT_DIR}")

print("\n" + "="*50)
print("Gradient Attribution Analysis Complete!")
print("="*50)




