"""
分析SAE特征与空间过程执行任务的关系
找出哪些SAE特征对空间坐标变换最重要
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error
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
parser.add_argument("--train_data_file", "-tr", type=str, required=True)
parser.add_argument("--test_data_file", "-te", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default="./sae_analysis")
parser.add_argument("--device", type=str, default="cuda:0")
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
TRAIN_DATA_FILE = args.train_data_file
TEST_DATA_FILE = args.test_data_file
OUTPUT_DIR = args.output_dir
TOP_K = args.top_k

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*50)
print("SAE Feature Analysis Configuration (Spatial Procedure)")
print("="*50)
print(f"Model: {MODEL_NAME}")
print(f"SAE Path: {SAE_PATH}")
print(f"Output Dir: {OUTPUT_DIR}")
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
print("\nLoading data...")
with open(TRAIN_DATA_FILE, "r") as f:
    train_data = json.load(f)

with open(TEST_DATA_FILE, "r") as f:
    test_data = json.load(f)

print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# =========================
# 收集SAE特征激活
# =========================
def collect_sae_features(data, model, sae, layer_idx):
    """收集SAE特征激活和对应的target (最终坐标)"""
    features = []
    targets = []
    
    for sample in tqdm(data, desc="Collecting SAE features"):
        prompt = sample['prompt']
        # target 是最终位置坐标 [x, y, z]
        target = np.array(sample['target'])
        
        tokens = model.to_tokens(prompt, truncate=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_resid_post"
            )
        
        # 获取最后一个token的激活
        hidden = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1]  # [d_model]
        
        # 通过SAE编码
        with torch.no_grad():
            feature_acts = sae.encode(hidden.unsqueeze(0))[0]  # [d_sae]
        
        features.append(feature_acts.cpu().numpy())
        targets.append(target)
    
    return np.array(features), np.array(targets)

print(f"\nCollecting SAE features from layer {LAYER}...")
X_train, Y_train = collect_sae_features(train_data, model, sae, LAYER)
X_test, Y_test = collect_sae_features(test_data, model, sae, LAYER)

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# =========================
# 分析1: 使用SAE特征进行Ridge回归
# =========================
print("\n" + "="*50)
print("Analysis 1: Ridge Regression on SAE Features")
print("="*50)

probe = Ridge(alpha=1.0)
probe.fit(X_train, Y_train)
Y_pred = probe.predict(X_test)

r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')
mae = mean_absolute_error(Y_test, Y_pred)

# 每个维度的R²
r2_per_dim = []
for i in range(Y_test.shape[1]):
    r2_dim = r2_score(Y_test[:, i], Y_pred[:, i])
    r2_per_dim.append(r2_dim)

print(f"R² (overall): {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²(x): {r2_per_dim[0]:.4f}, R²(y): {r2_per_dim[1]:.4f}, R²(z): {r2_per_dim[2]:.4f}")

# =========================
# 分析2: 特征重要性分析
# =========================
print("\n" + "="*50)
print("Analysis 2: Feature Importance")
print("="*50)

# 获取回归系数（绝对值）
coefficients = np.abs(probe.coef_)  # [3, d_sae]

# 计算每个特征对所有维度的平均重要性
feature_importance = coefficients.mean(axis=0)  # [d_sae]

# 找出Top K重要特征
top_k_indices = np.argsort(feature_importance)[-TOP_K:][::-1]

print(f"\nTop {TOP_K} most important SAE features:")
for rank, idx in enumerate(top_k_indices[:20], 1):  # 显示前20个
    print(f"  {rank}. Feature {idx}: importance={feature_importance[idx]:.6f}")

# 每个维度的Top特征
print("\nTop features per dimension:")
for dim, dim_name in enumerate(['x', 'y', 'z']):
    top_indices_dim = np.argsort(coefficients[dim])[-10:][::-1]
    print(f"\n  {dim_name}-axis (Top 10):")
    for rank, idx in enumerate(top_indices_dim, 1):
        print(f"    {rank}. Feature {idx}: coef={coefficients[dim, idx]:.6f}")

# =========================
# 分析3: 特征激活统计
# =========================
print("\n" + "="*50)
print("Analysis 3: Feature Activation Statistics")
print("="*50)

# 计算激活频率
activation_freq = (X_train > 0).mean(axis=0)  # [d_sae]
activation_mean = X_train.mean(axis=0)  # [d_sae]
activation_std = X_train.std(axis=0)  # [d_sae]

print("\nTop features activation statistics:")
for rank, idx in enumerate(top_k_indices[:20], 1):
    print(f"  {rank}. Feature {idx}:")
    print(f"      Freq: {activation_freq[idx]:.4f}, Mean: {activation_mean[idx]:.4f}, Std: {activation_std[idx]:.4f}")

# =========================
# 分析4: Sparse Regression (找出真正重要的特征)
# =========================
print("\n" + "="*50)
print("Analysis 4: Lasso Regression (Sparse Selection)")
print("="*50)

lasso = Lasso(alpha=0.01, max_iter=5000)
lasso.fit(X_train, Y_train)
Y_pred_lasso = lasso.predict(X_test)

r2_lasso = r2_score(Y_test, Y_pred_lasso, multioutput='uniform_average')
mae_lasso = mean_absolute_error(Y_test, Y_pred_lasso)

print(f"Lasso R²: {r2_lasso:.4f}")
print(f"Lasso MAE: {mae_lasso:.4f}")

# 非零系数数量
non_zero_features = np.sum(np.abs(lasso.coef_).sum(axis=0) > 1e-6)
print(f"Non-zero features: {non_zero_features} / {X_train.shape[1]}")

# Top Lasso特征
lasso_importance = np.abs(lasso.coef_).mean(axis=0)
top_lasso_indices = np.argsort(lasso_importance)[-TOP_K:][::-1]

print(f"\nTop {min(20, TOP_K)} Lasso-selected features:")
for rank, idx in enumerate(top_lasso_indices[:20], 1):
    if lasso_importance[idx] > 1e-6:
        print(f"  {rank}. Feature {idx}: importance={lasso_importance[idx]:.6f}")

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
    'ridge_regression': {
        'r2': float(r2),
        'mae': float(mae),
        'r2_per_dim': [float(x) for x in r2_per_dim],
    },
    'lasso_regression': {
        'r2': float(r2_lasso),
        'mae': float(mae_lasso),
        'non_zero_features': int(non_zero_features),
    },
    'top_features': {
        'indices': top_k_indices.tolist(),
        'importance': [float(feature_importance[i]) for i in top_k_indices],
        'activation_freq': [float(activation_freq[i]) for i in top_k_indices],
    },
    'top_features_per_dim': {
        'x': {
            'indices': np.argsort(coefficients[0])[-TOP_K:][::-1].tolist(),
            'coefficients': [float(coefficients[0, i]) for i in np.argsort(coefficients[0])[-TOP_K:][::-1]],
        },
        'y': {
            'indices': np.argsort(coefficients[1])[-TOP_K:][::-1].tolist(),
            'coefficients': [float(coefficients[1, i]) for i in np.argsort(coefficients[1])[-TOP_K:][::-1]],
        },
        'z': {
            'indices': np.argsort(coefficients[2])[-TOP_K:][::-1].tolist(),
            'coefficients': [float(coefficients[2, i]) for i in np.argsort(coefficients[2])[-TOP_K:][::-1]],
        },
    },
}

output_file = os.path.join(OUTPUT_DIR, f"sae_analysis_layer{LAYER}.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")

# =========================
# 可视化
# =========================
print("\nGenerating visualizations...")

# 1. Feature importance分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(feature_importance, bins=50, edgecolor='black')
plt.xlabel('Feature Importance')
plt.ylabel('Count')
plt.title('Distribution of Feature Importance')
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.bar(range(TOP_K), [feature_importance[i] for i in top_k_indices])
plt.xlabel(f'Top {TOP_K} Features (ranked)')
plt.ylabel('Importance')
plt.title(f'Top {TOP_K} Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'feature_importance_layer{LAYER}.png'), dpi=300)
plt.close()

# 2. Per-dimension coefficients heatmap
plt.figure(figsize=(15, 4))
coef_top_k = coefficients[:, top_k_indices[:min(50, TOP_K)]]
sns.heatmap(coef_top_k, cmap='RdBu_r', center=0, 
            yticklabels=['x', 'y', 'z'],
            xticklabels=[f'F{i}' for i in top_k_indices[:min(50, TOP_K)]],
            cbar_kws={'label': 'Coefficient'})
plt.title(f'Top {min(50, TOP_K)} Features - Coefficients per Dimension')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'coefficients_heatmap_layer{LAYER}.png'), dpi=300)
plt.close()

# 3. Activation frequency vs importance
plt.figure(figsize=(8, 6))
plt.scatter(activation_freq, feature_importance, alpha=0.3, s=10)
# 高亮top features
plt.scatter([activation_freq[i] for i in top_k_indices[:20]], 
           [feature_importance[i] for i in top_k_indices[:20]], 
           c='red', s=50, alpha=0.8, label=f'Top {20} features')
plt.xlabel('Activation Frequency')
plt.ylabel('Feature Importance')
plt.title('Feature Importance vs Activation Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'importance_vs_freq_layer{LAYER}.png'), dpi=300)
plt.close()

print(f"Visualizations saved to {OUTPUT_DIR}")

print("\n" + "="*50)
print("SAE Feature Analysis Complete!")
print("="*50)




