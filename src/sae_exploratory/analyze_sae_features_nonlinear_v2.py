"""
改进版：先进行特征选择，再用神经网络训练
解决维度灾难问题
"""
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from config import PATHS

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt

try:
    from sae_lens.saes import StandardTrainingSAE, StandardTrainingSAEConfig
except ImportError as e:
    print(f"Warning: SAELens not installed: {e}")
    sys.exit(1)

# =========================
# 简化的神经网络
# =========================
class SimplePredictor(nn.Module):
    """轻量级MLP，适合特征选择后的低维输入"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, dropout=0.3):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, val_loader, device, epochs=100, lr=0.001, patience=15):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
                  f"Best: {best_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


# =========================
# 参数解析
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, required=True)
parser.add_argument("--sae_path", "-s", type=str, required=True)
parser.add_argument("--train_data_file", "-tr", type=str, required=True)
parser.add_argument("--test_data_file", "-te", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default="./sae_analysis_nonlinear_v2")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--top_k", type=int, default=500, help="Top K features to select")
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0.3)
args = parser.parse_args()

# =========================
# 配置
# =========================
os.makedirs(args.output_dir, exist_ok=True)

print("="*70)
print("SAE Nonlinear Analysis V2 (with Feature Selection)")
print("="*70)
print(f"Model: {args.model_name}")
print(f"Top-K features: {args.top_k}")
print(f"Hidden dim: {args.hidden_dim}")
print("="*70)

# =========================
# 加载模型和SAE
# =========================
print("\nLoading models...")
MODEL_PATH = PATHS[args.model_name]
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, trust_remote_code=True)
hf_model = hf_model.to(args.device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = HookedTransformer.from_pretrained(args.model_name, hf_model=hf_model, tokenizer=tokenizer,
                                          dtype=torch.float32, device=args.device, fold_ln=False,
                                          center_writing_weights=False, center_unembed=False, fold_value_biases=False)
model.eval()

checkpoint = torch.load(args.sae_path, map_location=args.device)
sae_config = checkpoint['config']
LAYER = sae_config['layer']

sae_cfg = StandardTrainingSAEConfig(d_in=sae_config['d_model'], d_sae=sae_config['d_sae'],
                                    l1_coefficient=sae_config.get('l1_coefficient', 0.001),
                                    dtype="float32", device=str(args.device), apply_b_dec_to_input=True,
                                    normalize_activations="none")
sae = StandardTrainingSAE(sae_cfg)
sae.load_state_dict(checkpoint['sae_state_dict'])
sae = sae.to(args.device)
sae.eval()

print(f"SAE: layer={LAYER}, d_sae={sae_config['d_sae']}")

# =========================
# 加载数据并收集特征
# =========================
print("\nLoading data...")
with open(args.train_data_file) as f:
    train_data = json.load(f)
with open(args.test_data_file) as f:
    test_data = json.load(f)

def collect_features(data):
    features, targets = [], []
    for sample in tqdm(data, desc="Collecting features"):
        tokens = model.to_tokens(sample['prompt'], truncate=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=f"blocks.{LAYER}.hook_resid_post")
            hidden = cache[f"blocks.{LAYER}.hook_resid_post"][0, -1]
            feature_acts = sae.encode(hidden.unsqueeze(0))[0]
        features.append(feature_acts.cpu().numpy())
        targets.append(np.array(sample['target']))
    return np.array(features), np.array(targets)

X_train, Y_train = collect_features(train_data)
X_test, Y_test = collect_features(test_data)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# =========================
# 步骤1: 使用Ridge回归选择重要特征
# =========================
print("\n" + "="*70)
print("Step 1: Feature Selection with Ridge Regression")
print("="*70)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)

# 计算特征重要性
feature_importance = np.abs(ridge.coef_).mean(axis=0)
top_k_indices = np.argsort(feature_importance)[-args.top_k:][::-1]

print(f"Selected top {args.top_k} features (from {X_train.shape[1]})")
print(f"Top 10 features: {top_k_indices[:10]}")

# 提取选中的特征
X_train_selected = X_train[:, top_k_indices]
X_test_selected = X_test[:, top_k_indices]

print(f"Reduced dimensions: {X_train.shape[1]} -> {X_train_selected.shape[1]}")

# Ridge在选中特征上的性能
Y_pred_ridge = ridge.predict(X_test)
r2_ridge = r2_score(Y_test, Y_pred_ridge, multioutput='uniform_average')
print(f"Ridge R² (all features): {r2_ridge:.4f}")

# =========================
# 步骤2: 在选中特征上训练神经网络
# =========================
print("\n" + "="*70)
print("Step 2: Training Neural Network on Selected Features")
print("="*70)

# 标准化
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train_selected)
Y_train_scaled = scaler_Y.fit_transform(Y_train)
X_test_scaled = scaler_X.transform(X_test_selected)
Y_test_scaled = scaler_Y.transform(Y_test)

# 数据加载器
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(Y_train_scaled))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(Y_test_scaled))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# 构建网络
predictor = SimplePredictor(input_dim=args.top_k, hidden_dim=args.hidden_dim, 
                            output_dim=3, dropout=args.dropout).to(args.device)

total_params = sum(p.numel() for p in predictor.parameters())
print(f"Network parameters: {total_params:,}")

# 训练
predictor, train_losses, val_losses = train_model(
    predictor, train_loader, test_loader, args.device, 
    epochs=args.epochs, lr=args.lr, patience=15
)

# =========================
# 评估
# =========================
print("\n" + "="*70)
print("Evaluation Results")
print("="*70)

def evaluate(loader):
    predictor.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = predictor(batch_x.to(args.device))
            preds.append(outputs.cpu().numpy())
            targets.append(batch_y.numpy())
    return np.vstack(preds), np.vstack(targets)

train_preds_scaled, train_targets_scaled = evaluate(train_loader)
test_preds_scaled, test_targets_scaled = evaluate(test_loader)

# 反标准化
train_preds = scaler_Y.inverse_transform(train_preds_scaled)
train_targets = scaler_Y.inverse_transform(train_targets_scaled)
test_preds = scaler_Y.inverse_transform(test_preds_scaled)
test_targets = scaler_Y.inverse_transform(test_targets_scaled)

# 计算指标
r2_train_nn = r2_score(train_targets, train_preds, multioutput='uniform_average')
mae_train_nn = mean_absolute_error(train_targets, train_preds)
r2_test_nn = r2_score(test_targets, test_preds, multioutput='uniform_average')
mae_test_nn = mean_absolute_error(test_targets, test_preds)

r2_per_dim = [r2_score(test_targets[:, i], test_preds[:, i]) for i in range(3)]

print(f"\nRidge (baseline, all features):")
print(f"  Test R²: {r2_ridge:.4f}")

print(f"\nNeural Network (top-{args.top_k} features):")
print(f"  Train R²: {r2_train_nn:.4f}, MAE: {mae_train_nn:.4f}")
print(f"  Test R²:  {r2_test_nn:.4f}, MAE: {mae_test_nn:.4f}")
print(f"  R²(x): {r2_per_dim[0]:.4f}, R²(y): {r2_per_dim[1]:.4f}, R²(z): {r2_per_dim[2]:.4f}")

improvement = ((r2_test_nn - r2_ridge) / r2_ridge * 100) if r2_ridge > 0 else 0
print(f"\n  Improvement over Ridge: {improvement:+.1f}%")

overfit_gap = r2_train_nn - r2_test_nn
print(f"  Overfitting gap: {overfit_gap:.4f}")
if overfit_gap > 0.15:
    print("  ⚠️  Severe overfitting")
elif overfit_gap > 0.05:
    print("  ⚠️  Moderate overfitting")
else:
    print("  ✓  Overfitting under control")

# =========================
# 保存结果
# =========================
results = {
    'model_name': args.model_name,
    'layer': LAYER,
    'd_sae': sae_config['d_sae'],
    'top_k_features': args.top_k,
    'selected_features': top_k_indices.tolist(),
    'performance': {
        'ridge_r2_baseline': float(r2_ridge),
        'nn_train_r2': float(r2_train_nn),
        'nn_test_r2': float(r2_test_nn),
        'nn_test_mae': float(mae_test_nn),
        'r2_per_dim': [float(x) for x in r2_per_dim],
        'improvement_over_ridge': float(improvement),
        'overfitting_gap': float(overfit_gap),
    }
}

output_file = os.path.join(args.output_dir, f"results_layer{LAYER}.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

# 保存模型
torch.save({
    'model_state_dict': predictor.state_dict(),
    'scaler_X': scaler_X,
    'scaler_Y': scaler_Y,
    'top_k_indices': top_k_indices,
}, os.path.join(args.output_dir, f"predictor_layer{LAYER}.pt"))

print(f"\nResults saved to: {args.output_dir}")

# =========================
# 可视化
# =========================
# 1. 训练曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train', alpha=0.8)
plt.plot(val_losses, label='Val', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curve')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(args.output_dir, f'training_curve_layer{LAYER}.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. 预测vs真实
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.scatter(test_targets[:, i], test_preds[:, i], alpha=0.5, s=20)
    lims = [min(test_targets[:, i].min(), test_preds[:, i].min()),
            max(test_targets[:, i].max(), test_preds[:, i].max())]
    ax.plot(lims, lims, 'r--', lw=2, alpha=0.7)
    ax.set_xlabel(f'True {["X","Y","Z"][i]}')
    ax.set_ylabel(f'Predicted {["X","Y","Z"][i]}')
    ax.set_title(f'{["X","Y","Z"][i]}-axis (R²={r2_per_dim[i]:.3f})')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'predictions_layer{LAYER}.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)




