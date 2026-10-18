"""
极简版：使用单层神经网络 + 强正则化
目标：验证是否存在可学习的非线性模式，而不追求高性能
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
    print(f"Error: {e}")
    sys.exit(1)

# =========================
# 极简网络（单隐藏层）
# =========================
class MinimalPredictor(nn.Module):
    """单隐藏层，强正则化"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def train_simple(model, train_loader, val_loader, device, epochs=50, lr=0.0001):
    """简单训练循环，强调泛化"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)  # 强weight decay
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 严格梯度裁剪
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_loss += criterion(model(batch_x), batch_y).item()
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    print(f"Best model from epoch {best_epoch+1}")
    return model


# =========================
# 参数
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--sae_path", type=str, required=True)
parser.add_argument("--train_data_file", type=str, required=True)
parser.add_argument("--test_data_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./sae_simple_nn")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--top_k", type=int, default=100)  # 只用100个特征
parser.add_argument("--hidden_dim", type=int, default=32)  # 很小的隐藏层
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print("="*70)
print("SAE Simple Neural Network Analysis")
print("="*70)
print(f"Strategy: Minimal network + Strong regularization")
print(f"Top-K: {args.top_k}, Hidden: {args.hidden_dim}")
print("="*70)

# =========================
# 加载
# =========================
print("\nLoading...")
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
sae = sae.to(args.device).eval()

# =========================
# 数据
# =========================
print("Loading data...")
with open(args.train_data_file) as f:
    train_data = json.load(f)
with open(args.test_data_file) as f:
    test_data = json.load(f)

def collect(data):
    X, Y = [], []
    for s in tqdm(data, desc="Collecting"):
        tokens = model.to_tokens(s['prompt'], truncate=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=f"blocks.{LAYER}.hook_resid_post")
            h = cache[f"blocks.{LAYER}.hook_resid_post"][0, -1]
            acts = sae.encode(h.unsqueeze(0))[0].cpu().numpy()
        X.append(acts)
        Y.append(np.array(s['target']))
    return np.array(X), np.array(Y)

X_train, Y_train = collect(train_data)
X_test, Y_test = collect(test_data)

# =========================
# 特征选择
# =========================
print("\nFeature selection with Ridge...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, Y_train)
importance = np.abs(ridge.coef_).mean(axis=0)
top_k_idx = np.argsort(importance)[-args.top_k:][::-1]

X_train_sel = X_train[:, top_k_idx]
X_test_sel = X_test[:, top_k_idx]

# Baseline
Y_pred_ridge = ridge.predict(X_test)
r2_ridge = r2_score(Y_test, Y_pred_ridge, multioutput='uniform_average')
print(f"Ridge baseline R²: {r2_ridge:.4f}")

# =========================
# 神经网络
# =========================
print(f"\nTraining minimal network (top-{args.top_k} features)...")

# 标准化
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train_sel)
Y_train_sc = scaler_Y.fit_transform(Y_train)
X_test_sc = scaler_X.transform(X_test_sel)

# 数据
train_ds = TensorDataset(torch.FloatTensor(X_train_sc), torch.FloatTensor(Y_train_sc))
test_ds = TensorDataset(torch.FloatTensor(X_test_sc), torch.FloatTensor(scaler_Y.transform(Y_test)))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 网络
predictor = MinimalPredictor(args.top_k, args.hidden_dim, 3, dropout=0.5).to(args.device)
params = sum(p.numel() for p in predictor.parameters())
print(f"Parameters: {params:,}")

# 训练
predictor = train_simple(predictor, train_loader, test_loader, args.device, epochs=50, lr=0.0001)

# =========================
# 评估
# =========================
print("\n" + "="*70)
print("Results")
print("="*70)

def eval_loader(loader):
    predictor.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(predictor(x.to(args.device)).cpu().numpy())
            targets.append(y.numpy())
    return np.vstack(preds), np.vstack(targets)

train_pred_sc, train_tgt_sc = eval_loader(train_loader)
test_pred_sc, test_tgt_sc = eval_loader(test_loader)

# 反标准化
train_pred = scaler_Y.inverse_transform(train_pred_sc)
train_tgt = scaler_Y.inverse_transform(train_tgt_sc)
test_pred = scaler_Y.inverse_transform(test_pred_sc)
test_tgt = Y_test  # 已经是原始尺度

r2_train = r2_score(train_tgt, train_pred, multioutput='uniform_average')
r2_test = r2_score(test_tgt, test_pred, multioutput='uniform_average')
mae_test = mean_absolute_error(test_tgt, test_pred)

r2_dims = [r2_score(test_tgt[:, i], test_pred[:, i]) for i in range(3)]

print(f"\nRidge (baseline): R²={r2_ridge:.4f}")
print(f"\nMinimal NN:")
print(f"  Train R²: {r2_train:.4f}")
print(f"  Test R²:  {r2_test:.4f}")
print(f"  Test MAE: {mae_test:.4f}")
print(f"  Per-dim: X={r2_dims[0]:.4f}, Y={r2_dims[1]:.4f}, Z={r2_dims[2]:.4f}")

gap = r2_train - r2_test
print(f"\n  Overfitting gap: {gap:.4f}")
if gap > 0.1:
    print("  ⚠️  Still overfitting")
elif gap > 0.05:
    print("  ⚠️  Mild overfitting")
else:
    print("  ✓  Generalization OK")

if r2_test > r2_ridge:
    improvement = (r2_test - r2_ridge) / r2_ridge * 100
    print(f"\n  ✓ Improvement over Ridge: +{improvement:.1f}%")
    print("  → Nonlinear effects detected!")
else:
    decline = (r2_ridge - r2_test) / r2_ridge * 100
    print(f"\n  ✗ Worse than Ridge: -{decline:.1f}%")
    print("  → Task appears mostly linear")

# =========================
# 保存
# =========================
results = {
    'top_k': args.top_k,
    'hidden_dim': args.hidden_dim,
    'params': int(params),
    'ridge_r2': float(r2_ridge),
    'nn_train_r2': float(r2_train),
    'nn_test_r2': float(r2_test),
    'nn_test_mae': float(mae_test),
    'r2_per_dim': [float(x) for x in r2_dims],
    'overfit_gap': float(gap),
}

with open(os.path.join(args.output_dir, f"results_layer{LAYER}.json"), 'w') as f:
    json.dump(results, f, indent=2)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, ax in enumerate(axes):
    ax.scatter(test_tgt[:, i], test_pred[:, i], alpha=0.6, s=30)
    lims = [min(test_tgt[:, i].min(), test_pred[:, i].min()),
            max(test_tgt[:, i].max(), test_pred[:, i].max())]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlabel(f'True {["X","Y","Z"][i]}')
    ax.set_ylabel(f'Pred {["X","Y","Z"][i]}')
    ax.set_title(f'{["X","Y","Z"][i]} (R²={r2_dims[i]:.3f})')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, f'predictions_layer{LAYER}.png'), dpi=200, bbox_inches='tight')

print(f"\nSaved to: {args.output_dir}")
print("="*70)




