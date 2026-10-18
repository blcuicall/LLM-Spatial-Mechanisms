"""
使用神经网络（非线性预测器）分析SAE特征与空间坐标的关系
相比线性模型(Ridge/Lasso)，神经网络可以捕捉特征间的复杂交互
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
# 定义神经网络模型
# =========================
class FeaturePredictor(nn.Module):
    """多层感知机预测器"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=3, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class AttentionFeaturePredictor(nn.Module):
    """带自注意力机制的预测器，可以学习特征间的交互"""
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, output_dim=3, dropout=0.3):
        super().__init__()
        
        # 投影到hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # x: [batch, input_dim]
        x = self.input_projection(x)  # [batch, hidden_dim]
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)  # [batch, 1, hidden_dim]
        x = self.ln1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch, hidden_dim]
        
        # Output
        return self.output_head(x)


# =========================
# 训练函数
# =========================
def train_model(model, train_loader, val_loader, device, 
                epochs=100, lr=0.001, patience=10):
    """训练神经网络模型"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\nTraining neural network...")
    print(f"Epochs: {epochs}, Learning rate: {lr}, Patience: {patience}")
    
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
            
            # Gradient clipping
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
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"\nLearning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Best Val: {best_val_loss:.6f}, Patience: {patience_counter}/{patience}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
    
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # 计算指标
    r2 = r2_score(targets, predictions, multioutput='uniform_average')
    mae = mean_absolute_error(targets, predictions)
    
    r2_per_dim = []
    mae_per_dim = []
    for i in range(targets.shape[1]):
        r2_dim = r2_score(targets[:, i], predictions[:, i])
        mae_dim = mean_absolute_error(targets[:, i], predictions[:, i])
        r2_per_dim.append(r2_dim)
        mae_per_dim.append(mae_dim)
    
    return {
        'r2': r2,
        'mae': mae,
        'r2_per_dim': r2_per_dim,
        'mae_per_dim': mae_per_dim,
        'predictions': predictions,
        'targets': targets
    }


# =========================
# 特征重要性分析（基于梯度）
# =========================
def compute_gradient_importance(model, X_test, Y_test, device, top_k=50):
    """使用梯度计算特征重要性"""
    model.eval()
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    X_tensor.requires_grad = True
    
    Y_tensor = torch.FloatTensor(Y_test).to(device)
    
    # Forward pass
    outputs = model(X_tensor)
    loss = nn.MSELoss()(outputs, Y_tensor)
    
    # Backward pass
    loss.backward()
    
    # 获取梯度
    gradients = X_tensor.grad.abs().mean(dim=0).cpu().numpy()  # [d_sae]
    
    # Top-K特征
    top_k_indices = np.argsort(gradients)[-top_k:][::-1]
    
    return gradients, top_k_indices


# =========================
# 参数解析
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", "-m", type=str, required=True)
parser.add_argument("--sae_path", "-s", type=str, required=True, help="Path to trained SAE")
parser.add_argument("--train_data_file", "-tr", type=str, required=True)
parser.add_argument("--test_data_file", "-te", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, default="./sae_analysis_nonlinear")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--top_k", type=int, default=50, help="Top K features to analyze")
parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "attention"],
                   help="Type of neural network: mlp or attention")
parser.add_argument("--hidden_dims", type=str, default="256,128,64", 
                   help="Hidden dimensions for MLP (comma-separated)")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--dropout", type=float, default=0.5)
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

print("="*70)
print("SAE Feature Analysis with Neural Networks (Nonlinear Predictor)")
print("="*70)
print(f"Model: {MODEL_NAME}")
print(f"SAE Path: {SAE_PATH}")
print(f"Predictor Type: {args.model_type}")
print(f"Hidden Dims: {args.hidden_dims}")
print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
print(f"Output Dir: {OUTPUT_DIR}")
print("="*70)

# =========================
# 加载模型
# =========================
print("\nLoading language model...")
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
    """收集SAE特征激活和对应的target"""
    features = []
    targets = []
    
    for sample in tqdm(data, desc="Collecting SAE features"):
        prompt = sample['prompt']
        target = np.array(sample['target'])
        
        tokens = model.to_tokens(prompt, truncate=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_resid_post"
            )
        
        hidden = cache[f"blocks.{layer_idx}.hook_resid_post"][0, -1]
        
        with torch.no_grad():
            feature_acts = sae.encode(hidden.unsqueeze(0))[0]
        
        features.append(feature_acts.cpu().numpy())
        targets.append(target)
    
    return np.array(features), np.array(targets)

print(f"\nCollecting SAE features from layer {LAYER}...")
X_train, Y_train = collect_sae_features(train_data, model, sae, LAYER)
X_test, Y_test = collect_sae_features(test_data, model, sae, LAYER)

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# =========================
# 数据标准化
# =========================
print("\nStandardizing features...")
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
Y_train_scaled = scaler_Y.fit_transform(Y_train)
X_test_scaled = scaler_X.transform(X_test)
Y_test_scaled = scaler_Y.transform(Y_test)

# 创建数据加载器
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(Y_train_scaled)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test_scaled),
    torch.FloatTensor(Y_test_scaled)
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# =========================
# 构建神经网络
# =========================
print("\n" + "="*70)
print(f"Building {args.model_type.upper()} predictor...")
print("="*70)

input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]

if args.model_type == "mlp":
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    predictor = FeaturePredictor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=args.dropout
    )
elif args.model_type == "attention":
    predictor = AttentionFeaturePredictor(
        input_dim=input_dim,
        hidden_dim=512,
        num_heads=8,
        output_dim=output_dim,
        dropout=args.dropout
    )

predictor = predictor.to(DEVICE)

# 统计参数数量
total_params = sum(p.numel() for p in predictor.parameters())
trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# =========================
# 训练模型
# =========================
predictor, train_losses, val_losses = train_model(
    predictor, train_loader, test_loader, DEVICE,
    epochs=args.epochs, lr=args.lr, patience=15
)

# =========================
# 评估模型
# =========================
print("\n" + "="*70)
print("Evaluating Neural Network on Train and Test Sets")
print("="*70)

train_results = evaluate_model(predictor, train_loader, DEVICE)
test_results = evaluate_model(predictor, test_loader, DEVICE)

# 反标准化预测结果 - 训练集
train_predictions_original = scaler_Y.inverse_transform(train_results['predictions'])
train_targets_original = scaler_Y.inverse_transform(train_results['targets'])

r2_train = r2_score(train_targets_original, train_predictions_original, multioutput='uniform_average')
mae_train = mean_absolute_error(train_targets_original, train_predictions_original)

# 反标准化预测结果 - 测试集
predictions_original = scaler_Y.inverse_transform(test_results['predictions'])
targets_original = scaler_Y.inverse_transform(test_results['targets'])

# 重新计算指标（原始尺度）
r2_original = r2_score(targets_original, predictions_original, multioutput='uniform_average')
mae_original = mean_absolute_error(targets_original, predictions_original)

r2_per_dim_original = []
for i in range(targets_original.shape[1]):
    r2_dim = r2_score(targets_original[:, i], predictions_original[:, i])
    r2_per_dim_original.append(r2_dim)

print(f"\nNeural Network Performance:")
print(f"  TRAIN SET:")
print(f"    R² (overall): {r2_train:.4f}")
print(f"    MAE: {mae_train:.4f}")
print(f"  TEST SET:")
print(f"    R² (overall): {r2_original:.4f}")
print(f"    MAE: {mae_original:.4f}")
print(f"    R²(x): {r2_per_dim_original[0]:.4f}")
print(f"    R²(y): {r2_per_dim_original[1]:.4f}")
print(f"    R²(z): {r2_per_dim_original[2]:.4f}")

# 检查过拟合
if r2_train > 0 and r2_original > 0:
    overfitting_gap = r2_train - r2_original
    print(f"\n  Overfitting Analysis:")
    print(f"    Gap (Train R² - Test R²): {overfitting_gap:.4f}")
    if overfitting_gap > 0.2:
        print(f"    ⚠️  Severe overfitting detected!")
    elif overfitting_gap > 0.1:
        print(f"    ⚠️  Moderate overfitting detected")
    else:
        print(f"    ✓  Overfitting is under control")

# =========================
# 特征重要性分析（基于梯度）
# =========================
print("\n" + "="*70)
print("Feature Importance Analysis (Gradient-based)")
print("="*70)

gradients, top_k_indices = compute_gradient_importance(
    predictor, X_test_scaled, Y_test_scaled, DEVICE, TOP_K
)

print(f"\nTop {TOP_K} most important features (by gradient):")
for rank, idx in enumerate(top_k_indices[:20], 1):
    print(f"  {rank}. Feature {idx}: gradient={gradients[idx]:.6f}")

# =========================
# 保存结果
# =========================
print("\n" + "="*70)
print("Saving Results")
print("="*70)

results = {
    'model_name': MODEL_NAME,
    'layer': LAYER,
    'd_sae': sae_config['d_sae'],
    'predictor_type': args.model_type,
    'predictor_params': {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'hidden_dims': args.hidden_dims if args.model_type == "mlp" else "attention",
        'dropout': args.dropout,
    },
    'training': {
        'epochs_trained': len(train_losses),
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
    },
    'performance': {
        'r2': float(r2_original),
        'mae': float(mae_original),
        'r2_per_dim': [float(x) for x in r2_per_dim_original],
    },
    'top_features_gradient': {
        'indices': top_k_indices.tolist(),
        'gradients': [float(gradients[i]) for i in top_k_indices],
    },
}

output_file = os.path.join(OUTPUT_DIR, f"sae_analysis_nn_layer{LAYER}.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {output_file}")

# 保存模型
model_save_path = os.path.join(OUTPUT_DIR, f"predictor_layer{LAYER}.pt")
torch.save({
    'model_state_dict': predictor.state_dict(),
    'scaler_X': scaler_X,
    'scaler_Y': scaler_Y,
    'config': results['predictor_params'],
}, model_save_path)
print(f"Model saved to {model_save_path}")

# =========================
# 可视化
# =========================
print("\nGenerating visualizations...")

# 1. 训练曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(val_losses, label='Validation Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'training_curve_layer{LAYER}.png'), dpi=300)
plt.close()

# 2. 预测 vs 真实值
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
dim_names = ['X', 'Y', 'Z']
for i, (ax, dim_name) in enumerate(zip(axes, dim_names)):
    ax.scatter(targets_original[:, i], predictions_original[:, i], alpha=0.5, s=20)
    
    # 添加y=x线
    min_val = min(targets_original[:, i].min(), predictions_original[:, i].min())
    max_val = max(targets_original[:, i].max(), predictions_original[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7)
    
    ax.set_xlabel(f'True {dim_name}')
    ax.set_ylabel(f'Predicted {dim_name}')
    ax.set_title(f'{dim_name}-axis (R²={r2_per_dim_original[i]:.3f})')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'predictions_vs_true_layer{LAYER}.png'), dpi=300)
plt.close()

# 3. 特征重要性（梯度）
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(gradients, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Gradient Magnitude')
plt.ylabel('Count')
plt.title('Distribution of Feature Importance (Gradient)')
plt.yscale('log')

plt.subplot(1, 2, 2)
top_20_gradients = [gradients[i] for i in top_k_indices[:20]]
plt.barh(range(20), top_20_gradients[::-1])
plt.xlabel('Gradient Magnitude')
plt.ylabel('Feature Rank')
plt.title('Top 20 Features by Gradient')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'feature_importance_gradient_layer{LAYER}.png'), dpi=300)
plt.close()

print(f"Visualizations saved to {OUTPUT_DIR}")

print("\n" + "="*70)
print("SAE Feature Analysis with Neural Networks Complete!")
print("="*70)
print(f"\nKey Results:")
print(f"  Neural Network R²: {r2_original:.4f}")
print(f"  Parameters: {trainable_params:,}")
print(f"  Training epochs: {len(train_losses)}")
print("="*70)

