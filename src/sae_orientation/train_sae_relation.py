"""
训练SAE用于空间推理任务（基于SAELens）
"""
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from config import PATHS

import os
import json
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

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
parser.add_argument("--model_name", "-m", type=str, required=True, help="Model name")
parser.add_argument("--train_data_file", "-tr", type=str, required=True, help="Training data file")
parser.add_argument("--layer", "-l", type=int, required=True, help="Layer to train SAE on")
parser.add_argument("--expansion_factor", "-e", type=int, default=8, help="SAE expansion factor")
parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
parser.add_argument("--num_tokens", "-n", type=int, default=100000, help="Number of tokens to train on")
parser.add_argument("--l1_coefficient", type=float, default=0.001, help="L1 sparsity coefficient")
parser.add_argument("--output_dir", "-o", type=str, default="./sae_checkpoints", help="Output directory")
parser.add_argument("--device", type=str, default="cuda:0", help="Device")
args = parser.parse_args()

# =========================
# 配置
# =========================
MODEL_NAME = args.model_name
MODEL_PATH = PATHS[MODEL_NAME]
DEVICE = args.device
DTYPE = torch.float32
LAYER = args.layer
EXPANSION_FACTOR = args.expansion_factor
BATCH_SIZE = args.batch_size
NUM_TOKENS = args.num_tokens
L1_COEFFICIENT = args.l1_coefficient
TRAIN_DATA_FILE = args.train_data_file
LR = 3e-4  # Learning rate
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# 创建输出目录
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = time.strftime('%Y%m%d_%H%M%S')
exp_name = f"sae_layer{LAYER}_exp{EXPANSION_FACTOR}_{timestamp}"
exp_dir = output_dir / exp_name
exp_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = str(exp_dir)

print("="*50)
print("SAE Training Configuration")
print("="*50)
print(f"Model: {MODEL_NAME}")
print(f"Layer: {LAYER}")
print(f"Expansion Factor: {EXPANSION_FACTOR}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Num Tokens: {NUM_TOKENS}")
print(f"L1 Coefficient: {L1_COEFFICIENT}")
print(f"Output Dir: {OUTPUT_DIR}")
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

print("Converting to HookedTransformer...")
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

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model

print(f"Model loaded: {n_layers} layers, d_model={d_model}")

# =========================
# 加载数据
# =========================
print(f"\nLoading training data from {TRAIN_DATA_FILE}...")
with open(TRAIN_DATA_FILE, "r") as f:
    train_data = json.load(f)

print(f"Loaded {len(train_data)} training samples")

# =========================
# 准备激活数据集
# =========================
print(f"\nCollecting activations from layer {LAYER}...")

def collect_activations(data, layer_idx, max_tokens=None):
    """收集指定层的激活值"""
    activations = []
    token_count = 0
    
    for sample in tqdm(data, desc=f"Collecting activations"):
        if max_tokens and token_count >= max_tokens:
            break
            
        prompt = sample['prompt']
        tokens = model.to_tokens(prompt, truncate=True)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer_idx}.hook_resid_post"
            )
        
        # 获取所有token的激活（不只是最后一个）
        layer_acts = cache[f"blocks.{layer_idx}.hook_resid_post"][0]  # [seq_len, d_model]
        
        for token_act in layer_acts:
            activations.append(token_act.cpu())
            token_count += 1
            if max_tokens and token_count >= max_tokens:
                break
    
    return torch.stack(activations)  # [num_tokens, d_model]

activation_data = collect_activations(train_data, LAYER, max_tokens=NUM_TOKENS)
print(f"Collected activations shape: {activation_data.shape}")

# =========================
# 配置SAE训练
# =========================
print("\nConfiguring SAE training...")

# SAE配置
d_sae = d_model * EXPANSION_FACTOR

# 使用 StandardTrainingSAEConfig
sae_cfg = StandardTrainingSAEConfig(
    d_in=d_model,
    d_sae=d_sae,
    l1_coefficient=L1_COEFFICIENT,
    dtype="float32",
    device=str(DEVICE),
    apply_b_dec_to_input=True,
    normalize_activations="none",
)

print(f"SAE config: d_in={d_model}, d_sae={d_sae}")

# =========================
# 初始化SAE
# =========================
print("\nInitializing SAE...")

sae = StandardTrainingSAE(sae_cfg)
sae = sae.to(DEVICE)

print(f"SAE architecture:")
print(f"  Encoder: {d_model} -> {d_sae}")
print(f"  Decoder: {d_sae} -> {d_model}")

# =========================
# 训练SAE
# =========================
print("\nTraining SAE...")

# 将激活数据移到设备
activation_data = activation_data.to(DEVICE)

# 训练循环
num_batches = len(activation_data) // BATCH_SIZE
optimizer = torch.optim.Adam(sae.parameters(), lr=LR)

print(f"Training for {num_batches} batches...")

training_history = {
    'losses': [],
    'reconstruction_losses': [],
    'l1_losses': [],
    'sparsities': [],
    'l0s': [],
}

for batch_idx in tqdm(range(num_batches), desc="Training SAE"):
    start_idx = batch_idx * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    batch = activation_data[start_idx:end_idx]
    
    # Forward pass using encode/decode
    feature_acts = sae.encode(batch)
    reconstructed = sae.decode(feature_acts)
    
    # 计算损失
    reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, batch)
    l1_loss = L1_COEFFICIENT * feature_acts.abs().sum(dim=-1).mean()
    loss = reconstruction_loss + l1_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # 记录统计信息
    with torch.no_grad():
        sparsity = (feature_acts.abs() > 1e-8).float().mean().item()
        l0 = (feature_acts.abs() > 1e-8).float().sum(dim=-1).mean().item()
    
    training_history['losses'].append(loss.item())
    training_history['reconstruction_losses'].append(reconstruction_loss.item())
    training_history['l1_losses'].append(l1_loss.item())
    training_history['sparsities'].append(sparsity)
    training_history['l0s'].append(l0)
    
    if batch_idx % 100 == 0:
        print(f"\nBatch {batch_idx}/{num_batches}")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Reconstruction: {reconstruction_loss.item():.6f}")
        print(f"  L1: {l1_loss.item():.6f}")
        print(f"  Sparsity: {sparsity:.4f}")
        print(f"  L0: {l0:.2f}/{d_sae}")

# =========================
# 保存SAE
# =========================
print("\nSaving SAE...")
save_path = os.path.join(OUTPUT_DIR, "sae_final.pt")
torch.save({
    'sae_state_dict': sae.state_dict(),
    'config': {
        'model_name': MODEL_NAME,
        'layer': LAYER,
        'd_model': d_model,
        'd_sae': d_sae,
        'expansion_factor': EXPANSION_FACTOR,
        'l1_coefficient': L1_COEFFICIENT,
        'num_tokens': NUM_TOKENS,
        'lr': LR,
    },
    'training_history': training_history,
}, save_path)

print(f"SAE saved to {save_path}")

# 保存训练历史
history_path = os.path.join(OUTPUT_DIR, "training_history.json")
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"Training history saved to {history_path}")

# =========================
# 评估SAE
# =========================
print("\nEvaluating SAE...")

sae.eval()
with torch.no_grad():
    # 随机采样一些激活
    sample_size = min(1000, len(activation_data))
    sample_indices = torch.randperm(len(activation_data))[:sample_size]
    sample_acts = activation_data[sample_indices]
    
    # SAE前向传播
    feature_acts = sae.encode(sample_acts)
    reconstructed = sae.decode(feature_acts)
    
    # 计算重构误差
    reconstruction_error = torch.nn.functional.mse_loss(reconstructed, sample_acts)
    
    # 计算稀疏度和L0
    sparsity = (feature_acts.abs() > 1e-8).float().mean()
    l0 = (feature_acts.abs() > 1e-8).float().sum(dim=-1).mean()
    
    # 计算explained variance
    var_orig = sample_acts.var()
    var_error = (sample_acts - reconstructed).var()
    explained_var = 1 - var_error / var_orig
    
    print("\n" + "="*50)
    print("SAE Evaluation Metrics")
    print("="*50)
    print(f"Reconstruction Error (MSE): {reconstruction_error.item():.6f}")
    print(f"Explained Variance: {explained_var.item():.4f}")
    print(f"Sparsity: {sparsity.item():.4f}")
    print(f"L0 (avg active features): {l0.item():.2f} / {d_sae}")
    print("="*50)

# 保存评估结果
eval_results = {
    'reconstruction_error': reconstruction_error.item(),
    'explained_variance': explained_var.item(),
    'sparsity': sparsity.item(),
    'l0': l0.item(),
    'd_sae': d_sae,
    'layer': LAYER,
    'model_name': MODEL_NAME,
}

eval_path = os.path.join(OUTPUT_DIR, "eval_results.json")
with open(eval_path, 'w') as f:
    json.dump(eval_results, f, indent=2)

print(f"\nEvaluation results saved to {eval_path}")
print(f"\nSAE training complete! Model saved to: {OUTPUT_DIR}")





