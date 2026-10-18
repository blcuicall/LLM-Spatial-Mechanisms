"""
分析SAE特征的语义可解释性
通过最大激活样本分析（Max Activating Examples）来理解特征含义
"""

import torch
import json
import sys
import os
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import re

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# 导入配置
try:
    from config import PATHS
except ImportError:
    print("Warning: Could not import PATHS from config.py")
    PATHS = {}

from transformer_lens import HookedTransformer
from sae_lens.saes import StandardTrainingSAE, StandardTrainingSAEConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_sae_and_model(sae_path, model_name="Qwen/Qwen2.5-7B-Instruct"):
    """加载SAE和模型"""
    print(f"Loading model: {model_name}...")
    
    # 首先检查 config.py 中的 PATHS
    if model_name in PATHS:
        local_path = PATHS[model_name]
        if os.path.exists(local_path):
            print(f"✓ Using local model from config.py: {local_path}")
            # 用transformers加载本地模型
            hf_model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.float32,
                device_map='cpu'
            )
            tokenizer = AutoTokenizer.from_pretrained(local_path)
            # 转换为HookedTransformer
            print("Converting to HookedTransformer...")
            model = HookedTransformer.from_pretrained(
                model_name,  # 使用官方名称以获取正确配置
                hf_model=hf_model,
                tokenizer=tokenizer,
                device='cpu'
            )
        else:
            print(f"Warning: Path in config.py not found: {local_path}")
            print(f"Falling back to HuggingFace...")
            model = HookedTransformer.from_pretrained(model_name, device='cpu')
    else:
        print(f"Model not in config.py, using HuggingFace cache...")
        model = HookedTransformer.from_pretrained(model_name, device='cpu')
    
    print(f"Loading SAE from: {sae_path}...")
    checkpoint = torch.load(sae_path, map_location='cpu')
    config = checkpoint['config']
    
    # 创建SAE配置
    sae_cfg = StandardTrainingSAEConfig(
        d_in=config['d_model'],
        d_sae=config['d_sae'],
        l1_coefficient=config.get('l1_coefficient', 0.0001),
        dtype="float32",
        device='cpu',
        apply_b_dec_to_input=True,
        normalize_activations="none",
    )
    
    # 创建SAE并加载权重
    sae = StandardTrainingSAE(sae_cfg)
    sae.load_state_dict(checkpoint['sae_state_dict'])
    sae.eval()
    
    return sae, model, config


def get_feature_activations(model, sae, dataset, feature_id, layer, device='cuda'):
    """获取指定特征在所有样本上的激活"""
    model = model.to(device)
    sae = sae.to(device)
    
    activations = []
    
    print(f"\nAnalyzing Feature {feature_id} activations...")
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(dataset)}...")
        
        try:
            # 准备输入
            text = sample.get('prompt', sample.get('input', ''))
            
            # 跳过空文本
            if not text or len(text.strip()) == 0:
                activations.append({
                    'sample': sample,
                    'activation': 0.0,
                    'text': text
                })
                continue
            
            # Tokenize - 添加截断以避免过长
            tokens = model.to_tokens(text, truncate=True, move_to_device=True)
            
            # 检查tokens维度
            if tokens.size(0) == 0 or tokens.size(1) == 0:
                activations.append({
                    'sample': sample,
                    'activation': 0.0,
                    'text': text
                })
                continue
            
            # 获取隐层激活
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, device=device)
                hidden_states = cache[f'blocks.{layer}.hook_resid_post']
                
                # 通过SAE encoder获取特征激活
                hidden_flat = hidden_states.view(-1, hidden_states.size(-1))
                sae_features = sae.encode(hidden_flat)
                
                # 获取指定特征的最大激活（跨所有token）
                feature_act = sae_features[:, feature_id].max().item()
            
            activations.append({
                'sample': sample,
                'activation': feature_act,
                'text': text
            })
            
        except Exception as e:
            print(f"  Warning: Error processing sample {i}: {e}")
            activations.append({
                'sample': sample,
                'activation': 0.0,
                'text': text
            })
            continue
    
    return activations


def extract_spatial_keywords(text):
    """提取空间关系关键词"""
    # 英文空间词
    en_keywords = {
        'left': 'X-left',
        'right': 'X-right',
        'front': 'Y-front',
        'behind': 'Y-behind',
        'back': 'Y-behind',
        'above': 'Z-above',
        'below': 'Z-below',
        'under': 'Z-below',
        'over': 'Z-above',
        'on top': 'Z-above',
        'beneath': 'Z-below',
        'next to': 'X-adjacent',
        'beside': 'X-adjacent',
        'near': 'general',
        'far': 'general',
        'inside': 'containment',
        'outside': 'containment',
    }
    
    # 中文空间词
    cn_keywords = {
        '左': 'X-left',
        '右': 'X-right',
        '左边': 'X-left',
        '右边': 'X-right',
        '前': 'Y-front',
        '后': 'Y-behind',
        '前面': 'Y-front',
        '后面': 'Y-behind',
        '上': 'Z-above',
        '下': 'Z-below',
        '上面': 'Z-above',
        '下面': 'Z-below',
        '旁边': 'X-adjacent',
        '附近': 'general',
        '里面': 'containment',
        '外面': 'containment',
    }
    
    found = []
    text_lower = text.lower()
    
    # 检查英文
    for word, category in en_keywords.items():
        if word in text_lower:
            found.append(category)
    
    # 检查中文
    for word, category in cn_keywords.items():
        if word in text:
            found.append(category)
    
    return list(set(found))


def analyze_top_activating_samples(activations, top_k=20):
    """分析顶级激活样本"""
    # 按激活强度排序
    sorted_acts = sorted(activations, key=lambda x: x['activation'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"TOP {top_k} ACTIVATING SAMPLES")
    print(f"{'='*80}\n")
    
    spatial_keywords_counter = Counter()
    answer_coords = []
    
    for i, item in enumerate(sorted_acts[:top_k]):
        sample = item['sample']
        activation = item['activation']
        text = item['text']
        
        # 提取空间关键词
        keywords = extract_spatial_keywords(text)
        spatial_keywords_counter.update(keywords)
        
        # 提取答案坐标（如果有）
        answer = sample.get('answer', sample.get('output', ''))
        coords = extract_coordinates(answer)
        if coords:
            answer_coords.append(coords)
        
        print(f"#{i+1} | Activation: {activation:.4f}")
        print(f"{'─'*80}")
        print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
        
        if 'relation' in sample:
            print(f"Relation: {sample['relation']}")
        if coords:
            print(f"Answer coordinates: {coords}")
        if keywords:
            print(f"Spatial keywords: {', '.join(keywords)}")
        
        print()
    
    # 统计分析
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Spatial keyword frequency in top {top_k}:")
    for keyword, count in spatial_keywords_counter.most_common():
        percentage = (count / top_k) * 100
        print(f"  {keyword}: {count}/{top_k} ({percentage:.1f}%)")
    
    if answer_coords:
        print(f"\nAnswer coordinate statistics (n={len(answer_coords)}):")
        coords_array = np.array(answer_coords)
        print(f"  X - Mean: {coords_array[:, 0].mean():.2f}, Std: {coords_array[:, 0].std():.2f}")
        print(f"  Y - Mean: {coords_array[:, 1].mean():.2f}, Std: {coords_array[:, 1].std():.2f}")
        print(f"  Z - Mean: {coords_array[:, 2].mean():.2f}, Std: {coords_array[:, 2].std():.2f}")
    
    # 激活强度分析
    all_activations = [item['activation'] for item in sorted_acts]
    top_acts = all_activations[:top_k]
    print(f"\nActivation statistics:")
    print(f"  Top {top_k} mean: {np.mean(top_acts):.4f}")
    print(f"  Top {top_k} std: {np.std(top_acts):.4f}")
    print(f"  Top {top_k} min: {np.min(top_acts):.4f}")
    print(f"  Overall mean: {np.mean(all_activations):.4f}")
    print(f"  Overall max: {np.max(all_activations):.4f}")
    print(f"  Overall nonzero ratio: {(np.array(all_activations) > 0).mean():.2%}")
    
    return {
        'top_samples': sorted_acts[:top_k],
        'spatial_keywords': dict(spatial_keywords_counter),
        'answer_coords': answer_coords,
        'activation_stats': {
            'top_mean': float(np.mean(top_acts)),
            'top_std': float(np.std(top_acts)),
            'overall_mean': float(np.mean(all_activations)),
            'overall_max': float(np.max(all_activations)),
            'nonzero_ratio': float((np.array(all_activations) > 0).mean())
        }
    }


def extract_coordinates(text):
    """从文本中提取坐标"""
    # 匹配 (x, y, z) 或 [x, y, z] 格式
    pattern = r'[\(\[]?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\)\]]?'
    matches = re.findall(pattern, text)
    
    if matches:
        return [float(x) for x in matches[0]]
    return None


def compare_top_vs_bottom_samples(activations, k=10):
    """对比顶级激活和底部激活样本"""
    sorted_acts = sorted(activations, key=lambda x: x['activation'], reverse=True)
    
    top_samples = sorted_acts[:k]
    bottom_samples = sorted_acts[-k:]
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: TOP {k} vs BOTTOM {k}")
    print(f"{'='*80}\n")
    
    # 提取关键词
    top_keywords = []
    bottom_keywords = []
    
    for item in top_samples:
        keywords = extract_spatial_keywords(item['text'])
        top_keywords.extend(keywords)
    
    for item in bottom_samples:
        keywords = extract_spatial_keywords(item['text'])
        bottom_keywords.extend(keywords)
    
    top_counter = Counter(top_keywords)
    bottom_counter = Counter(bottom_keywords)
    
    print("Spatial keywords in TOP samples:")
    for keyword, count in top_counter.most_common(5):
        print(f"  {keyword}: {count}/{k}")
    
    print("\nSpatial keywords in BOTTOM samples:")
    for keyword, count in bottom_counter.most_common(5):
        print(f"  {keyword}: {count}/{k}")
    
    # 差异分析
    all_keywords = set(top_counter.keys()) | set(bottom_counter.keys())
    print("\nKeyword enrichment in TOP vs BOTTOM:")
    enrichments = []
    for keyword in all_keywords:
        top_freq = top_counter.get(keyword, 0) / k
        bottom_freq = bottom_counter.get(keyword, 0) / k
        if bottom_freq > 0:
            enrichment = top_freq / bottom_freq
        else:
            enrichment = float('inf') if top_freq > 0 else 1.0
        enrichments.append((keyword, top_freq, bottom_freq, enrichment))
    
    enrichments.sort(key=lambda x: x[3], reverse=True)
    for keyword, top_f, bottom_f, enrich in enrichments[:5]:
        print(f"  {keyword}: {enrich:.2f}x (top={top_f:.2f}, bottom={bottom_f:.2f})")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze semantic interpretability of SAE features')
    parser.add_argument('--sae_path', type=str, required=True, help='Path to SAE checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--feature_ids', type=str, required=True, help='Comma-separated feature IDs to analyze')
    parser.add_argument('--layer', type=int, default=20, help='Layer number')
    parser.add_argument('--top_k', type=int, default=20, help='Number of top samples to display')
    parser.add_argument('--output_dir', type=str, default='./feature_semantic_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='Model name (will use config.py PATHS if available)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 解析特征ID
    feature_ids = [int(fid.strip()) for fid in args.feature_ids.split(',')]
    
    # 加载模型和SAE
    sae, model, config = load_sae_and_model(
        args.sae_path, 
        model_name=args.model_name
    )
    
    # 加载数据
    print(f"\nLoading dataset from: {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} samples")
    
    # 限制样本数（如果太多）
    if len(dataset) > 500:
        print(f"Using first 500 samples for analysis...")
        dataset = dataset[:500]
    
    # 分析每个特征
    all_results = {}
    
    for feature_id in feature_ids:
        print(f"\n{'#'*80}")
        print(f"# ANALYZING FEATURE {feature_id}")
        print(f"{'#'*80}")
        
        # 获取激活
        activations = get_feature_activations(
            model, sae, dataset, feature_id, args.layer, args.device
        )
        
        # 分析顶级样本
        results = analyze_top_activating_samples(activations, args.top_k)
        
        # 对比分析
        compare_top_vs_bottom_samples(activations, k=min(10, len(activations)//10))
        
        # 保存结果
        all_results[str(feature_id)] = results
        
        # 保存详细结果到文件
        output_file = output_dir / f"feature_{feature_id}_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    # 保存总结
    summary_file = output_dir / "analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

