"""
Activation Patching for Orientation Reasoning (CN) — 按组件拆分版本
========================================================================
在 v2 基础上，不仅 patch layer output (resid_post)，还分别 patch：
  1. residual stream (resid_pre) — 进入该层时的残差流
  2. attention output (attn_out) — 该层 attention 的输出
  3. MLP output (mlp_out)     — 该层 MLP 的输出
  4. layer output (resid_post) — 该层整层输出（与原 v2 一致）

这样可以看出：空间信息主要来自 attention 还是 MLP。

Metric 与 v2 相同：归一化恢复率
  recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import PATHS

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# 四种 patch 类型：hook 名称与说明
PATCH_TYPES = [
    ("resid_pre", "blocks.{}.hook_resid_pre", "residual stream（进入该层时的残差流）"),
    ("attn_out", "blocks.{}.hook_attn_out", "attention output"),
    ("mlp_out", "blocks.{}.hook_mlp_out", "MLP output"),
    ("resid_post", "blocks.{}.hook_resid_post", "layer output（整层输出）"),
]

# =========================
# 数据路径与配置
# =========================
def replace_question_in_full_prompt(full_prompt: str, new_question: str) -> str:
    """将完整 prompt 中的「初始方向+动作+问题」块替换为 new_question。"""
    block = new_question.replace(
        "\n\n你现在面向哪个方向？", "\n\n\n问题：\n你现在面向哪个方向？"
    )
    start_marker = "初始方向和动作：\n"
    option_marker = "\n\n选项："
    if start_marker not in full_prompt or option_marker not in full_prompt:
        return full_prompt
    i0 = full_prompt.index(start_marker) + len(start_marker)
    i1 = full_prompt.index(option_marker, i0)
    return full_prompt[:i0] + block + full_prompt[i1:]


def load_clean_by_id(clean_path: Path) -> Dict[int, dict]:
    """加载 clean JSON，返回 id -> sample（含 prompt, options, correct_option）。"""
    with open(clean_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return {int(item["id"]): item for item in data}


def load_corrupted_pairs(
    corrupted_path: Path,
    clean_by_id: Dict[int, dict],
    max_samples: Optional[int] = None,
) -> List[dict]:
    """
    加载 corrupted JSONL，与 clean 对齐，生成 (clean_full_prompt, corrupted_full_prompt, correct_letter, wrong_letter)。
    """
    pairs = []
    buffer = ""
    with open(corrupted_path, "r", encoding="utf-8") as f:
        for line in f:
            buffer += line
            try:
                row = json.loads(buffer)
                buffer = ""
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            original_id = row.get("original_id")
            if original_id is None:
                continue
            clean_sample = clean_by_id.get(int(original_id))
            if not clean_sample:
                continue
            full_prompt = clean_sample.get("prompt")
            if not full_prompt:
                continue
            clean_full = full_prompt
            corrupted_full = replace_question_in_full_prompt(
                full_prompt, row["corrupted_prompt"]
            )
            correct_letter = clean_sample.get("correct_option", "A")
            options = clean_sample.get("options", [])
            corrupted_short = row.get("corrupted_answer_short") or row.get("corrupted_answer", "")
            if isinstance(corrupted_short, str) and "方" in corrupted_short:
                corrupted_short = corrupted_short.replace("方", "").strip() or corrupted_short
            wrong_letter = None
            if options and corrupted_short:
                try:
                    idx = options.index(corrupted_short)
                    wrong_letter = "ABCD"[idx]
                except (ValueError, IndexError):
                    pass
            if wrong_letter is None:
                wrong_letter = "ABCD"[0] if correct_letter != "A" else "B"
            pairs.append({
                "clean_full": clean_full,
                "corrupted_full": corrupted_full,
                "correct_letter": correct_letter,
                "wrong_letter": wrong_letter,
                "original_id": original_id,
                "pair_id": row.get("id", len(pairs)),
            })
            if max_samples and len(pairs) >= max_samples:
                break
    return pairs


def get_option_token_ids(tokenizer, option_letters: List[str]) -> Dict[str, int]:
    """获取选项字母对应 token id（单 token）。"""
    ids = {}
    for letter in option_letters:
        tok = tokenizer.encode(letter, add_special_tokens=False)
        if len(tok) == 1:
            ids[letter] = tok[0]
        else:
            ids[letter] = tok[0] if tok else 0
    return ids


def logit_diff_metric(
    logits: torch.Tensor,
    correct_idx: int,
    wrong_idx: int,
    position: int = -1,
) -> float:
    """LD = logit(correct answer) - logit(wrong answer)。"""
    logits_at = logits[0, position, :].float()
    return (logits_at[correct_idx] - logits_at[wrong_idx]).item()


def normalized_recovery(
    LD_patched: float,
    LD_corrupt: float,
    LD_clean: float,
    eps: float = 1e-10,
) -> float:
    """归一化恢复率：recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)。"""
    denom = LD_clean - LD_corrupt
    if abs(denom) < eps:
        return np.nan
    return (LD_patched - LD_corrupt) / denom


def make_patch_hook(clean_act: torch.Tensor):
    """构造 patch hook：用 clean 激活替换 corrupt，若 shape 不一致则只 patch 最后一位置。"""
    def hook(act, hook):
        if act.shape == clean_act.shape:
            return clean_act
        out = act.clone()
        out[:, -1, :] = clean_act[:, -1, :]
        return out
    return hook


def main():
    parser = argparse.ArgumentParser(
        description="Activation Patching (Orientation CN): resid_pre / attn_out / mlp_out / resid_post"
    )
    parser.add_argument("--model_name", "-m", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--clean_data",
        type=str,
        default="src/data_generation/task_family_2/data_orientation/orientation_reasoning_dataset_CN_with_prompt.json",
    )
    parser.add_argument(
        "--corrupted_data",
        type=str,
        default="src/data_generation/task_family_2/data_orientation_corrupted/orientation_reasoning_dataset_CN_corrupted.jsonl",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="src/activation_patching/task_family_2/out_orientation_cn_v2_by_components",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    clean_path = project_root / args.clean_data
    corrupted_path = project_root / args.corrupted_data
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # -------------------------
    # 加载模型
    # -------------------------
    model_name = args.model_name
    model_path = PATHS.get(model_name, model_name)
    print("Loading model:", model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    hf_model = hf_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    model.eval()
    n_layers = model.cfg.n_layers
    print(f"Model: {n_layers} layers")
    print("Patch types: resid_pre (residual stream), attn_out, mlp_out, resid_post (layer output)")

    # -------------------------
    # 加载数据
    # -------------------------
    clean_by_id = load_clean_by_id(clean_path)
    pairs = load_corrupted_pairs(corrupted_path, clean_by_id, args.max_samples)
    print(f"Loaded {len(pairs)} clean/corrupted pairs")

    option_ids = get_option_token_ids(tokenizer, ["A", "B", "C", "D"])

    # 按 (patch_type, layer) 收集 recovery
    # layer_recovery_lists[pt][layer] = list of recovery values
    patch_type_keys = [pt[0] for pt in PATCH_TYPES]
    layer_recovery_lists: Dict[str, List[List[float]]] = {
        pt: [[] for _ in range(n_layers)] for pt in patch_type_keys
    }
    per_sample_results = []

    for pair_idx, pair in enumerate(tqdm(pairs, desc="Activation Patching (by component)")):
        clean_full = pair["clean_full"]
        corrupted_full = pair["corrupted_full"]
        correct_letter = pair["correct_letter"]
        wrong_letter = pair["wrong_letter"]
        correct_tok = option_ids.get(correct_letter, 0)
        wrong_tok = option_ids.get(wrong_letter, 0)

        tokens_clean = model.to_tokens(clean_full, truncate=True)
        tokens_corrupt = model.to_tokens(corrupted_full, truncate=True)

        # Step 1: clean run → cache + LD_clean
        with torch.no_grad():
            logits_clean, cache_clean = model.run_with_cache(
                tokens_clean, return_type="logits"
            )
        seq_len_clean = logits_clean.shape[1]
        LD_clean = logit_diff_metric(
            logits_clean, correct_tok, wrong_tok, position=seq_len_clean - 1
        )

        # Step 2: corrupted run → cache + LD_corrupt
        with torch.no_grad():
            logits_corrupt, cache_corrupt = model.run_with_cache(
                tokens_corrupt, return_type="logits"
            )
        seq_len = logits_corrupt.shape[1]
        LD_corrupt = logit_diff_metric(
            logits_corrupt, correct_tok, wrong_tok, position=seq_len - 1
        )

        # Step 3 & 4: 对每种 patch 类型、每一层做 patching，算 recovery
        sample_recovery_by_type: Dict[str, List[float]] = {
            pt: [np.nan] * n_layers for pt in patch_type_keys
        }

        for patch_key, hook_fmt, _ in PATCH_TYPES:
            for layer in range(n_layers):
                hook_name = hook_fmt.format(layer)
                try:
                    clean_act = cache_clean[hook_name]
                except KeyError:
                    continue
                hook_fn = make_patch_hook(clean_act)
                with torch.no_grad():
                    logits_patch = model.run_with_hooks(
                        tokens_corrupt,
                        fwd_hooks=[(hook_name, hook_fn)],
                        return_type="logits",
                    )
                LD_patched = logit_diff_metric(
                    logits_patch, correct_tok, wrong_tok, position=seq_len - 1
                )
                rec = normalized_recovery(LD_patched, LD_corrupt, LD_clean)
                sample_recovery_by_type[patch_key][layer] = rec
                if not np.isnan(rec):
                    layer_recovery_lists[patch_key][layer].append(rec)

        per_sample_results.append({
            "pair_id": pair.get("pair_id"),
            "original_id": pair.get("original_id"),
            "LD_clean": LD_clean,
            "LD_corrupt": LD_corrupt,
            **{
                f"recovery_per_layer_{pt}": [
                    float(x) if not np.isnan(x) else None
                    for x in sample_recovery_by_type[pt]
                ]
                for pt in patch_type_keys
            },
        })

    # 每种 patch 类型：每层平均 recovery、best layer
    layer_recovery_scores_by_type: Dict[str, List[float]] = {}
    best_layer_by_type: Dict[str, int] = {}

    for patch_key in patch_type_keys:
        lists = layer_recovery_lists[patch_key]
        scores = np.array([
            np.mean(v) if len(v) > 0 else np.nan for v in lists
        ])
        layer_recovery_scores_by_type[patch_key] = [
            float(x) if not np.isnan(x) else None for x in scores
        ]
        valid = ~np.isnan(scores)
        best_layer_by_type[patch_key] = int(
            np.argmax(np.where(valid, scores, -np.inf))
        )

    # -------------------------
    # 保存结果
    # -------------------------
    summary = {
        "n_layers": n_layers,
        "n_samples": len(pairs),
        "metric_note": "recovery = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt); 0=no recovery, 1=full",
        "patch_types": {
            "resid_pre": "residual stream（进入该层时的残差流）",
            "attn_out": "attention output",
            "mlp_out": "MLP output",
            "resid_post": "layer output（整层输出）",
        },
        "layer_recovery_score_mean": {
            pt: layer_recovery_scores_by_type[pt] for pt in patch_type_keys
        },
        "best_layer": best_layer_by_type,
        "model_name": model_name,
    }
    out_summary = output_dir / "activation_patching_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {out_summary}")

    for pt in patch_type_keys:
        scores = layer_recovery_scores_by_type[pt]
        best = best_layer_by_type[pt]
        print(f"  [{pt}] recovery (mean): {[round(x, 4) if x is not None else 'nan' for x in scores]}")
        print(f"  [{pt}] best_layer: {best}")

    out_detail = output_dir / "activation_patching_per_sample.jsonl"
    with open(out_detail, "w", encoding="utf-8") as f:
        for r in per_sample_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Per-sample results saved to {out_detail}")


if __name__ == "__main__":
    main()
