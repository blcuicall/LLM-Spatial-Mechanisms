#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对中文方向推理数据集进行损坏生成。

读取 orientation_reasoning_dataset_CN_with_prompt.json，
对每条样本的 question 做干净/损坏对生成（改起始方向、单动作替换等），
将结果写入 data_orientation_corrupted/ 目录。
"""

from __future__ import annotations

import argparse
import json
import re
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

# ----------------------------
# 领域定义（与 generate_orientation 损坏逻辑一致）
# ----------------------------

DIRECTIONS = ["北方", "东方", "南方", "西方"]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTIONS)}
IDX_TO_DIR = {i: d for i, d in enumerate(DIRECTIONS)}

ACTIONS = {"向右转", "向左转", "转身"}
ACTION_PATTERN = re.compile(r"^(向右转|向左转|转身)[。．]?$")
START_PATTERN = re.compile(r"^你面向(北方|东方|南方|西方)[。．]?$")
QUESTION_PATTERN = re.compile(r"^你现在面向哪个方向\？?$")


@dataclass
class ParsedOrientationTask:
    start_direction: str
    actions: List[str]
    question: str


@dataclass
class PromptPair:
    id: str
    clean_prompt: str
    clean_answer: str
    corrupted_prompt: str
    corrupted_answer: str
    corruption_type: str
    meta: dict


def normalize_lines(text: str) -> List[str]:
    """将 prompt 按行切分，去除空白与空行。"""
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def parse_orientation_prompt(text: str) -> ParsedOrientationTask:
    lines = normalize_lines(text)
    if len(lines) < 2:
        raise ValueError("Prompt 过短，至少需要起始行和问题行。")

    start_match = START_PATTERN.match(lines[0])
    if not start_match:
        raise ValueError(f"首行应为「你面向X方。」，得到: {lines[0]!r}")
    start_direction = start_match.group(1)

    question = lines[-1]
    if "你现在面向哪个方向" not in question:
        raise ValueError(f"末行应为询问当前朝向，得到: {question!r}")

    actions = []
    for line in lines[1:-1]:
        m = ACTION_PATTERN.match(line)
        if not m:
            raise ValueError(f"非法动作行: {line!r}")
        actions.append(m.group(1))

    if not actions:
        raise ValueError("未找到动作行。")

    return ParsedOrientationTask(
        start_direction=start_direction,
        actions=actions,
        question="你现在面向哪个方向？",
    )


def format_orientation_prompt(task: ParsedOrientationTask) -> str:
    parts = [f"你面向{task.start_direction}。"]
    parts.extend(f"{a}。" for a in task.actions)
    parts.append("")
    parts.append(task.question)
    return "\n".join(parts)


def apply_action(direction: str, action: str) -> str:
    idx = DIR_TO_IDX[direction]
    if action == "向右转":
        idx = (idx + 1) % 4
    elif action == "向左转":
        idx = (idx - 1) % 4
    elif action == "转身":
        idx = (idx + 2) % 4
    else:
        raise ValueError(f"未知动作: {action}")
    return IDX_TO_DIR[idx]


def solve_orientation(task: ParsedOrientationTask) -> str:
    cur = task.start_direction
    for action in task.actions:
        cur = apply_action(cur, action)
    return cur


def generate_start_direction_corruptions(
    clean_task: ParsedOrientationTask,
) -> List[Tuple[ParsedOrientationTask, str, dict]]:
    """仅修改起始方向。返回 (corrupted_task, corruption_type, meta)。"""
    out = []
    for d in DIRECTIONS:
        if d == clean_task.start_direction:
            continue
        corrupt = ParsedOrientationTask(
            start_direction=d,
            actions=list(clean_task.actions),
            question=clean_task.question,
        )
        out.append(
            (
                corrupt,
                "change_start_direction",
                {"from_start": clean_task.start_direction, "to_start": d},
            )
        )
    return out


def generate_single_action_corruptions(
    clean_task: ParsedOrientationTask,
) -> List[Tuple[ParsedOrientationTask, str, dict]]:
    """仅修改一个动作：向右转<->向左转，转身可改为向右转或向左转。"""
    out = []
    for i, action in enumerate(clean_task.actions):
        if action == "向右转":
            candidates = ["向左转", "转身"]
        elif action == "向左转":
            candidates = ["向右转", "转身"]
        elif action == "转身":
            candidates = ["向右转", "向左转"]
        else:
            continue
        for new_action in candidates:
            new_actions = list(clean_task.actions)
            new_actions[i] = new_action
            corrupt = ParsedOrientationTask(
                start_direction=clean_task.start_direction,
                actions=new_actions,
                question=clean_task.question,
            )
            out.append(
                (
                    corrupt,
                    "change_single_action",
                    {
                        "action_index": i,
                        "from_action": action,
                        "to_action": new_action,
                    },
                )
            )
    return out


def generate_swap_action_corruptions(
    clean_task: ParsedOrientationTask,
) -> List[Tuple[ParsedOrientationTask, str, dict]]:
    """交换相邻且不同的两个动作。"""
    out = []
    acts = clean_task.actions
    for i in range(len(acts) - 1):
        if acts[i] == acts[i + 1]:
            continue
        new_actions = list(acts)
        new_actions[i], new_actions[i + 1] = new_actions[i + 1], new_actions[i]
        corrupt = ParsedOrientationTask(
            start_direction=clean_task.start_direction,
            actions=new_actions,
            question=clean_task.question,
        )
        out.append(
            (
                corrupt,
                "swap_adjacent_actions",
                {
                    "swap_indices": [i, i + 1],
                    "before": [acts[i], acts[i + 1]],
                    "after": [new_actions[i], new_actions[i + 1]],
                },
            )
        )
    return out


def deduplicate_pairs(
    pairs: List[Tuple[ParsedOrientationTask, str, dict]]
) -> List[Tuple[ParsedOrientationTask, str, dict]]:
    seen = set()
    out = []
    for task, ctype, meta in pairs:
        key = (task.start_direction, tuple(task.actions), task.question)
        if key not in seen:
            seen.add(key)
            out.append((task, ctype, meta))
    return out


def build_corrupted_pairs(
    clean_prompt: str,
    include_action_corruptions: bool = True,
    include_swap_corruptions: bool = False,
    require_different_answer: bool = True,
    max_pairs: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
) -> List[PromptPair]:
    clean_task = parse_orientation_prompt(clean_prompt)
    clean_answer = solve_orientation(clean_task)

    candidates: List[Tuple[ParsedOrientationTask, str, dict]] = []
    candidates.extend(generate_start_direction_corruptions(clean_task))
    if include_action_corruptions:
        candidates.extend(generate_single_action_corruptions(clean_task))
    if include_swap_corruptions:
        candidates.extend(generate_swap_action_corruptions(clean_task))
    candidates = deduplicate_pairs(candidates)

    pairs: List[PromptPair] = []
    for idx, (corrupt_task, ctype, meta) in enumerate(candidates):
        corrupted_answer = solve_orientation(corrupt_task)
        if require_different_answer and corrupted_answer == clean_answer:
            continue
        pairs.append(
            PromptPair(
                id=f"pair_{idx:04d}",
                clean_prompt=format_orientation_prompt(clean_task),
                clean_answer=clean_answer,
                corrupted_prompt=format_orientation_prompt(corrupt_task),
                corrupted_answer=corrupted_answer,
                corruption_type=ctype,
                meta=meta,
            )
        )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def replace_question_in_full_prompt(full_prompt: str, new_question: str) -> str:
    """
    将完整 prompt 中的「初始方向+动作+问题」块替换为新的 question 内容。
    原格式: ...初始方向和动作：\n你面向X。\n...\n\n\n问题：\n你现在面向哪个方向？\n\n选项：...
    new_question 格式: 你面向X。\n...\n\n你现在面向哪个方向？
    """
    # 将 new_question 中的「\n\n你现在」改为「\n\n\n问题：\n你现在」以匹配原 prompt 结构
    block = new_question.replace(
        "\n\n你现在面向哪个方向？", "\n\n\n问题：\n你现在面向哪个方向？"
    )
    # 定位：从「你面向」到「你现在面向哪个方向？」的整段（含其后的 \n 到选项前）
    start_marker = "初始方向和动作：\n"
    option_marker = "\n\n选项："
    if start_marker not in full_prompt or option_marker not in full_prompt:
        return full_prompt
    i0 = full_prompt.index(start_marker) + len(start_marker)
    i1 = full_prompt.index(option_marker, i0)
    old_block = full_prompt[i0:i1]
    # 用 block 替换（block 已含 问题：）
    return full_prompt[:i0] + block + full_prompt[i1:]


def load_dataset(path: Path) -> List[dict]:
    """加载 JSON 数组格式的数据集。"""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        data = [data]
    return data


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, indent=4) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对中文方向推理数据集进行损坏生成，结果写入 data_orientation_corrupted。"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data_orientation/orientation_reasoning_dataset_CN_with_prompt.json",
        help="输入 JSON 数据集路径（默认: data_orientation/orientation_reasoning_dataset_CN_with_prompt.json）",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="orientation_reasoning_dataset_CN_corrupted.jsonl",
        help="输出文件名（默认: orientation_reasoning_dataset_CN_corrupted）",
    )
    parser.add_argument(
        "--max_pairs_per_prompt",
        type=int,
        default=8,
        help="每个干净 prompt 最多保留的损坏对数量",
    )
    parser.add_argument(
        "--no_action_corruptions",
        action="store_true",
        help="不生成单动作损坏，仅生成起始方向损坏",
    )
    parser.add_argument(
        "--include_swap_corruptions",
        action="store_true",
        help="同时生成相邻动作交换类损坏",
    )
    parser.add_argument(
        "--allow_same_answer",
        action="store_true",
        help="允许损坏后答案与干净答案相同（默认会过滤掉）",
    )
    parser.add_argument(
        "--with_full_prompt",
        action="store_true",
        help="在输出中附带基于原 prompt 的完整版 clean/corrupted prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多处理的样本数（默认处理全部）",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / args.input
    output_dir = base_dir / "data_orientation_corrupted"
    output_name = args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    dataset = load_dataset(input_path)
    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]

    all_rows: List[dict] = []
    error_count = 0
    for item in dataset:
        original_id = item.get("id", len(all_rows))
        clean_prompt = item.get("question")
        full_prompt = item.get("prompt")
        if not clean_prompt:
            all_rows.append(
                {
                    "original_id": original_id,
                    "error": "missing 'question' field",
                    "item": item,
                }
            )
            error_count += 1
            continue
        try:
            pairs = build_corrupted_pairs(
                clean_prompt=clean_prompt,
                include_action_corruptions=not args.no_action_corruptions,
                include_swap_corruptions=args.include_swap_corruptions,
                require_different_answer=not args.allow_same_answer,
                max_pairs=args.max_pairs_per_prompt,
                shuffle=True,
                seed=args.seed + original_id,
            )
        except Exception as e:
            all_rows.append(
                {
                    "original_id": original_id,
                    "error": str(e),
                    "clean_prompt": clean_prompt,
                }
            )
            error_count += 1
            continue

        for i, pair in enumerate(pairs):
            row = asdict(pair)
            row["original_id"] = original_id
            row["pair_id"] = i + 1
            if args.with_full_prompt and full_prompt:
                row["clean_prompt_full"] = full_prompt
                row["corrupted_prompt_full"] = replace_question_in_full_prompt(
                    full_prompt, pair.corrupted_prompt
                )
            # 与原始数据集答案格式一致：增加短答案（北/东/南/西）
            dir_short = {"北方": "北", "东方": "东", "南方": "南", "西方": "西"}
            row["clean_answer_short"] = dir_short.get(
                row["clean_answer"], row["clean_answer"]
            )
            row["corrupted_answer_short"] = dir_short.get(
                row["corrupted_answer"], row["corrupted_answer"]
            )
            all_rows.append(row)

    out_jsonl = output_dir / output_name
    write_jsonl(out_jsonl, all_rows)

    # 简单统计
    success_rows = [r for r in all_rows if "error" not in r]
    meta = {
        "input_file": str(input_path),
        "total_original_samples": len(dataset),
        "total_output_rows": len(all_rows),
        "successful_pairs": len(success_rows),
        "error_count": error_count,
    }
    print(meta)

if __name__ == "__main__":
    main()
