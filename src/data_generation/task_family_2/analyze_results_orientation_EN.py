#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析方向推理测试结果
"""

import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List


def analyze_results(results_file: str, dataset_file: str):
    """
    分析测试结果
    
    Args:
        results_file: 结果文件路径
        dataset_file: 原始数据集文件路径
    """
    # 读取结果
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 读取原始数据集
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 创建id到步骤数量的映射
    id_to_num_steps = {}
    for item in dataset:
        id_to_num_steps[item['id']] = item['num_steps']
    
    # 按步骤数量分组统计
    steps_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    total_success = 0
    total_correct = 0
    
    for result in results:
        if result['success']:
            total_success += 1
            item_id = result['id']
            num_steps = id_to_num_steps[item_id]
            
            is_correct = result['predicted_option'] == result['correct_option']
            
            # 按num_steps统计
            steps_stats[num_steps]['total'] += 1
            if is_correct:
                steps_stats[num_steps]['correct'] += 1
                total_correct += 1
    
    # 打印统计结果
    print("="*60)
    print("Orientation Reasoning Test Results Analysis")
    print("="*60)
    print(f"\nTotal samples: {total_success}")
    print(f"Correct predictions: {total_correct}")
    print(f"Overall accuracy: {total_correct / total_success * 100:.2f}%")
    
    # 按步骤数分析
    print("\n" + "="*60)
    print("Accuracy by Number of Steps:")
    print("="*60)
    
    for num_steps in sorted(steps_stats.keys()):
        stats = steps_stats[num_steps]
        accuracy = stats['correct'] / stats['total'] * 100
        print(f"  {num_steps:2d} steps: {accuracy:6.2f}% "
              f"({stats['correct']:3d}/{stats['total']:3d})")
    
    # 生成趋势分析可视化
    print("\n" + "="*60)
    print("Trend Analysis:")
    print("="*60)
    print("\nAccuracy change with increasing steps:")
    for num_steps in sorted(steps_stats.keys()):
        stats = steps_stats[num_steps]
        accuracy = stats['correct'] / stats['total'] * 100
        bar_length = int(accuracy / 2)  # 最大50个字符
        bar = '█' * bar_length
        print(f"{num_steps:>2} steps: [{bar:<50}] {accuracy:>6.2f}% ({stats['correct']}/{stats['total']})")
    
    # 保存分析结果到JSON文件
    analysis_data = {
        'by_steps': {
            str(k): {
                'total': v['total'],
                'correct': v['correct'],
                'accuracy': v['correct'] / v['total'] * 100
            }
            for k, v in steps_stats.items()
        },
        'overall': {
            'total': total_success,
            'correct': total_correct,
            'accuracy': total_correct / total_success * 100
        }
    }
    
    analysis_output = os.path.join(os.path.dirname(results_file), 'accuracy_analysis_EN_test.json')
    with open(analysis_output, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed analysis data saved to: {analysis_output}")
    
    # 绘制图表
    plot_accuracy_by_steps(steps_stats)


def plot_accuracy_by_steps(steps_stats: Dict):
    """
    绘制按步骤数的准确率图表
    
    Args:
        steps_stats: 步骤统计字典
    """
    steps = sorted(steps_stats.keys())
    accuracies = [steps_stats[s]['correct'] / steps_stats[s]['total'] for s in steps]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Orientation Reasoning Accuracy by Number of Steps', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    # 添加数值标签
    for step, acc in zip(steps, accuracies):
        plt.text(step, acc + 0.02, f'{acc:.1%}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('orientation_accuracy_by_steps_EN.png', dpi=300)
    print("\n" + "="*60)
    print("Plot saved to: orientation_accuracy_by_steps_EN.png")
    print("="*60)


def main():
    """主函数"""
    # 内置文件路径
    results_file = os.path.join(os.path.dirname(__file__), './data_orientation/api_results_EN_test.json')
    dataset_file = os.path.join(os.path.dirname(__file__), './data_orientation/orientation_reasoning_dataset_EN_test_with_prompt.json')
    
    # 检查文件是否存在
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found at {dataset_file}")
        return
    
    print(f"Analyzing results from: {results_file}")
    print(f"Using dataset: {dataset_file}\n")
    
    # 分析方向推理结果
    analyze_results(results_file, dataset_file)


if __name__ == "__main__":
    main()

