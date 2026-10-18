#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تحليل نتائج اختبار الاستدلال الاتجاهي (النسخة العربية)
"""

import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List

# تكوين matplotlib لدعم اللغة العربية
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']


def analyze_results(results_file: str, dataset_file: str):
    """
    تحليل نتائج الاختبار
    
    Args:
        results_file: مسار ملف النتائج
        dataset_file: مسار ملف مجموعة البيانات الأصلية
    """
    # قراءة النتائج
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # قراءة مجموعة البيانات الأصلية
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # إنشاء خريطة من المعرف إلى عدد الخطوات
    id_to_num_steps = {}
    for item in dataset:
        id_to_num_steps[item['id']] = item['num_steps']
    
    # تجميع الإحصائيات حسب عدد الخطوات
    steps_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    total_success = 0
    total_correct = 0
    
    for result in results:
        if result['success']:
            total_success += 1
            item_id = result['id']
            num_steps = id_to_num_steps[item_id]
            
            is_correct = result['predicted_option'] == result['correct_option']
            
            # الإحصاء حسب num_steps
            steps_stats[num_steps]['total'] += 1
            if is_correct:
                steps_stats[num_steps]['correct'] += 1
                total_correct += 1
    
    # طباعة نتائج الإحصاء
    print("="*60)
    print("تحليل نتائج اختبار الاستدلال الاتجاهي")
    print("="*60)
    print(f"\nإجمالي العينات: {total_success}")
    print(f"عدد التنبؤات الصحيحة: {total_correct}")
    print(f"الدقة الإجمالية: {total_correct / total_success * 100:.2f}%")
    
    # التحليل حسب عدد الخطوات
    print("\n" + "="*60)
    print("الدقة حسب عدد الخطوات:")
    print("="*60)
    
    for num_steps in sorted(steps_stats.keys()):
        stats = steps_stats[num_steps]
        accuracy = stats['correct'] / stats['total'] * 100
        print(f"  {num_steps:2d} خطوة: {accuracy:6.2f}% "
              f"({stats['correct']:3d}/{stats['total']:3d})")
    
    # توليد تحليل الاتجاه المرئي
    print("\n" + "="*60)
    print("تحليل الاتجاه:")
    print("="*60)
    print("\nتغيير الدقة مع زيادة الخطوات:")
    for num_steps in sorted(steps_stats.keys()):
        stats = steps_stats[num_steps]
        accuracy = stats['correct'] / stats['total'] * 100
        bar_length = int(accuracy / 2)  # أقصى 50 حرفًا
        bar = '█' * bar_length
        print(f"{num_steps:>2} خطوة: [{bar:<50}] {accuracy:>6.2f}% ({stats['correct']}/{stats['total']})")
    
    # حفظ نتائج التحليل في ملف JSON
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
    
    analysis_output = os.path.join(os.path.dirname(results_file), 'accuracy_analysis_AR_test.json')
    with open(analysis_output, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nتم حفظ بيانات التحليل التفصيلية في: {analysis_output}")
    
    # رسم المخططات
    plot_accuracy_by_steps(steps_stats)


def plot_accuracy_by_steps(steps_stats: Dict):
    """
    رسم مخطط الدقة حسب عدد الخطوات
    
    Args:
        steps_stats: قاموس إحصاءات الخطوات
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
    
    # إضافة تسميات رقمية
    for step, acc in zip(steps, accuracies):
        plt.text(step, acc + 0.02, f'{acc:.1%}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('orientation_accuracy_by_steps_AR.png', dpi=300)
    print("\n" + "="*60)
    print("تم حفظ المخطط في: orientation_accuracy_by_steps_AR.png")
    print("="*60)


def main():
    """الدالة الرئيسية"""
    # مسارات الملفات المدمجة
    results_file = os.path.join(os.path.dirname(__file__), './data_orientation/api_results_AR_test.json')
    dataset_file = os.path.join(os.path.dirname(__file__), './data_orientation/orientation_reasoning_dataset_AR_test_with_prompt.json')
    
    # التحقق من وجود الملفات
    if not os.path.exists(results_file):
        print(f"خطأ: لم يتم العثور على ملف النتائج في {results_file}")
        return
    
    if not os.path.exists(dataset_file):
        print(f"خطأ: لم يتم العثور على ملف مجموعة البيانات في {dataset_file}")
        return
    
    print(f"تحليل النتائج من: {results_file}")
    print(f"استخدام مجموعة البيانات: {dataset_file}\n")
    
    # تحليل نتائج الاستدلال الاتجاهي
    analyze_results(results_file, dataset_file)


if __name__ == "__main__":
    main()


