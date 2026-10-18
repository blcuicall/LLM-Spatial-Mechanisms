import json
import os
from collections import defaultdict

def analyze_accuracy_by_steps():
    """分析正确率与步骤数量的关系"""
    
    # 读取结果文件
    results_path = os.path.join(os.path.dirname(__file__), './data_procedure/api_results_AR_test.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 读取原始数据文件
    data_path = os.path.join(os.path.dirname(__file__), './data_procedure/spatial_procedure_dataset_AR_test_with_prompt.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 创建id到步骤数量的映射
    id_to_num_steps = {}
    for item in dataset:
        id_to_num_steps[item['id']] = item['num_steps']
    
    # 按步骤数量分组统计
    steps_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for result in results:
        if result['success']:
            item_id = result['id']
            num_steps = id_to_num_steps[item_id]
            
            is_correct = result['predicted_option'] == result['correct_option']
            
            # 按num_steps统计
            steps_stats[num_steps]['total'] += 1
            if is_correct:
                steps_stats[num_steps]['correct'] += 1
    
    # 打印按num_steps的统计结果
    print("=" * 70)
    print("正确率与Steps数量的关系")
    print("=" * 70)
    print(f"{'Steps数量':<15} {'总数':<10} {'正确数':<10} {'正确率':<10}")
    print("-" * 70)
    
    total_all = 0
    correct_all = 0
    
    for num_steps in sorted(steps_stats.keys()):
        stats = steps_stats[num_steps]
        accuracy = stats['correct'] / stats['total'] * 100
        total_all += stats['total']
        correct_all += stats['correct']
        print(f"{num_steps:<15} {stats['total']:<10} {stats['correct']:<10} {accuracy:>6.2f}%")
    
    print("-" * 70)
    overall_accuracy = correct_all / total_all * 100
    print(f"{'总计':<15} {total_all:<10} {correct_all:<10} {overall_accuracy:>6.2f}%")
    print("=" * 70)
    
    # 生成详细的分析数据
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
            'total': total_all,
            'correct': correct_all,
            'accuracy': overall_accuracy
        }
    }
    
    # 保存分析结果
    analysis_path = os.path.join(os.path.dirname(__file__), './data_procedure/accuracy_analysis_AR_test.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细分析数据已保存到: {analysis_path}")
    
    # 生成可视化的趋势分析
    print("\n" + "=" * 70)
    print("趋势分析")
    print("=" * 70)
    
    # 按steps数量分析趋势
    steps_sorted = sorted(steps_stats.items())
    print("\nSteps数量增加时的正确率变化:")
    for i, (num_steps, stats) in enumerate(steps_sorted):
        accuracy = stats['correct'] / stats['total'] * 100
        bar_length = int(accuracy / 2)  # 最大50个字符
        bar = '█' * bar_length
        print(f"{num_steps:>2} steps: [{bar:<50}] {accuracy:>6.2f}% ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    analyze_accuracy_by_steps()




