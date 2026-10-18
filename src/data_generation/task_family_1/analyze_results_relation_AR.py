import json
import os
from collections import defaultdict

def analyze_accuracy_by_entities():
    """分析正确率与entities数量的关系"""
    
    # 读取结果文件
    results_path = os.path.join(os.path.dirname(__file__), './data_relation/api_results_AR_test.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 读取原始数据文件
    data_path = os.path.join(os.path.dirname(__file__), './data_relation/spatial_reasoning_dataset_AR_test_with_prompt.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 创建id到entities数量的映射
    id_to_entities_count = {}
    id_to_num_steps = {}
    for item in dataset:
        id_to_entities_count[item['id']] = len(item['entities'])
        id_to_num_steps[item['id']] = item['num_steps']
    
    # 按entities数量分组统计
    entities_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    steps_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for result in results:
        if result['success']:
            item_id = result['id']
            entities_count = id_to_entities_count[item_id]
            num_steps = id_to_num_steps[item_id]
            
            is_correct = result['predicted_option'] == result['correct_option']
            
            # 按entities数量统计
            entities_stats[entities_count]['total'] += 1
            if is_correct:
                entities_stats[entities_count]['correct'] += 1
            
            # 按num_steps统计
            steps_stats[num_steps]['total'] += 1
            if is_correct:
                steps_stats[num_steps]['correct'] += 1
    
    # 打印按entities数量的统计结果
    print("=" * 70)
    print("العلاقة بين الدقة وعدد الكيانات")
    print("=" * 70)
    print(f"{'عدد الكيانات':<15} {'الإجمالي':<10} {'الصحيحة':<10} {'الدقة':<10}")
    print("-" * 70)
    
    total_all = 0
    correct_all = 0
    
    for entities_count in sorted(entities_stats.keys()):
        stats = entities_stats[entities_count]
        accuracy = stats['correct'] / stats['total'] * 100
        total_all += stats['total']
        correct_all += stats['correct']
        print(f"{entities_count:<15} {stats['total']:<10} {stats['correct']:<10} {accuracy:>6.2f}%")
    
    print("-" * 70)
    overall_accuracy = correct_all / total_all * 100
    print(f"{'المجموع':<15} {total_all:<10} {correct_all:<10} {overall_accuracy:>6.2f}%")
    print("=" * 70)
    
    # 打印按num_steps的统计结果
    print("\n" + "=" * 70)
    print("العلاقة بين الدقة وعدد الخطوات")
    print("=" * 70)
    print(f"{'عدد الخطوات':<15} {'الإجمالي':<10} {'الصحيحة':<10} {'الدقة':<10}")
    print("-" * 70)
    
    for num_steps in sorted(steps_stats.keys()):
        stats = steps_stats[num_steps]
        accuracy = stats['correct'] / stats['total'] * 100
        print(f"{num_steps:<15} {stats['total']:<10} {stats['correct']:<10} {accuracy:>6.2f}%")
    
    print("-" * 70)
    print(f"{'المجموع':<15} {total_all:<10} {correct_all:<10} {overall_accuracy:>6.2f}%")
    print("=" * 70)
    
    # 生成详细的分析数据
    analysis_data = {
        'by_entities': {
            str(k): {
                'total': v['total'],
                'correct': v['correct'],
                'accuracy': v['correct'] / v['total'] * 100
            }
            for k, v in entities_stats.items()
        },
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
    analysis_path = os.path.join(os.path.dirname(__file__), './data_relation/accuracy_analysis_AR_test.json')
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nتم حفظ بيانات التحليل التفصيلية في: {analysis_path}")
    
    # 生成可视化的趋势分析
    print("\n" + "=" * 70)
    print("تحليل الاتجاه")
    print("=" * 70)
    
    # 按entities数量分析趋势
    entities_sorted = sorted(entities_stats.items())
    print("\nتغير الدقة مع زيادة عدد الكيانات:")
    for i, (entities_count, stats) in enumerate(entities_sorted):
        accuracy = stats['correct'] / stats['total'] * 100
        bar_length = int(accuracy / 2)  # 最大50个字符
        bar = '█' * bar_length
        print(f"{entities_count:>2} كيانات: [{bar:<50}] {accuracy:>6.2f}% ({stats['correct']}/{stats['total']})")
    
    # 按steps数量分析趋势
    steps_sorted = sorted(steps_stats.items())
    print("\nتغير الدقة مع زيادة عدد الخطوات:")
    for i, (num_steps, stats) in enumerate(steps_sorted):
        accuracy = stats['correct'] / stats['total'] * 100
        bar_length = int(accuracy / 2)  # 最大50个字符
        bar = '█' * bar_length
        print(f"{num_steps:>2} خطوات: [{bar:<50}] {accuracy:>6.2f}% ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    analyze_accuracy_by_entities()







