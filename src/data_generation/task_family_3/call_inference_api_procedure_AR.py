from openai import OpenAI
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

MODEL_NAME = "Qwen2.5-7B-Instruct"

# 初始化模型
client = OpenAI(
    base_url=os.environ.get("INFERENCE_API_BASE", "http://127.0.0.1:8010/v1"),
    api_key='EMPTY'
)

system_prompt = """أنت مساعد للتفكير المكاني."""
instruction = """سيتم إعطاؤك موضع بداية وسلسلة من العمليات المكانية.
كل عملية تحول الموضع في الفضاء ثلاثي الأبعاد.

العمليات المكانية:
{statements}

السؤال:
{question}

الخيارات:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

التعليمات:
اكتب فقط حرف الخيار الصحيح (A أو B أو C أو D).
لا تقدم أي تفسير أو خطوات استنتاج أو نص إضافي.
"""

def call_api(input_text):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
    )
    return completion.choices[0].message.content

def parse_question(question_text):
    """从 question 字段中解析出 statements 和 question"""
    lines = question_text.strip().split('\n')
    statements_lines = []
    question_line = ""
    
    for line in lines:
        if line.startswith("ما هو الموضع"):
            question_line = line
        else:
            statements_lines.append(line)
    
    statements = '\n'.join(statements_lines)
    return statements, question_line

def process_one_item(item):
    """处理单个数据项"""
    try:
        # 解析 question 字段
        statements, question = parse_question(item['question'])
        
        # 获取选项
        options = item['options']
        option_A = options[0]
        option_B = options[1]
        option_C = options[2]
        option_D = options[3]
        
        # 构造 prompt
        input_text = instruction.format(
            statements=statements, 
            question=question, 
            option_A=option_A, 
            option_B=option_B, 
            option_C=option_C, 
            option_D=option_D
        )
        
        # 调用 API
        response = call_api(input_text)
        
        # 返回结果
        return {
            'id': item['id'],
            'correct_option': item['correct_option'],
            'predicted_option': response.strip(),
            'answer': item['answer'],
            'success': True
        }
    except Exception as e:
        return {
            'id': item.get('id', 'unknown'),
            'error': str(e),
            'success': False
        }

if __name__ == "__main__":
    # 读取 JSON 数据
    json_path = os.path.join(os.path.dirname(__file__), './data_procedure/spatial_procedure_dataset_AR_test_with_prompt.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from {json_path}")
    
    # 并行处理
    max_workers = int(os.getenv("API_TEST_MAX_WORKERS", "32"))
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_item, item) for item in data]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results.append(result)
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    correct_count = sum(1 for r in results if r['success'] and r['predicted_option'] == r['correct_option'])
    
    print(f"\nResults:")
    print(f"Total: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {correct_count / success_count * 100:.2f}%")
    
    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), './data_procedure/api_results_AR_test.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")




