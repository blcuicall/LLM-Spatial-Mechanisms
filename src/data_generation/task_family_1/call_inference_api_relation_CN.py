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

system_prompt = """你是一个空间推理助手。"""
instruction = """你将获得几条描述三维空间中物体相对位置的陈述。
每条陈述描述的是沿单个轴上恰好一个单位距离的相对位置。（物体可能在空间中占据相同的位置。）

陈述：
{statements}

问题：
{question}

选项：
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

指令：
只输出正确选项的字母（A、B、C 或 D）。
不要提供任何解释、推理步骤或额外的文本。
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
        if "在" in line and "的哪个位置" in line:
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
    json_path = os.path.join(os.path.dirname(__file__), './data_relation/spatial_reasoning_dataset_CN_test_with_prompt.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"从 {json_path} 加载了 {len(data)} 个数据项")
    
    # 并行处理
    max_workers = int(os.getenv("API_TEST_MAX_WORKERS", "32"))
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_item, item) for item in data]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="处理中"):
            result = future.result()
            results.append(result)
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    correct_count = sum(1 for r in results if r['success'] and r['predicted_option'] == r['correct_option'])
    
    print(f"\n结果:")
    print(f"总数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"正确: {correct_count}")
    print(f"准确率: {correct_count / success_count * 100:.2f}%")
    
    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), './data_relation/api_results_CN_test.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存至 {output_path}")




