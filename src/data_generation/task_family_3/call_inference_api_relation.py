from openai import OpenAI
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm
import argparse

system_prompt = """You are a spatial reasoning assistant."""

# 初始化模型
client = OpenAI(
    base_url=os.environ.get("INFERENCE_API_BASE", "http://127.0.0.1:8010/v1"),
    api_key='EMPTY'
)

def call_api(input_text):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
    )
    return completion.choices[0].message.content

def process_one_item(item):
    """处理单个数据项"""
    # 调用 API
    response = call_api(item['prompt'])
    
    # 返回结果
    return {
        'id': item['id'],
        'correct_option': item['correct_option'],
        'predicted_option': response.strip(),
        'answer': item['answer'],
        'success': True
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', '-l', type=str, default='EN')
    parser.add_argument('--model', '-m', type=str, default='Qwen2.5-7B-Instruct')
    args = parser.parse_args()

    MODEL_NAME = args.model
    
    # 读取 JSON 数据
    json_path = os.path.join(os.path.dirname(__file__), f'./data_procedure/spatial_procedure_dataset_{args.language}_test_with_prompt.json')
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
    output_path = os.path.join(os.path.dirname(__file__), f'./results/api_results_{args.model}_{args.language}_test.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_path}")