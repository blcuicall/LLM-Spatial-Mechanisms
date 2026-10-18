#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为空间过程执行数据集添加提示词（阿拉伯语版）
"""

INSTRUCTION_TEMPLATE = """
سيتم إعطاؤك موضع بداية وسلسلة من العمليات المكانية.
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
""".strip()

import json

def parse_question(question_text):
    """من question 字段中解析出 statements 和 question"""
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

def construct_prompt(sample):
    """构造完整的 prompt"""
    # 解析 question 字段
    statements, question = parse_question(sample['question'])
    
    # 获取选项
    options = sample['options']
    option_A = options[0]
    option_B = options[1]
    option_C = options[2]
    option_D = options[3]
    
    # 构造 prompt
    prompt = INSTRUCTION_TEMPLATE.format(
        statements=statements, 
        question=question, 
        option_A=option_A, 
        option_B=option_B, 
        option_C=option_C, 
        option_D=option_D
    )
    
    return prompt

def add_prompt_to_dataset(input_file: str, output_file: str):
    """
    为数据集添加提示词
    
    Args:
        input_file: 输入的数据集文件
        output_file: 输出的带提示词的数据集文件
    """
    # 读取原始数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    # 为每个样本添加提示词
    for sample in dataset:
        prompt = construct_prompt(sample)
        sample['prompt'] = prompt
    
    # 保存带提示词的数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Added prompts to {len(dataset)} samples")
    print(f"Dataset with prompts saved to {output_file}")


def main():
    """主函数"""
    import os
    
    # 检查并处理完整数据集
    if os.path.exists('./data_procedure/spatial_procedure_dataset_AR.json'):
        print("Adding prompts to full dataset (Arabic)...")
        add_prompt_to_dataset(
            input_file='./data_procedure/spatial_procedure_dataset_AR.json',
            output_file='./data_procedure/spatial_procedure_dataset_AR_with_prompt.json'
        )
    else:
        print("Full dataset not found, skipping...")
    
    # 检查并处理测试数据集
    if os.path.exists('./data_procedure/spatial_procedure_dataset_AR_test.json'):
        print("\nAdding prompts to test dataset (Arabic)...")
        add_prompt_to_dataset(
            input_file='./data_procedure/spatial_procedure_dataset_AR_test.json',
            output_file='./data_procedure/spatial_procedure_dataset_AR_test_with_prompt.json'
        )
    else:
        print("Test dataset not found, skipping...")
    
    # 打印一个示例（优先使用完整数据集）
    sample_file = None
    if os.path.exists('./data_procedure/spatial_procedure_dataset_AR_with_prompt.json'):
        sample_file = './data_procedure/spatial_procedure_dataset_AR_with_prompt.json'
    elif os.path.exists('./data_procedure/spatial_procedure_dataset_AR_test_with_prompt.json'):
        sample_file = './data_procedure/spatial_procedure_dataset_AR_test_with_prompt.json'
    
    if sample_file:
        with open(sample_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print("\n" + "="*60)
        print("Sample with Prompt:")
        print("="*60)
        print(dataset[0]['prompt'])
        print("\n" + "="*60)
        print(f"Correct Answer: {dataset[0]['correct_option']} ({dataset[0]['answer']})")
        print("="*60)


if __name__ == "__main__":
    main()





