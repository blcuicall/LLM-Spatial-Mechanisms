#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为视角变换数据集添加提示词（中文版）
"""

INSTRUCTION_TEMPLATE = """
你将获得一系列转向动作。
从一个初始方向开始，你需要追踪方向变化并确定最终方向。

初始方向和动作：
{statements}

问题：
{question}

选项：
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}

说明：
只输出正确选项的字母（A、B、C或D）。
不要提供任何解释、推理步骤或额外文本。
""".strip()

import json

def parse_question(question_text):
    """从 question 字段中解析出 statements 和 question"""
    lines = question_text.strip().split('\n')
    statements_lines = []
    question_line = ""
    
    for line in lines:
        if line.startswith("你现在面向"):
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
    
    print(f"已为 {len(dataset)} 个样本添加提示词")
    print(f"带提示词的数据集已保存到 {output_file}")


def main():
    """主函数"""
    import os
    
    # 检查并处理完整数据集
    if os.path.exists('./data_orientation/orientation_reasoning_dataset_CN.json'):
        print("正在为完整数据集添加提示词...")
        add_prompt_to_dataset(
            input_file='./data_orientation/orientation_reasoning_dataset_CN.json',
            output_file='./data_orientation/orientation_reasoning_dataset_CN_with_prompt.json'
        )
    else:
        print("未找到完整数据集，跳过...")
    
    # 检查并处理测试数据集
    if os.path.exists('./data_orientation/orientation_reasoning_dataset_CN_test.json'):
        print("\n正在为测试数据集添加提示词...")
        add_prompt_to_dataset(
            input_file='./data_orientation/orientation_reasoning_dataset_CN_test.json',
            output_file='./data_orientation/orientation_reasoning_dataset_CN_test_with_prompt.json'
        )
    else:
        print("未找到测试数据集，跳过...")
    
    # 打印一个示例（优先使用完整数据集）
    sample_file = None
    if os.path.exists('./data_orientation/orientation_reasoning_dataset_CN_with_prompt.json'):
        sample_file = './data_orientation/orientation_reasoning_dataset_CN_with_prompt.json'
    elif os.path.exists('./data_orientation/orientation_reasoning_dataset_CN_test_with_prompt.json'):
        sample_file = './data_orientation/orientation_reasoning_dataset_CN_test_with_prompt.json'
    
    if sample_file:
        with open(sample_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print("\n" + "="*60)
        print("示例（带提示词）：")
        print("="*60)
        print(dataset[0]['prompt'])
        print("\n" + "="*60)
        print(f"正确答案：{dataset[0]['correct_option']} ({dataset[0]['answer']})")
        print("="*60)


if __name__ == "__main__":
    main()





