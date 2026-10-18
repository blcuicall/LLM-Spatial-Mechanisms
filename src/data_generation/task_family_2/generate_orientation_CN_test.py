#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方向推理数据集生成器（测试集）（中文版）
生成基于转向操作的方向推理问题
"""

import json
import random
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TurnAction:
    """转向动作"""
    action: str
    
    def __str__(self):
        return f"{self.action}。"


class OrientationReasoningGenerator:
    """方向推理数据集生成器"""
    
    # 定义4个基本方向（顺时针顺序）
    DIRECTIONS = ['北', '东', '南', '西']
    
    # 定义转向动作
    ACTIONS = ['向右转', '向左转', '转身']
    
    # 定义动作对应的旋转角度（以90度为单位，顺时针为正）
    ACTION_ROTATIONS = {
        '向右转': 1,      # 顺时针90度
        '向左转': -1,      # 逆时针90度
        '转身': 2      # 180度
    }
    
    # 定义方向对应的角度（度数，从东开始逆时针）
    # 东: 0°, 北: 90°, 西: 180°, 南: 270°
    DIRECTION_ANGLES = {
        '东': 0,
        '北': 90,
        '西': 180,
        '南': 270
    }
    
    def __init__(self, seed: Optional[int] = None):
        """初始化生成器"""
        if seed is not None:
            random.seed(seed)
    
    def encode_direction(self, direction: str) -> List[float]:
        """
        将方向编码为 [cos(θ), sin(θ)]
        
        Args:
            direction: 方向字符串 (北/东/南/西)
            
        Returns:
            encoded: [cos(θ), sin(θ)] 编码
        """
        angle_degrees = self.DIRECTION_ANGLES[direction]
        angle_radians = math.radians(angle_degrees)
        
        cos_val = math.cos(angle_radians)
        sin_val = math.sin(angle_radians)
        
        # 处理浮点数精度问题，四舍五入到小数点后10位
        cos_val = round(cos_val, 10)
        sin_val = round(sin_val, 10)
        
        return [cos_val, sin_val]
    
    def apply_action(self, current_direction: str, action: str) -> str:
        """
        应用转向动作
        
        Args:
            current_direction: 当前方向
            action: 转向动作
            
        Returns:
            new_direction: 新的方向
        """
        # 获取当前方向的索引
        current_index = self.DIRECTIONS.index(current_direction)
        
        # 获取旋转步数
        rotation = self.ACTION_ROTATIONS[action]
        
        # 计算新方向的索引
        new_index = (current_index + rotation) % 4
        
        return self.DIRECTIONS[new_index]
    
    def generate_action_sequence(self, num_steps: int) -> Tuple[List[TurnAction], str, str]:
        """
        生成转向动作序列
        
        Args:
            num_steps: 转向步骤数量（2-15）
            
        Returns:
            actions: 转向动作列表
            start_direction: 初始方向
            end_direction: 最终方向
        """
        # 随机选择初始方向
        start_direction = random.choice(self.DIRECTIONS)
        current_direction = start_direction
        
        # 生成动作序列
        actions = []
        for _ in range(num_steps):
            # 随机选择一个动作
            action = random.choice(self.ACTIONS)
            actions.append(TurnAction(action=action))
            
            # 应用动作
            current_direction = self.apply_action(current_direction, action)
        
        end_direction = current_direction
        
        return actions, start_direction, end_direction
    
    def verify_needs_reasoning(
        self,
        actions: List[TurnAction],
        start_direction: str,
        end_direction: str
    ) -> bool:
        """
        验证问题是否需要多步推理（最终方向与初始方向不同）
        
        Args:
            actions: 转向动作列表
            start_direction: 初始方向
            end_direction: 最终方向
            
        Returns:
            True if the question requires multi-step reasoning
        """
        # 如果只有一步，则不需要多步推理
        if len(actions) <= 1:
            return False
        
        # 如果最终方向与初始方向相同，则认为问题过于简单
        # 但这不是必要条件，因为多次转向后可能回到初始方向
        return True
    
    def generate_distractor_options(self, correct_answer: str) -> List[str]:
        """
        生成干扰选项
        
        Args:
            correct_answer: 正确答案
            
        Returns:
            distractors: 干扰选项列表（其他三个方向）
        """
        distractors = [d for d in self.DIRECTIONS if d != correct_answer]
        return distractors
    
    def generate_question_answer(
        self, 
        actions: List[TurnAction],
        start_direction: str,
        end_direction: str
    ) -> Tuple[str, str, List[str], str]:
        """
        生成问题和答案
        
        Args:
            actions: 转向动作列表
            start_direction: 初始方向
            end_direction: 最终方向
            
        Returns:
            question: 问题字符串
            answer: 答案字符串
            options: 选项列表（包含正确答案和干扰项的随机排列）
            correct_option: 正确选项标识（A/B/C/D）
        """
        # 构建问题
        premise_lines = [f"你面向{start_direction}方。"]
        premise_lines.extend([str(action) for action in actions])
        premise = "\n".join(premise_lines)
        question = f"{premise}\n\n你现在面向哪个方向？"
        
        # 答案
        answer = end_direction
        
        # 生成干扰选项
        distractors = self.generate_distractor_options(answer)
        
        # 将正确答案和干扰项组合并随机打乱
        all_options = [answer] + distractors
        random.shuffle(all_options)
        
        # 找到正确答案的位置
        correct_index = all_options.index(answer)
        correct_option = chr(65 + correct_index)  # 0->A, 1->B, 2->C, 3->D
        
        return question, answer, all_options, correct_option
    
    def generate_dataset(
        self, 
        num_samples: int = 1000,
        min_steps: int = 2,
        max_steps: int = 10,
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        生成完整的数据集
        
        Args:
            num_samples: 样本数量
            min_steps: 最小推理步骤
            max_steps: 最大推理步骤
            output_file: 输出文件路径（可选）
            
        Returns:
            dataset: 数据集列表
        """
        dataset = []
        
        for i in range(num_samples):
            # 随机选择推理步骤数
            num_steps = random.randint(min_steps, max_steps)
            
            # 生成动作序列
            actions, start_direction, end_direction = self.generate_action_sequence(num_steps)
            
            # 验证是否需要推理
            if not self.verify_needs_reasoning(actions, start_direction, end_direction):
                # 如果不需要推理，重新生成
                continue
            
            # 生成问题和答案
            question, answer, options, correct_option = self.generate_question_answer(
                actions, start_direction, end_direction
            )
            
            # 编码最终方向为 [cos(θ), sin(θ)]
            target_encoding = self.encode_direction(end_direction)
            
            # 构建数据样本
            sample = {
                'id': i + 1,
                'num_steps': num_steps,
                'question': question,
                'answer': answer,
                'options': options,
                'correct_option': correct_option,
                'start_direction': start_direction,
                'end_direction': end_direction,
                'actions': [action.action for action in actions],
                'target': target_encoding
            }
            
            dataset.append(sample)
            
            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{num_samples} 个样本...")
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"\n数据集已保存到 {output_file}")
        
        return dataset
    
    def print_sample(self, sample: Dict):
        """打印样本示例"""
        print(f"\n{'='*60}")
        print(f"样本 ID: {sample['id']}")
        print(f"推理步骤数: {sample['num_steps']}")
        print(f"\n{sample['question']}")
        print(f"\n选项：")
        for i, option in enumerate(sample['options']):
            option_label = chr(65 + i)  # A, B, C, D
            mark = " ✓" if option_label == sample['correct_option'] else ""
            print(f"  {option_label}. {option}{mark}")
        print(f"\n正确答案: {sample['correct_option']} ({sample['answer']})")
        print(f"起始: {sample['start_direction']} -> 结束: {sample['end_direction']}")
        print(f"目标编码 [cos(θ), sin(θ)]: {sample['target']}")
        print(f"{'='*60}\n")


def main():
    """主函数"""
    # 创建生成器
    generator = OrientationReasoningGenerator(seed=36)
    
    # 生成数据集
    print("正在生成方向推理数据集（测试集）（中文版）...")
    dataset = generator.generate_dataset(
        num_samples=200,
        min_steps=2,
        max_steps=10,
        output_file='./data_orientation/orientation_reasoning_dataset_CN_test.json'
    )
    
    # 打印一些示例
    print("\n" + "="*60)
    print("样本示例：")
    print("="*60)
    
    # 打印不同推理步骤的示例
    for num_steps in [2, 5, 8, 10]:
        samples_with_n_steps = [s for s in dataset if s['num_steps'] == num_steps]
        if samples_with_n_steps:
            generator.print_sample(samples_with_n_steps[0])
    
    # 统计信息
    print("\n" + "="*60)
    print("数据集统计：")
    print("="*60)
    print(f"总样本数: {len(dataset)}")
    
    step_distribution = defaultdict(int)
    for sample in dataset:
        step_distribution[sample['num_steps']] += 1
    
    print("\n按步骤数分布：")
    for steps in sorted(step_distribution.keys()):
        print(f"  {steps} 步: {step_distribution[steps]} 个样本")
    
    # 统计最终方向分布
    direction_distribution = defaultdict(int)
    for sample in dataset:
        direction_distribution[sample['end_direction']] += 1
    
    print("\n按最终方向分布：")
    for direction in generator.DIRECTIONS:
        print(f"  {direction}: {direction_distribution[direction]} 个样本")


if __name__ == "__main__":
    main()





