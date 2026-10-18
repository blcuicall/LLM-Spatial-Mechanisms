#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间过程执行数据集生成器
生成基于空间操作序列的推理问题
"""

import json
import random
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SpatialAction:
    """空间操作动作"""
    action_type: str
    params: Dict
    
    def __str__(self):
        if self.action_type == "move":
            direction = self.params['direction']
            units = self.params['units']
            return f"Move {direction} by {units} {'unit' if units == 1 else 'units'}."
        elif self.action_type == "reflect":
            axis = self.params['axis']
            return f"Reflect the position across the {axis}-axis."
        elif self.action_type == "rotate":
            axis = self.params['axis']
            degrees = self.params['degrees']
            return f"Rotate {degrees}° around the {axis}-axis."
        elif self.action_type == "scale":
            factor = self.params['factor']
            return f"Scale all coordinates by {factor}."
        elif self.action_type == "translate":
            dx = self.params['dx']
            dy = self.params['dy']
            dz = self.params['dz']
            return f"Translate by ({dx}, {dy}, {dz})."
        return ""


class SpatialProcedureGenerator:
    """空间过程执行数据集生成器"""
    
    # 定义移动方向及其对应的坐标变化
    MOVE_DIRECTIONS = {
        'up': (0, 0, 1),        # z+
        'down': (0, 0, -1),     # z-
        'forward': (0, 1, 0),   # y+
        'backward': (0, -1, 0), # y-
        'left': (-1, 0, 0),     # x-
        'right': (1, 0, 0)      # x+
    }
    
    # 定义坐标轴
    AXES = ['x', 'y', 'z']
    
    # 定义可能的旋转角度
    ROTATION_ANGLES = [90, 180, 270]
    
    # 定义缩放因子
    SCALE_FACTORS = [2, 3, 0.5, -1]
    
    def __init__(self, seed: Optional[int] = None):
        """初始化生成器"""
        if seed is not None:
            random.seed(seed)
    
    def apply_move(self, position: Tuple[float, float, float], 
                   direction: str, units: int) -> Tuple[float, float, float]:
        """
        应用移动操作
        
        Args:
            position: 当前位置 (x, y, z)
            direction: 移动方向
            units: 移动单位数
            
        Returns:
            new_position: 新位置
        """
        dx, dy, dz = self.MOVE_DIRECTIONS[direction]
        x, y, z = position
        return (x + dx * units, y + dy * units, z + dz * units)
    
    def apply_reflect(self, position: Tuple[float, float, float], 
                      axis: str) -> Tuple[float, float, float]:
        """
        应用反射操作
        
        Args:
            position: 当前位置 (x, y, z)
            axis: 反射轴
            
        Returns:
            new_position: 新位置
        """
        x, y, z = position
        if axis == 'x':
            return (x, -y, z)  # 沿x轴反射，y坐标取反
        elif axis == 'y':
            return (-x, y, z)  # 沿y轴反射，x坐标取反
        else:  # z
            return (x, y, -z)  # 沿z轴反射，z坐标取反
    
    def apply_rotate(self, position: Tuple[float, float, float], 
                     axis: str, degrees: int) -> Tuple[float, float, float]:
        """
        应用旋转操作
        
        Args:
            position: 当前位置 (x, y, z)
            axis: 旋转轴
            degrees: 旋转角度
            
        Returns:
            new_position: 新位置
        """
        x, y, z = position
        rad = math.radians(degrees)
        cos_val = math.cos(rad)
        sin_val = math.sin(rad)
        
        # 处理浮点数精度
        cos_val = round(cos_val, 10)
        sin_val = round(sin_val, 10)
        
        if axis == 'x':
            # 绕x轴旋转，y和z坐标变化
            new_y = cos_val * y - sin_val * z
            new_z = sin_val * y + cos_val * z
            return (x, round(new_y, 10), round(new_z, 10))
        elif axis == 'y':
            # 绕y轴旋转，x和z坐标变化
            new_x = cos_val * x + sin_val * z
            new_z = -sin_val * x + cos_val * z
            return (round(new_x, 10), y, round(new_z, 10))
        else:  # z
            # 绕z轴旋转，x和y坐标变化
            new_x = cos_val * x - sin_val * y
            new_y = sin_val * x + cos_val * y
            return (round(new_x, 10), round(new_y, 10), z)
    
    def apply_scale(self, position: Tuple[float, float, float], 
                    factor: float) -> Tuple[float, float, float]:
        """
        应用缩放操作
        
        Args:
            position: 当前位置 (x, y, z)
            factor: 缩放因子
            
        Returns:
            new_position: 新位置
        """
        x, y, z = position
        return (x * factor, y * factor, z * factor)
    
    def apply_translate(self, position: Tuple[float, float, float], 
                        dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
        """
        应用平移操作
        
        Args:
            position: 当前位置 (x, y, z)
            dx, dy, dz: 各轴平移量
            
        Returns:
            new_position: 新位置
        """
        x, y, z = position
        return (x + dx, y + dy, z + dz)
    
    def apply_action(self, position: Tuple[float, float, float], 
                     action: SpatialAction) -> Tuple[float, float, float]:
        """
        应用空间操作
        
        Args:
            position: 当前位置
            action: 空间操作
            
        Returns:
            new_position: 新位置
        """
        if action.action_type == "move":
            return self.apply_move(position, action.params['direction'], 
                                  action.params['units'])
        elif action.action_type == "reflect":
            return self.apply_reflect(position, action.params['axis'])
        elif action.action_type == "rotate":
            return self.apply_rotate(position, action.params['axis'], 
                                    action.params['degrees'])
        elif action.action_type == "scale":
            return self.apply_scale(position, action.params['factor'])
        elif action.action_type == "translate":
            return self.apply_translate(position, action.params['dx'], 
                                       action.params['dy'], action.params['dz'])
        return position
    
    def generate_random_action(self) -> SpatialAction:
        """
        生成随机空间操作
        
        Returns:
            action: 随机空间操作
        """
        action_type = random.choice(['move', 'reflect', 'rotate', 'scale', 'translate'])
        
        if action_type == 'move':
            direction = random.choice(list(self.MOVE_DIRECTIONS.keys()))
            units = random.randint(1, 5)
            return SpatialAction('move', {'direction': direction, 'units': units})
        
        elif action_type == 'reflect':
            axis = random.choice(self.AXES)
            return SpatialAction('reflect', {'axis': axis})
        
        elif action_type == 'rotate':
            axis = random.choice(self.AXES)
            degrees = random.choice(self.ROTATION_ANGLES)
            return SpatialAction('rotate', {'axis': axis, 'degrees': degrees})
        
        elif action_type == 'scale':
            factor = random.choice(self.SCALE_FACTORS)
            return SpatialAction('scale', {'factor': factor})
        
        else:  # translate
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            dz = random.randint(-3, 3)
            return SpatialAction('translate', {'dx': dx, 'dy': dy, 'dz': dz})
    
    def generate_action_sequence(self, num_steps: int) -> Tuple[List[SpatialAction], 
                                                                Tuple[float, float, float]]:
        """
        生成空间操作序列
        
        Args:
            num_steps: 操作步骤数量
            
        Returns:
            actions: 操作列表
            final_position: 最终位置
        """
        # 起始位置
        start_position = (0.0, 0.0, 0.0)
        current_position = start_position
        
        # 生成操作序列
        actions = []
        for _ in range(num_steps):
            action = self.generate_random_action()
            actions.append(action)
            current_position = self.apply_action(current_position, action)
        
        # 四舍五入最终结果
        final_position = tuple(round(coord, 2) for coord in current_position)
        
        return actions, final_position
    
    def generate_distractor_options(self, correct_answer: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """
        生成干扰选项
        
        Args:
            correct_answer: 正确答案
            
        Returns:
            distractors: 干扰选项列表
        """
        x, y, z = correct_answer
        distractors = []
        
        # 生成不同类型的干扰项
        variations = [
            (x, y, -z),      # z坐标符号错误
            (x, -y, z),      # y坐标符号错误
            (-x, y, z),      # x坐标符号错误
            (y, x, z),       # x和y交换
            (x, z, y),       # y和z交换
            (z, y, x),       # x和z交换
            (x + 1, y, z),   # x坐标偏移
            (x, y + 1, z),   # y坐标偏移
            (x, y, z + 1),   # z坐标偏移
            (x * 2, y, z),   # x坐标缩放
            (x, y * 2, z),   # y坐标缩放
            (x, y, z * 2),   # z坐标缩放
        ]
        
        # 选择不同的干扰项
        for var in variations:
            if var != correct_answer and var not in distractors:
                distractors.append(var)
            if len(distractors) >= 3:
                break
        
        # 如果干扰项不够，生成随机的
        while len(distractors) < 3:
            offset = random.randint(1, 3)
            sign = random.choice([-1, 1])
            coord_idx = random.randint(0, 2)
            
            distractor = list(correct_answer)
            distractor[coord_idx] += sign * offset
            distractor = tuple(distractor)
            
            if distractor not in distractors and distractor != correct_answer:
                distractors.append(distractor)
        
        return distractors[:3]
    
    def format_position(self, position: Tuple[float, float, float]) -> str:
        """格式化位置为字符串"""
        x, y, z = position
        # 如果是整数，就显示为整数
        x_str = str(int(x)) if x == int(x) else str(x)
        y_str = str(int(y)) if y == int(y) else str(y)
        z_str = str(int(z)) if z == int(z) else str(z)
        return f"({x_str}, {y_str}, {z_str})"
    
    def generate_question_answer(
        self, 
        actions: List[SpatialAction],
        start_position: Tuple[float, float, float],
        final_position: Tuple[float, float, float]
    ) -> Tuple[str, str, List[str], str]:
        """
        生成问题和答案
        
        Args:
            actions: 操作列表
            start_position: 起始位置
            final_position: 最终位置
            
        Returns:
            question: 问题字符串
            answer: 答案字符串
            options: 选项列表
            correct_option: 正确选项标识
        """
        # 构建问题
        premise_lines = [f"Start at {self.format_position(start_position)}."]
        premise_lines.extend([str(action) for action in actions])
        premise = "\n".join(premise_lines)
        question = f"{premise}\n\nWhat is the final position?"
        
        # 答案
        answer = self.format_position(final_position)
        
        # 生成干扰选项
        distractors = self.generate_distractor_options(final_position)
        distractor_strs = [self.format_position(d) for d in distractors]
        
        # 将正确答案和干扰项组合并随机打乱
        all_options = [answer] + distractor_strs
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
            min_steps: 最小步骤数
            max_steps: 最大步骤数
            output_file: 输出文件路径（可选）
            
        Returns:
            dataset: 数据集列表
        """
        dataset = []
        
        for i in range(num_samples):
            # 随机选择步骤数
            num_steps = random.randint(min_steps, max_steps)
            
            # 生成操作序列
            actions, final_position = self.generate_action_sequence(num_steps)
            start_position = (0.0, 0.0, 0.0)
            
            # 生成问题和答案
            question, answer, options, correct_option = self.generate_question_answer(
                actions, start_position, final_position
            )
            
            # 构建数据样本
            sample = {
                'id': i + 1,
                'num_steps': num_steps,
                'question': question,
                'answer': answer,
                'options': options,
                'correct_option': correct_option,
                'start_position': list(start_position),
                'final_position': list(final_position),
                'actions': [
                    {
                        'type': action.action_type,
                        'params': action.params,
                        'description': str(action)
                    }
                    for action in actions
                ],
                'target': list(final_position)  # 用于探针的目标向量
            }
            
            dataset.append(sample)
            
            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples...")
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"\nDataset saved to {output_file}")
        
        return dataset
    
    def print_sample(self, sample: Dict):
        """打印样本示例"""
        print(f"\n{'='*60}")
        print(f"Sample ID: {sample['id']}")
        print(f"Number of steps: {sample['num_steps']}")
        print(f"\n{sample['question']}")
        print(f"\nOptions:")
        for i, option in enumerate(sample['options']):
            option_label = chr(65 + i)  # A, B, C, D
            mark = " ✓" if option_label == sample['correct_option'] else ""
            print(f"  {option_label}. {option}{mark}")
        print(f"\nCorrect Answer: {sample['correct_option']} ({sample['answer']})")
        print(f"Start: {sample['start_position']} -> Final: {sample['final_position']}")
        print(f"{'='*60}\n")


def main():
    """主函数"""
    # 创建生成器
    generator = SpatialProcedureGenerator(seed=42)
    
    # 生成数据集
    print("Generating spatial procedure execution dataset...")
    dataset = generator.generate_dataset(
        num_samples=2000,
        min_steps=2,
        max_steps=10,
        output_file='./data_procedure/spatial_procedure_dataset_EN.json'
    )
    
    # 打印一些示例
    print("\n" + "="*60)
    print("Sample Examples:")
    print("="*60)
    
    # 打印不同步骤数的示例
    for num_steps in [2, 4, 6, 8, 10]:
        samples_with_n_steps = [s for s in dataset if s['num_steps'] == num_steps]
        if samples_with_n_steps:
            generator.print_sample(samples_with_n_steps[0])
    
    # 统计信息
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(f"Total samples: {len(dataset)}")
    
    step_distribution = defaultdict(int)
    for sample in dataset:
        step_distribution[sample['num_steps']] += 1
    
    print("\nDistribution by number of steps:")
    for steps in sorted(step_distribution.keys()):
        print(f"  {steps} steps: {step_distribution[steps]} samples")
    
    # 统计操作类型分布
    action_type_count = defaultdict(int)
    for sample in dataset:
        for action in sample['actions']:
            action_type_count[action['type']] += 1
    
    print("\nDistribution by action type:")
    for action_type in sorted(action_type_count.keys()):
        print(f"  {action_type}: {action_type_count[action_type]} actions")


if __name__ == "__main__":
    main()

