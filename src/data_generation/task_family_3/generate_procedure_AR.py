#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间过程执行数据集生成器（阿拉伯语版）
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
            direction_map = {
                'up': 'الأعلى',
                'down': 'الأسفل',
                'forward': 'الأمام',
                'backward': 'الخلف',
                'left': 'اليسار',
                'right': 'اليمين'
            }
            direction = direction_map[self.params['direction']]
            units = self.params['units']
            unit_word = 'وحدة' if units == 1 else 'وحدات'
            return f"تحرك إلى {direction} بمقدار {units} {unit_word}."
        elif self.action_type == "reflect":
            axis = self.params['axis']
            return f"انعكس عبر محور {axis}."
        elif self.action_type == "rotate":
            axis = self.params['axis']
            degrees = self.params['degrees']
            return f"دوران {degrees}° حول محور {axis}."
        elif self.action_type == "scale":
            factor = self.params['factor']
            return f"قم بتحجيم جميع الإحداثيات بمعامل {factor}."
        elif self.action_type == "translate":
            dx = self.params['dx']
            dy = self.params['dy']
            dz = self.params['dz']
            return f"إزاحة بمقدار ({dx}, {dy}, {dz})."
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
    
    # 定义缩放因子 (修改为更温和的值，避免极端放大)
    SCALE_FACTORS = [2, 0.5, -1]  # 移除 3，避免过度放大
    
    # 坐标范围限制
    MAX_COORD_VALUE = 50  # 单个坐标的最大绝对值
    
    def __init__(self, seed: Optional[int] = None):
        """初始化生成器"""
        if seed is not None:
            random.seed(seed)
        self.consecutive_scale_count = 0  # 跟踪连续 scale 操作
    
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
    
    def generate_random_action(self, current_position: Optional[Tuple[float, float, float]] = None) -> SpatialAction:
        """
        生成随机空间操作（改进版：避免连续 scale 和极端值）
        
        Args:
            current_position: 当前位置（用于避免生成导致越界的操作）
        
        Returns:
            action: 随机空间操作
        """
        # 如果连续 2 次 scale，则降低 scale 的概率
        if self.consecutive_scale_count >= 2:
            action_type = random.choice(['move', 'move', 'reflect', 'rotate', 'translate'])
        else:
            action_type = random.choice(['move', 'reflect', 'rotate', 'scale', 'translate'])
        
        if action_type == 'move':
            direction = random.choice(list(self.MOVE_DIRECTIONS.keys()))
            units = random.randint(1, 5)
            self.consecutive_scale_count = 0
            return SpatialAction('move', {'direction': direction, 'units': units})
        
        elif action_type == 'reflect':
            axis = random.choice(self.AXES)
            self.consecutive_scale_count = 0
            return SpatialAction('reflect', {'axis': axis})
        
        elif action_type == 'rotate':
            axis = random.choice(self.AXES)
            degrees = random.choice(self.ROTATION_ANGLES)
            self.consecutive_scale_count = 0
            return SpatialAction('rotate', {'axis': axis, 'degrees': degrees})
        
        elif action_type == 'scale':
            # 如果当前坐标已经很大，避免放大操作
            if current_position is not None:
                max_abs_coord = max(abs(current_position[0]), abs(current_position[1]), abs(current_position[2]))
                if max_abs_coord > 20:
                    # 只允许缩小或取反
                    factor = random.choice([0.5, -1])
                else:
                    factor = random.choice(self.SCALE_FACTORS)
            else:
                factor = random.choice(self.SCALE_FACTORS)
            
            self.consecutive_scale_count += 1
            return SpatialAction('scale', {'factor': factor})
        
        else:  # translate
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            dz = random.randint(-3, 3)
            self.consecutive_scale_count = 0
            return SpatialAction('translate', {'dx': dx, 'dy': dy, 'dz': dz})
    
    def generate_action_sequence(self, num_steps: int, max_retries: int = 10) -> Tuple[List[SpatialAction], 
                                                                Tuple[float, float, float]]:
        """
        生成空间操作序列（改进版：添加坐标范围检查和重试机制）
        
        Args:
            num_steps: 操作步骤数量
            max_retries: 最大重试次数
            
        Returns:
            actions: 操作列表
            final_position: 最终位置
        """
        for retry in range(max_retries):
            # 重置连续 scale 计数
            self.consecutive_scale_count = 0
            
            # 起始位置
            start_position = (0.0, 0.0, 0.0)
            current_position = start_position
            
            # 生成操作序列
            actions = []
            valid_sequence = True
            
            for _ in range(num_steps):
                action = self.generate_random_action(current_position)
                new_position = self.apply_action(current_position, action)
                
                # 检查是否超出范围
                if any(abs(coord) > self.MAX_COORD_VALUE for coord in new_position):
                    valid_sequence = False
                    break
                
                actions.append(action)
                current_position = new_position
            
            # 如果生成的序列有效，返回结果
            if valid_sequence:
                # 四舍五入最终结果
                final_position = tuple(round(coord, 2) for coord in current_position)
                return actions, final_position
        
        # 如果重试多次仍失败，使用最后一次的结果（裁剪到范围内）
        final_position = tuple(
            round(max(-self.MAX_COORD_VALUE, min(self.MAX_COORD_VALUE, coord)), 2) 
            for coord in current_position
        )
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
        premise_lines = [f"ابدأ من {self.format_position(start_position)}."]
        premise_lines.extend([str(action) for action in actions])
        premise = "\n".join(premise_lines)
        question = f"{premise}\n\nما هو الموضع النهائي؟"
        
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
        
        # 数据集质量验证
        self._validate_dataset_quality(dataset)
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"\nDataset saved to {output_file}")
        
        return dataset
    
    def _validate_dataset_quality(self, dataset: List[Dict]):
        """
        验证数据集质量，确保方差和范围在合理范围内
        
        Args:
            dataset: 数据集
        """
        import numpy as np
        
        targets = np.array([s['target'] for s in dataset])
        
        mean = targets.mean(axis=0)
        std = targets.std(axis=0)
        min_vals = targets.min(axis=0)
        max_vals = targets.max(axis=0)
        ranges = max_vals - min_vals
        
        print("\n" + "="*60)
        print("Dataset Quality Validation:")
        print("="*60)
        print(f"Target Statistics:")
        print(f"  Mean:  x={mean[0]:.2f}, y={mean[1]:.2f}, z={mean[2]:.2f}")
        print(f"  Std:   x={std[0]:.2f}, y={std[1]:.2f}, z={std[2]:.2f}")
        print(f"  Range: x=[{min_vals[0]:.1f}, {max_vals[0]:.1f}], "
              f"y=[{min_vals[1]:.1f}, {max_vals[1]:.1f}], "
              f"z=[{min_vals[2]:.1f}, {max_vals[2]:.1f}]")
        
        # 检查是否有轴的方差过大
        max_std = std.max()
        min_std = std.min()
        std_ratio = max_std / min_std if min_std > 0 else float('inf')
        
        print(f"\nBalance Check:")
        print(f"  Max std: {max_std:.2f}")
        print(f"  Min std: {min_std:.2f}")
        print(f"  Std ratio: {std_ratio:.2f}")
        
        # 警告
        if std_ratio > 3.0:
            print(f"\n⚠️  WARNING: Std ratio > 3.0, data may be imbalanced!")
            print(f"   Consider regenerating with different seed or adjusting SCALE_FACTORS.")
        elif std_ratio > 2.0:
            print(f"\n⚡ CAUTION: Std ratio > 2.0, data slightly imbalanced.")
        else:
            print(f"\n✓ Data balance is good (std ratio < 2.0)")
        
        # 检查极端值
        max_abs = np.abs(targets).max()
        extreme_count = (np.abs(targets) > self.MAX_COORD_VALUE * 0.8).sum()
        
        print(f"\nExtreme Values Check:")
        print(f"  Max absolute value: {max_abs:.1f}")
        print(f"  Coords near limit (>{self.MAX_COORD_VALUE * 0.8:.1f}): {extreme_count}")
        
        if max_abs > self.MAX_COORD_VALUE:
            print(f"\n⚠️  WARNING: Some coordinates exceed MAX_COORD_VALUE ({self.MAX_COORD_VALUE})!")
        else:
            print(f"\n✓ All coordinates within bounds")
        
        print("="*60)
    
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
    print("Generating spatial procedure execution dataset (Arabic)...")
    dataset = generator.generate_dataset(
        num_samples=2000,
        min_steps=2,
        max_steps=10,
        output_file='./data_procedure/spatial_procedure_dataset_AR.json'
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





