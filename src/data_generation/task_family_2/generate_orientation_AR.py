#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مولد مجموعة بيانات الاستدلال الاتجاهي (النسخة العربية)
توليد أسئلة الاستدلال الاتجاهي بناءً على عمليات الدوران
"""

import json
import random
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TurnAction:
    """إجراء الدوران"""
    action: str
    
    def __str__(self):
        return f"{self.action}."


class OrientationReasoningGenerator:
    """مولد مجموعة بيانات الاستدلال الاتجاهي"""
    
    # تحديد 4 اتجاهات أساسية (بالترتيب مع عقارب الساعة)
    DIRECTIONS = ['شمال', 'شرق', 'جنوب', 'غرب']
    
    # تحديد إجراءات الدوران
    ACTIONS = ['استدر لليمين', 'استدر لليسار', 'استدر للخلف']
    
    # تحديد زاوية الدوران المقابلة لكل إجراء (بوحدة 90 درجة، مع عقارب الساعة موجب)
    ACTION_ROTATIONS = {
        'استدر لليمين': 1,      # 90 درجة مع عقارب الساعة
        'استدر لليسار': -1,     # 90 درجة عكس عقارب الساعة
        'استدر للخلف': 2        # 180 درجة
    }
    
    # تحديد الزاوية المقابلة لكل اتجاه (بالدرجات، بدءًا من الشرق عكس عقارب الساعة)
    # شرق: 0°، شمال: 90°، غرب: 180°، جنوب: 270°
    DIRECTION_ANGLES = {
        'شرق': 0,
        'شمال': 90,
        'غرب': 180,
        'جنوب': 270
    }
    
    def __init__(self, seed: Optional[int] = None):
        """تهيئة المولد"""
        if seed is not None:
            random.seed(seed)
    
    def encode_direction(self, direction: str) -> List[float]:
        """
        ترميز الاتجاه كـ [cos(θ), sin(θ)]
        
        Args:
            direction: سلسلة الاتجاه (شمال/شرق/جنوب/غرب)
            
        Returns:
            encoded: ترميز [cos(θ), sin(θ)]
        """
        angle_degrees = self.DIRECTION_ANGLES[direction]
        angle_radians = math.radians(angle_degrees)
        
        cos_val = math.cos(angle_radians)
        sin_val = math.sin(angle_radians)
        
        # معالجة مشكلة دقة الأعداد العشرية، تقريب إلى 10 منازل عشرية
        cos_val = round(cos_val, 10)
        sin_val = round(sin_val, 10)
        
        return [cos_val, sin_val]
    
    def apply_action(self, current_direction: str, action: str) -> str:
        """
        تطبيق إجراء الدوران
        
        Args:
            current_direction: الاتجاه الحالي
            action: إجراء الدوران
            
        Returns:
            new_direction: الاتجاه الجديد
        """
        # الحصول على فهرس الاتجاه الحالي
        current_index = self.DIRECTIONS.index(current_direction)
        
        # الحصول على عدد خطوات الدوران
        rotation = self.ACTION_ROTATIONS[action]
        
        # حساب فهرس الاتجاه الجديد
        new_index = (current_index + rotation) % 4
        
        return self.DIRECTIONS[new_index]
    
    def generate_action_sequence(self, num_steps: int) -> Tuple[List[TurnAction], str, str]:
        """
        توليد تسلسل إجراءات الدوران
        
        Args:
            num_steps: عدد خطوات الدوران (2-15)
            
        Returns:
            actions: قائمة إجراءات الدوران
            start_direction: الاتجاه الأولي
            end_direction: الاتجاه النهائي
        """
        # اختيار اتجاه أولي عشوائي
        start_direction = random.choice(self.DIRECTIONS)
        current_direction = start_direction
        
        # توليد تسلسل الإجراءات
        actions = []
        for _ in range(num_steps):
            # اختيار إجراء عشوائي
            action = random.choice(self.ACTIONS)
            actions.append(TurnAction(action=action))
            
            # تطبيق الإجراء
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
        التحقق مما إذا كانت المسألة تتطلب استدلالاً متعدد الخطوات
        
        Args:
            actions: قائمة إجراءات الدوران
            start_direction: الاتجاه الأولي
            end_direction: الاتجاه النهائي
            
        Returns:
            True if the question requires multi-step reasoning
        """
        # إذا كانت خطوة واحدة فقط، فلا حاجة للاستدلال متعدد الخطوات
        if len(actions) <= 1:
            return False
        
        # إذا كان الاتجاه النهائي هو نفس الاتجاه الأولي، فالمسألة بسيطة للغاية
        # لكن هذا ليس شرطًا ضروريًا، لأنه قد يعود إلى الاتجاه الأولي بعد عدة دورانات
        return True
    
    def generate_distractor_options(self, correct_answer: str) -> List[str]:
        """
        توليد خيارات مشتتة
        
        Args:
            correct_answer: الإجابة الصحيحة
            
        Returns:
            distractors: قائمة الخيارات المشتتة (الاتجاهات الثلاثة الأخرى)
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
        توليد السؤال والإجابة
        
        Args:
            actions: قائمة إجراءات الدوران
            start_direction: الاتجاه الأولي
            end_direction: الاتجاه النهائي
            
        Returns:
            question: سلسلة السؤال
            answer: سلسلة الإجابة
            options: قائمة الخيارات (ترتيب عشوائي للإجابة الصحيحة والخيارات المشتتة)
            correct_option: معرف الخيار الصحيح (A/B/C/D)
        """
        # بناء السؤال
        premise_lines = [f"أنت تواجه اتجاه {start_direction}."]
        premise_lines.extend([str(action) for action in actions])
        premise = "\n".join(premise_lines)
        question = f"{premise}\n\nفي أي اتجاه تواجه الآن؟"
        
        # الإجابة
        answer = end_direction
        
        # توليد خيارات مشتتة
        distractors = self.generate_distractor_options(answer)
        
        # دمج الإجابة الصحيحة والخيارات المشتتة وخلطها عشوائيًا
        all_options = [answer] + distractors
        random.shuffle(all_options)
        
        # إيجاد موقع الإجابة الصحيحة
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
        توليد مجموعة البيانات الكاملة
        
        Args:
            num_samples: عدد العينات
            min_steps: الحد الأدنى لخطوات الاستدلال
            max_steps: الحد الأقصى لخطوات الاستدلال
            output_file: مسار ملف الإخراج (اختياري)
            
        Returns:
            dataset: قائمة مجموعة البيانات
        """
        dataset = []
        
        for i in range(num_samples):
            # اختيار عدد خطوات الاستدلال عشوائيًا
            num_steps = random.randint(min_steps, max_steps)
            
            # توليد تسلسل الإجراءات
            actions, start_direction, end_direction = self.generate_action_sequence(num_steps)
            
            # التحقق مما إذا كان يتطلب استدلالاً
            if not self.verify_needs_reasoning(actions, start_direction, end_direction):
                # إذا لم يتطلب استدلالاً، أعد التوليد
                continue
            
            # توليد السؤال والإجابة
            question, answer, options, correct_option = self.generate_question_answer(
                actions, start_direction, end_direction
            )
            
            # ترميز الاتجاه النهائي كـ [cos(θ), sin(θ)]
            target_encoding = self.encode_direction(end_direction)
            
            # بناء عينة البيانات
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
            
            # طباعة التقدم
            if (i + 1) % 100 == 0:
                print(f"تم توليد {i + 1}/{num_samples} عينة...")
        
        # حفظ في ملف
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"\nتم حفظ مجموعة البيانات في {output_file}")
        
        return dataset
    
    def print_sample(self, sample: Dict):
        """طباعة مثال للعينة"""
        print(f"\n{'='*60}")
        print(f"معرف العينة: {sample['id']}")
        print(f"عدد خطوات الاستدلال: {sample['num_steps']}")
        print(f"\n{sample['question']}")
        print(f"\nالخيارات:")
        for i, option in enumerate(sample['options']):
            option_label = chr(65 + i)  # A, B, C, D
            mark = " ✓" if option_label == sample['correct_option'] else ""
            print(f"  {option_label}. {option}{mark}")
        print(f"\nالإجابة الصحيحة: {sample['correct_option']} ({sample['answer']})")
        print(f"البداية: {sample['start_direction']} -> النهاية: {sample['end_direction']}")
        print(f"الترميز المستهدف [cos(θ), sin(θ)]: {sample['target']}")
        print(f"{'='*60}\n")


def main():
    """الدالة الرئيسية"""
    # إنشاء المولد
    generator = OrientationReasoningGenerator(seed=42)
    
    # توليد مجموعة البيانات
    print("توليد مجموعة بيانات الاستدلال الاتجاهي (النسخة العربية)...")
    dataset = generator.generate_dataset(
        num_samples=2000,
        min_steps=2,
        max_steps=10,
        output_file='./data_orientation/orientation_reasoning_dataset_AR.json'
    )
    
    # طباعة بعض الأمثلة
    print("\n" + "="*60)
    print("أمثلة العينات:")
    print("="*60)
    
    # طباعة أمثلة لخطوات استدلال مختلفة
    for num_steps in [2, 5, 8, 10]:
        samples_with_n_steps = [s for s in dataset if s['num_steps'] == num_steps]
        if samples_with_n_steps:
            generator.print_sample(samples_with_n_steps[0])
    
    # معلومات إحصائية
    print("\n" + "="*60)
    print("إحصائيات مجموعة البيانات:")
    print("="*60)
    print(f"إجمالي العينات: {len(dataset)}")
    
    step_distribution = defaultdict(int)
    for sample in dataset:
        step_distribution[sample['num_steps']] += 1
    
    print("\nالتوزيع حسب عدد الخطوات:")
    for steps in sorted(step_distribution.keys()):
        print(f"  {steps} خطوة: {step_distribution[steps]} عينة")
    
    # إحصائيات توزيع الاتجاه النهائي
    direction_distribution = defaultdict(int)
    for sample in dataset:
        direction_distribution[sample['end_direction']] += 1
    
    print("\nالتوزيع حسب الاتجاه النهائي:")
    for direction in generator.DIRECTIONS:
        print(f"  {direction}: {direction_distribution[direction]} عينة")


if __name__ == "__main__":
    main()





