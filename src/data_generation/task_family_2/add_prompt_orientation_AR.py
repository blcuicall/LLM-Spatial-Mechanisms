#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إضافة تعليمات للبيانات الخاصة بتحويل المنظور (النسخة العربية)
"""

INSTRUCTION_TEMPLATE = """
سيتم إعطاؤك سلسلة من إجراءات الدوران.
بدءًا من اتجاه أولي، تحتاج إلى تتبع التغييرات في الاتجاه وتحديد الاتجاه النهائي.

الاتجاه الأولي والإجراءات:
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
لا تقدم أي تفسير أو خطوات استدلال أو نص إضافي.
""".strip()

import json

def parse_question(question_text):
    """تحليل حقل السؤال لاستخراج العبارات والسؤال"""
    lines = question_text.strip().split('\n')
    statements_lines = []
    question_line = ""
    
    for line in lines:
        if line.startswith("في أي اتجاه"):
            question_line = line
        else:
            statements_lines.append(line)
    
    statements = '\n'.join(statements_lines)
    return statements, question_line

def construct_prompt(sample):
    """بناء التعليمات الكاملة"""
    # تحليل حقل السؤال
    statements, question = parse_question(sample['question'])
    
    # الحصول على الخيارات
    options = sample['options']
    option_A = options[0]
    option_B = options[1]
    option_C = options[2]
    option_D = options[3]
    
    # بناء التعليمات
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
    إضافة تعليمات إلى مجموعة البيانات
    
    Args:
        input_file: ملف مجموعة البيانات المدخل
        output_file: ملف مجموعة البيانات المخرج مع التعليمات
    """
    # قراءة مجموعة البيانات الأصلية
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    # إضافة تعليمات لكل عينة
    for sample in dataset:
        prompt = construct_prompt(sample)
        sample['prompt'] = prompt
    
    # حفظ مجموعة البيانات مع التعليمات
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"تمت إضافة تعليمات إلى {len(dataset)} عينة")
    print(f"تم حفظ مجموعة البيانات مع التعليمات في {output_file}")


def main():
    """الدالة الرئيسية"""
    import os
    
    # فحص ومعالجة مجموعة البيانات الكاملة
    if os.path.exists('./data_orientation/orientation_reasoning_dataset_AR.json'):
        print("إضافة تعليمات لمجموعة البيانات الكاملة...")
        add_prompt_to_dataset(
            input_file='./data_orientation/orientation_reasoning_dataset_AR.json',
            output_file='./data_orientation/orientation_reasoning_dataset_AR_with_prompt.json'
        )
    else:
        print("لم يتم العثور على مجموعة البيانات الكاملة، تخطي...")
    
    # فحص ومعالجة مجموعة بيانات الاختبار
    if os.path.exists('./data_orientation/orientation_reasoning_dataset_AR_test.json'):
        print("\nإضافة تعليمات لمجموعة بيانات الاختبار...")
        add_prompt_to_dataset(
            input_file='./data_orientation/orientation_reasoning_dataset_AR_test.json',
            output_file='./data_orientation/orientation_reasoning_dataset_AR_test_with_prompt.json'
        )
    else:
        print("لم يتم العثور على مجموعة بيانات الاختبار، تخطي...")
    
    # طباعة مثال (أولوية لمجموعة البيانات الكاملة)
    sample_file = None
    if os.path.exists('./data_orientation/orientation_reasoning_dataset_AR_with_prompt.json'):
        sample_file = './data_orientation/orientation_reasoning_dataset_AR_with_prompt.json'
    elif os.path.exists('./data_orientation/orientation_reasoning_dataset_AR_test_with_prompt.json'):
        sample_file = './data_orientation/orientation_reasoning_dataset_AR_test_with_prompt.json'
    
    if sample_file:
        with open(sample_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print("\n" + "="*60)
        print("مثال (مع التعليمات):")
        print("="*60)
        print(dataset[0]['prompt'])
        print("\n" + "="*60)
        print(f"الإجابة الصحيحة: {dataset[0]['correct_option']} ({dataset[0]['answer']})")
        print("="*60)


if __name__ == "__main__":
    main()





