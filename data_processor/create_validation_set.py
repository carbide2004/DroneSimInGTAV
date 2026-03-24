#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证数据集创建脚本
从 data/manual 中随机选择20%的样本作为验证数据，添加到 data/verification/sample.jsonl
"""

import os
import json
import random
import shutil
from pathlib import Path

def scan_manual_data(manual_dir):
    """扫描manual目录下的所有样本"""
    samples = []
    manual_path = Path(manual_dir)
    
    if not manual_path.exists():
        print(f"错误：目录 {manual_dir} 不存在")
        return samples
    
    for session_dir in manual_path.iterdir():
        if session_dir.is_dir():
            steps_file = session_dir / "steps.jsonl"
            metadata_file = session_dir / "metadata.jsonl"
            
            # 检查必要文件是否存在
            if steps_file.exists() and metadata_file.exists():
                samples.append({
                    'session_id': session_dir.name,
                    'session_path': str(session_dir),
                    'steps_file': str(steps_file),
                    'metadata_file': str(metadata_file)
                })
                print(f"找到样本：{session_dir.name}")
    
    return samples

def load_metadata(metadata_file):
    """加载元数据文件"""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.loads(f.read().strip())
    except Exception as e:
        print(f"警告：无法读取元数据文件 {metadata_file}: {e}")
        return {}

def create_validation_entry(sample):
    """为样本创建验证数据条目"""
    metadata = load_metadata(sample['metadata_file'])
    
    # 构建验证数据条目
    validation_entry = {
        'session_id': sample['session_id'],
        'session_path': sample['session_path'],
        'steps_file': sample['steps_file'],
        'metadata_file': sample['metadata_file']
    }
    
    # 添加元数据信息（如果存在）
    if metadata:
        validation_entry.update(metadata)
    
    return validation_entry

def main():
    """主函数"""
    # 设置路径
    manual_dir = "data/manual"
    verification_dir = "data/verification"
    validation_file = os.path.join(verification_dir, "sample.jsonl")
    
    # 创建验证目录
    os.makedirs(verification_dir, exist_ok=True)
    
    # 扫描所有样本
    print("正在扫描训练数据...")
    samples = scan_manual_data(manual_dir)
    
    if not samples:
        print("未找到任何样本数据")
        return
    
    print(f"总共找到 {len(samples)} 个样本")
    
    # 随机选择20%的样本
    validation_count = max(1, int(len(samples) * 0.2))
    selected_samples = random.sample(samples, validation_count)
    
    print(f"随机选择了 {len(selected_samples)} 个样本作为验证数据（{len(selected_samples)/len(samples)*100:.1f}%）")
    
    # 写入验证数据文件
    with open(validation_file, 'w', encoding='utf-8') as f:
        for sample in selected_samples:
            validation_entry = create_validation_entry(sample)
            f.write(json.dumps(validation_entry, ensure_ascii=False) + '\n')
            print(f"添加验证样本：{sample['session_id']}")
    
    print(f"\n验证数据集创建完成！")
    print(f"验证数据文件：{validation_file}")
    print(f"验证样本数量：{len(selected_samples)}")
    print(f"训练样本数量：{len(samples) - len(selected_samples)}")

if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    random.seed(42)
    main()