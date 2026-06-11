#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从已审查轨迹的 metadata.jsonl 生成在线/离线验证样本。"""

import argparse
import json
import random
from pathlib import Path


REQUIRED_FIELDS = (
    "scenario_id",
    "anomaly_type",
    "anomaly_position",
    "start_pose",
    "expected_steps",
    "task_description",
)


def scan_manual_data(manual_dir):
    """扫描轨迹目录下的所有 session。"""
    samples = []
    manual_path = Path(manual_dir)

    if not manual_path.exists():
        print(f"错误：目录 {manual_dir} 不存在")
        return samples

    for session_dir in sorted(manual_path.iterdir()):
        if not session_dir.is_dir():
            continue
        steps_file = session_dir / "steps.jsonl"
        metadata_file = session_dir / "metadata.jsonl"
        if steps_file.exists() and metadata_file.exists():
            samples.append({
                "session_id": session_dir.name,
                "session_path": str(session_dir),
                "steps_file": str(steps_file),
                "metadata_file": str(metadata_file),
            })
            print(f"找到样本：{session_dir.name}")

    return samples


def load_metadata(metadata_file):
    """读取 metadata.jsonl 的第一条非空 JSON 行。"""
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
    except Exception as e:
        print(f"警告：无法读取元数据文件 {metadata_file}: {e}")
    return {}


def validate_verification_entry(entry, metadata_file):
    missing = [key for key in REQUIRED_FIELDS if key not in entry]
    if missing:
        raise ValueError(f"{metadata_file} 缺少字段: {missing}")


def create_validation_entry(sample):
    """为单条采集 session 创建验证样本。"""
    metadata = load_metadata(sample["metadata_file"])
    validation_entry = {
        "session_id": sample["session_id"],
        "session_path": sample["session_path"],
        "steps_file": sample["steps_file"],
        "metadata_file": sample["metadata_file"],
    }
    if metadata:
        validation_entry.update(metadata)

    validate_verification_entry(validation_entry, sample["metadata_file"])
    return validation_entry


def parse_args():
    parser = argparse.ArgumentParser(description="从已审查轨迹生成 data/verification/samples.jsonl")
    parser.add_argument(
        "--manual_dir",
        required=True,
        help="已审查轨迹目录，例如 <GTA V>/data/manual/checked",
    )
    parser.add_argument(
        "--output_file",
        default=str(Path("data") / "verification" / "samples.jsonl"),
        help="输出验证样本 JSONL，默认 data/verification/samples.jsonl",
    )
    parser.add_argument("--ratio", type=float, default=0.2, help="抽样比例，默认 0.2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--limit", type=int, default=0, help="最多输出多少条；0 表示不限制")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(int(args.seed))

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("正在扫描轨迹数据...")
    samples = scan_manual_data(args.manual_dir)
    if not samples:
        print("未找到任何样本数据")
        return 1

    print(f"总共找到 {len(samples)} 个样本")
    sample_count = max(1, int(round(len(samples) * float(args.ratio))))
    if int(args.limit) > 0:
        sample_count = min(sample_count, int(args.limit))
    selected_samples = random.sample(samples, sample_count)
    print(f"随机选择了 {len(selected_samples)} 个样本作为验证数据（{len(selected_samples) / len(samples) * 100:.1f}%）")

    written = 0
    with open(output_file, "w", encoding="utf-8", newline="\n") as f:
        for sample in selected_samples:
            try:
                validation_entry = create_validation_entry(sample)
            except ValueError as e:
                print(f"跳过验证样本：{sample['session_id']}，原因：{e}")
                continue
            f.write(json.dumps(validation_entry, ensure_ascii=False) + "\n")
            print(f"添加验证样本：{sample['session_id']}")
            written += 1

    print("\n验证数据集创建完成！")
    print(f"验证数据文件：{output_file}")
    print(f"验证样本数量：{written}")
    print(f"未抽中训练样本数量：{len(samples) - len(selected_samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
