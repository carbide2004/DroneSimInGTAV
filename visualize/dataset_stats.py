"""
数据集统计脚本，用于填充论文表3-2（总体统计）和表3-3（动作分布）。

用法：
    python dataset_stats.py --input_json ../dataset/train_data_all_with_awareness.json
    
    可选：指定划分manifest文件一并统计训练/验证集信息
    python dataset_stats.py \
        --input_json ../dataset/train_data_all_with_awareness.json \
        --manifest_json ../dataset/train_val_split_manifest.json
"""

import argparse
import json
from pathlib import Path
from collections import Counter


ACTION_NAMES = [
    "AUTO_DOWN",
    "AUTO_UP",
    "AUTO_FORWARD",
    "AUTO_YAW_LEFT",
    "AUTO_YAW_RIGHT",
    "AUTO_STOP_REACHED",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="数据集统计（论文表3-2、表3-3）")
    parser.add_argument("--input_json", required=True, help="数据集JSON文件路径")
    parser.add_argument("--manifest_json", default=None, help="训练/验证划分manifest文件路径（可选）")
    args = parser.parse_args()

    data = load_json(args.input_json)
    assert isinstance(data, list), "数据集JSON必须是list"

    # ---- 按轨迹分组 ----
    traj_groups = {}
    for entry in data:
        tid = str(entry["trajectory_id"])
        traj_groups.setdefault(tid, []).append(entry)

    traj_lengths = [len(steps) for steps in traj_groups.values()]
    total_trajs = len(traj_groups)
    total_samples = len(data)
    avg_len = sum(traj_lengths) / total_trajs if total_trajs else 0
    min_len = min(traj_lengths) if traj_lengths else 0
    max_len = max(traj_lengths) if traj_lengths else 0

    # ---- 异常事件类型统计 ----
    task_set = set()
    for entry in data:
        task = entry.get("task", "")
        if isinstance(task, str) and task.strip():
            task_set.add(task.strip())

    # ---- awareness覆盖率 ----
    awareness_count = sum(1 for e in data if "awareness" in e and e["awareness"])
    awareness_rate = awareness_count / total_samples * 100 if total_samples else 0

    # ---- 动作分布 ----
    action_counter = Counter()
    for entry in data:
        action = entry.get("action", {})
        name = action.get("name", "") if isinstance(action, dict) else ""
        action_counter[name] += 1

    # ---- 训练/验证划分 ----
    manifest = None
    if args.manifest_json and Path(args.manifest_json).exists():
        manifest = load_json(args.manifest_json)

    # ============================================================
    # 输出：表 3-2 数据集总体统计
    # ============================================================
    print("=" * 60)
    print("表 3-2  数据集总体统计")
    print("=" * 60)
    print(f"  轨迹总数               {total_trajs}")
    print(f"  样本总数（帧）         {total_samples}")
    print(f"  平均轨迹长度（步）     {avg_len:.1f}")
    print(f"  最短轨迹长度           {min_len}")
    print(f"  最长轨迹长度           {max_len}")

    if manifest:
        n_train = manifest.get("train_trajectories", "?")
        n_val = manifest.get("val_trajectories", "?")
        m_train = manifest.get("train_entries", "?")
        m_val = manifest.get("val_entries", "?")
        print(f"  训练集轨迹数 / 样本数  {n_train} / {m_train}")
        print(f"  验证集轨迹数 / 样本数  {n_val} / {m_val}")
    else:
        # 没有manifest时用20%估算
        train_ids_count = int(total_trajs * 0.8)
        val_ids_count = total_trajs - train_ids_count
        print(f"  训练集轨迹数（估算80%）{train_ids_count}")
        print(f"  验证集轨迹数（估算20%）{val_ids_count}")

    print(f"  任务描述种类           {len(task_set)}")
    for t in sorted(task_set):
        print(f"    - {t}")
    print(f"  视觉特征维度           1728")
    print(f"  \"意识\"标注覆盖率      {awareness_rate:.1f}% ({awareness_count}/{total_samples})")

    # ============================================================
    # 输出：表 3-3 动作分布统计
    # ============================================================
    print()
    print("=" * 60)
    print("表 3-3  数据集中各动作的分布")
    print("=" * 60)
    print(f"  {'动作名称':<25s} {'出现次数':>8s} {'占比':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    for name in ACTION_NAMES:
        count = action_counter.get(name, 0)
        ratio = count / total_samples * 100 if total_samples else 0
        print(f"  {name:<25s} {count:>8d} {ratio:>7.1f}%")

    # 检查是否有未知动作
    unknown = {k: v for k, v in action_counter.items() if k not in ACTION_NAMES}
    if unknown:
        for name, count in unknown.items():
            ratio = count / total_samples * 100
            print(f"  {name or '(空)':<25s} {count:>8d} {ratio:>7.1f}%")

    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    print(f"  {'合计':<25s} {total_samples:>8d} {'100.0%':>8s}")

    # ============================================================
    # 输出：可直接粘贴到LaTeX的表格内容
    # ============================================================
    print()
    print("=" * 60)
    print("LaTeX 表 3-2 数据行（可直接粘贴）")
    print("=" * 60)
    print(f"        轨迹总数 & {total_trajs} \\\\")
    print(f"        样本总数（帧） & {total_samples} \\\\")
    print(f"        平均轨迹长度（步） & {avg_len:.1f} \\\\")
    print(f"        最短轨迹长度 & {min_len} \\\\")
    print(f"        最长轨迹长度 & {max_len} \\\\")
    if manifest:
        print(f"        训练集轨迹数 / 样本数 & {n_train} / {m_train} \\\\")
        print(f"        验证集轨迹数 / 样本数 & {n_val} / {m_val} \\\\")
    print(f"        异常事件类型 & 火灾、斗殴、交通事故 \\\\")
    print(f"        视觉特征维度 & 1728 \\\\")
    print(f"        \"意识\"标注覆盖率 & {awareness_rate:.0f}\\% \\\\")

    print()
    print("=" * 60)
    print("LaTeX 表 3-3 数据行（可直接粘贴）")
    print("=" * 60)
    for name in ACTION_NAMES:
        count = action_counter.get(name, 0)
        ratio = count / total_samples * 100 if total_samples else 0
        name_escaped = name.replace("_", "\\_")
        print(f"        {name_escaped} & {count} & {ratio:.1f}\\% \\\\")
    print(f"        \\hline")
    print(f"        合计 & {total_samples} & 100\\% \\\\")


if __name__ == "__main__":
    main()
