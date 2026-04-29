import argparse
import json
import shutil
from pathlib import Path


def get_last_action_name(steps_jsonl_path):
    if not steps_jsonl_path.exists():
        return None

    with steps_jsonl_path.open("r", encoding="utf-8") as f:
        last_line = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            last_line = line

    if last_line is None:
        return None

    try:
        last_record = json.loads(last_line)
    except json.JSONDecodeError:
        raise ValueError(f"无法解析 JSON 行: {last_line[:80]}...")

    return last_record.get("action", {}).get("name")


def ensure_unique_target(target_dir):
    if not target_dir.exists():
        return target_dir

    base_name = target_dir.name
    parent = target_dir.parent
    counter = 1
    while True:
        candidate = parent / f"{base_name}_{counter:02d}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_failed_trajectories(root_dir, failed_folder_name="failed", failed_action="AUTO_STOP_FAILED", dry_run=False):
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"指定目录不存在: {root_path}")

    failed_dir = root_path / failed_folder_name
    failed_dir.mkdir(exist_ok=True)

    moved = 0
    skipped = 0

    for session_dir in sorted(root_path.iterdir()):
        if not session_dir.is_dir() or session_dir.name == failed_folder_name:
            continue

        steps_path = session_dir / "steps.jsonl"
        if not steps_path.exists():
            skipped += 1
            print(f"跳过: {session_dir.name}，未找到 steps.jsonl")
            continue

        try:
            action_name = get_last_action_name(steps_path)
        except ValueError as e:
            skipped += 1
            print(f"跳过: {session_dir.name}，解析失败: {e}")
            continue

        if action_name == failed_action:
            target_dir = failed_dir / session_dir.name
            target_dir = ensure_unique_target(target_dir)
            print(f"移动: {session_dir.name} -> {target_dir}")
            if not dry_run:
                shutil.move(str(session_dir), str(target_dir))
            moved += 1
        else:
            skipped += 1

    print("\n处理完成")
    print(f"已移动: {moved}")
    print(f"未移动: {skipped}")


def parse_args():
    parser = argparse.ArgumentParser(description="将最后一帧动作为 AUTO_STOP_FAILED 的轨迹移动到 failed 目录")
    parser.add_argument("root_dir", help="轨迹所在根目录，例如 data/manual")
    parser.add_argument("--failed-folder", default="failed", help="存放失败轨迹的子目录名，默认为 failed")
    parser.add_argument("--failed-action", default="AUTO_STOP_FAILED", help="判定失败轨迹的动作名，默认为 AUTO_STOP_FAILED")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要移动的轨迹，不实际执行移动")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    move_failed_trajectories(
        root_dir=args.root_dir,
        failed_folder_name=args.failed_folder,
        failed_action=args.failed_action,
        dry_run=args.dry_run,
    )
