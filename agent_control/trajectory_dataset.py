import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _step_index(entry: Dict) -> int:
    step = entry.get("step_index")
    if step is None:
        return 0
    return int(step)


def _trajectory_id(entry: Dict, fallback_index: int) -> str:
    value = entry.get("trajectory_id")
    if value is None:
        return f"__unknown_{fallback_index}__"
    return str(value)


def group_by_trajectory(entries: Sequence[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for i, entry in enumerate(entries):
        traj = _trajectory_id(entry, i)
        groups.setdefault(traj, []).append(entry)
    for traj in groups:
        groups[traj] = sorted(groups[traj], key=_step_index)
    return groups


def split_trajectory_ids(trajectory_ids: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = list(trajectory_ids)
    random.Random(seed).shuffle(ids)
    if not ids:
        return [], []
    if val_ratio <= 0:
        return ids, []
    val_count = max(1, int(round(len(ids) * float(val_ratio))))
    val_ids = ids[:val_count]
    train_ids = ids[val_count:]
    if not train_ids and val_ids:
        train_ids = [val_ids.pop()]
    return train_ids, val_ids


def _resolve_image_paths(entry: Dict) -> Tuple[Optional[str], Optional[str]]:
    observations = entry.get("observations")
    if isinstance(observations, dict):
        rgb = observations.get("rgb") or {}
        depth = observations.get("depth") or {}
        rgb_path = rgb.get("path")
        depth_path = depth.get("path")
        if rgb_path and depth_path:
            return str(rgb_path), str(depth_path)
    images = entry.get("images") or []
    if len(images) < 2:
        return None, None
    return str(images[0]), str(images[1])


def _extract_action(entry: Dict) -> Tuple[Optional[str], int]:
    action = entry.get("action")
    if isinstance(action, dict):
        name = action.get("name")
    else:
        name = action

    if not name:
        messages = entry.get("messages") or []
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                raw = str(msg.get("content", "")).strip()
                if raw:
                    name = raw
                    break

    action_id = entry.get("action_id")
    if action_id is None:
        action_id = -1
    return (str(name) if name else None), int(action_id)


class TrajectoryStepDataset:
    def __init__(self, entries: Sequence[Dict], trajectory_ids: Optional[Sequence[str]] = None):
        groups = group_by_trajectory(entries)
        if trajectory_ids is None:
            selected_ids = sorted(groups.keys())
        else:
            selected_ids = [str(tid) for tid in trajectory_ids if str(tid) in groups]

        samples = []
        for tid in selected_ids:
            trajectory_entries = groups[tid]
            for entry in trajectory_entries:
                rgb_path, depth_path = _resolve_image_paths(entry)
                action_name, action_id = _extract_action(entry)
                pose = entry.get("pose") or {}
                samples.append({
                    "trajectory_id": tid,
                    "step_index": _step_index(entry),
                    "trajectory_length": int(entry.get("trajectory_length", len(trajectory_entries))),
                    "sample_id": str(entry.get("sample_id", f"{tid}:{_step_index(entry):06d}")),
                    "task": entry.get("task"),
                    "pose": {
                        "x": float(pose.get("x", 0.0)),
                        "y": float(pose.get("y", 0.0)),
                        "z": float(pose.get("z", 0.0)),
                        "rx": float(pose.get("rx", 0.0)),
                        "ry": float(pose.get("ry", 0.0)),
                        "rz": float(pose.get("rz", 0.0)),
                    },
                    "action_name": action_name,
                    "action_id": int(action_id),
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "awareness": entry.get("awareness"),
                })
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class TrajectorySequenceDataset:
    def __init__(self, entries: Sequence[Dict], trajectory_ids: Optional[Sequence[str]] = None):
        groups = group_by_trajectory(entries)
        if trajectory_ids is None:
            selected_ids = sorted(groups.keys())
        else:
            selected_ids = [str(tid) for tid in trajectory_ids if str(tid) in groups]
        self.trajectory_ids = selected_ids
        self.groups = groups

    def __len__(self):
        return len(self.trajectory_ids)

    def __getitem__(self, index: int):
        trajectory_id = self.trajectory_ids[index]
        entries = self.groups[trajectory_id]
        steps = []
        for entry in entries:
            rgb_path, depth_path = _resolve_image_paths(entry)
            action_name, action_id = _extract_action(entry)
            pose = entry.get("pose") or {}
            steps.append({
                "step_index": _step_index(entry),
                "sample_id": str(entry.get("sample_id", f"{trajectory_id}:{_step_index(entry):06d}")),
                "pose": {
                    "x": float(pose.get("x", 0.0)),
                    "y": float(pose.get("y", 0.0)),
                    "z": float(pose.get("z", 0.0)),
                    "rx": float(pose.get("rx", 0.0)),
                    "ry": float(pose.get("ry", 0.0)),
                    "rz": float(pose.get("rz", 0.0)),
                },
                "action_name": action_name,
                "action_id": int(action_id),
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "awareness": entry.get("awareness"),
            })

        return {
            "trajectory_id": trajectory_id,
            "trajectory_length": int(entries[0].get("trajectory_length", len(entries))) if entries else 0,
            "task": entries[0].get("task") if entries else None,
            "steps": steps,
        }


def collate_trajectory_sequences(batch: Sequence[Dict]):
    try:
        import torch
    except Exception as e:
        raise RuntimeError("无法导入 torch，请先安装 PyTorch。") from e

    lengths = [len(item["steps"]) for item in batch]
    max_len = max(lengths) if lengths else 0
    action_ids = torch.full((len(batch), max_len), fill_value=-1, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    rgb_paths = []
    depth_paths = []
    trajectory_ids = []
    tasks = []

    for i, item in enumerate(batch):
        trajectory_ids.append(item["trajectory_id"])
        tasks.append(item.get("task"))
        rgb_seq = []
        depth_seq = []
        for j, step in enumerate(item["steps"]):
            action_ids[i, j] = int(step["action_id"])
            mask[i, j] = True
            rgb_seq.append(step.get("rgb_path"))
            depth_seq.append(step.get("depth_path"))
        rgb_paths.append(rgb_seq)
        depth_paths.append(depth_seq)

    return {
        "trajectory_ids": trajectory_ids,
        "tasks": tasks,
        "lengths": lengths,
        "action_ids": action_ids,
        "mask": mask,
        "rgb_paths": rgb_paths,
        "depth_paths": depth_paths,
        "raw_batch": list(batch),
    }


def load_dataset_entries(dataset_json: Path) -> List[Dict]:
    data = _read_json(Path(dataset_json))
    if not isinstance(data, list):
        raise RuntimeError("数据集 JSON 必须是 list。")
    return data


def build_stage1_dataloaders(
    dataset_json: Path,
    batch_size: int = 4,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    mode: str = "sequence",
):
    try:
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError("无法导入 torch.utils.data.DataLoader，请先安装 PyTorch。") from e

    entries = load_dataset_entries(Path(dataset_json))
    groups = group_by_trajectory(entries)
    all_ids = sorted(groups.keys())
    train_ids, val_ids = split_trajectory_ids(all_ids, val_ratio=val_ratio, seed=seed)

    if mode == "sequence":
        train_dataset = TrajectorySequenceDataset(entries, train_ids)
        val_dataset = TrajectorySequenceDataset(entries, val_ids)
        collate_fn = collate_trajectory_sequences
    elif mode == "step":
        train_dataset = TrajectoryStepDataset(entries, train_ids)
        val_dataset = TrajectoryStepDataset(entries, val_ids)
        collate_fn = None
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_fn,
    )
    split_meta = {
        "total_trajectories": len(all_ids),
        "train_trajectories": len(train_ids),
        "val_trajectories": len(val_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "mode": mode,
        "batch_size": int(batch_size),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
    }
    return train_loader, val_loader, split_meta
