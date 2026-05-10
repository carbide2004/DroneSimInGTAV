import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _step_index(entry: Dict) -> int:
    if "step_index" not in entry:
        raise KeyError("Missing required field: step_index")
    return int(entry["step_index"])


def _trajectory_id(entry: Dict, fallback_index: int) -> str:
    del fallback_index
    if "trajectory_id" not in entry:
        raise KeyError("Missing required field: trajectory_id")
    value = str(entry["trajectory_id"]).strip()
    if not value:
        raise ValueError("trajectory_id must be non-empty")
    return value


def _pose_dict(entry: Dict) -> Dict[str, float]:
    pose = entry.get("pose")
    if not isinstance(pose, dict):
        raise KeyError("Missing required field: pose")
    return {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose["z"]),
        "rx": float(pose["rx"]),
        "ry": float(pose["ry"]),
        "rz": float(pose["rz"]),
    }


def group_by_trajectory(entries: Sequence[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for i, entry in enumerate(entries):
        traj = _trajectory_id(entry, i)
        groups.setdefault(traj, []).append(entry)
    for traj in groups:
        groups[traj] = sorted(groups[traj], key=_step_index)
    return groups


def filter_entries_by_max_trajectory_len(entries: Sequence[Dict], max_trajectory_len: int):
    groups = group_by_trajectory(entries)
    if int(max_trajectory_len) <= 0:
        return list(entries), {
            "max_trajectory_len": int(max_trajectory_len),
            "original_trajectories": len(groups),
            "filtered_trajectories": 0,
            "kept_trajectories": len(groups),
            "original_entries": len(entries),
            "filtered_entries": 0,
            "kept_entries": len(entries),
        }

    kept_ids = {tid for tid, items in groups.items() if len(items) <= int(max_trajectory_len)}
    filtered = [
        entry for i, entry in enumerate(entries)
        if _trajectory_id(entry, i) in kept_ids
    ]
    return filtered, {
        "max_trajectory_len": int(max_trajectory_len),
        "original_trajectories": len(groups),
        "filtered_trajectories": len(groups) - len(kept_ids),
        "kept_trajectories": len(kept_ids),
        "original_entries": len(entries),
        "filtered_entries": len(entries) - len(filtered),
        "kept_entries": len(filtered),
    }


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
    if not isinstance(observations, dict):
        raise KeyError("Missing required field: observations")
    rgb = observations.get("rgb")
    depth = observations.get("depth")
    if not isinstance(rgb, dict) or not isinstance(depth, dict):
        raise KeyError("Missing required field: observations.rgb/depth")
    rgb_path = str(rgb["path"]).strip()
    depth_path = str(depth["path"]).strip()
    if not rgb_path or not depth_path:
        raise ValueError("observations.rgb.path/depth.path must be non-empty")
    return rgb_path, depth_path


def _extract_action(entry: Dict) -> Tuple[Optional[str], int]:
    action = entry.get("action")
    if not isinstance(action, dict):
        raise KeyError("Missing required field: action")
    name = str(action["name"]).strip()
    if not name:
        raise ValueError("action.name must be non-empty")
    action_id = int(entry["action_id"])
    return name, action_id


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
                pose = _pose_dict(entry)
                samples.append({
                    "trajectory_id": tid,
                    "step_index": _step_index(entry),
                    "trajectory_length": int(entry.get("trajectory_length", len(trajectory_entries))),
                    "sample_id": str(entry.get("sample_id", f"{tid}:{_step_index(entry):06d}")),
                    "task": entry.get("task"),
                    "pose": pose,
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
            pose = _pose_dict(entry)
            steps.append({
                "step_index": _step_index(entry),
                "sample_id": str(entry.get("sample_id", f"{trajectory_id}:{_step_index(entry):06d}")),
                "pose": pose,
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
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise RuntimeError(f"第 {i} 条样本不是对象")
        schema_version = entry.get("schema_version")
        if int(schema_version) != 2:
            raise RuntimeError(f"第 {i} 条样本 schema_version 不是 2")
        _trajectory_id(entry, i)
        _step_index(entry)
        _resolve_image_paths(entry)
        _extract_action(entry)
        _pose_dict(entry)
    return data


def _build_datasets(entries: List[Dict], train_ids: List[str], val_ids: List[str], mode: str):
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
    return train_dataset, val_dataset, collate_fn


def _build_loaders(train_dataset, val_dataset, collate_fn, batch_size: int, num_workers: int):
    try:
        from torch.utils.data import DataLoader
    except Exception as e:
        raise RuntimeError("无法导入 torch.utils.data.DataLoader，请先安装 PyTorch。") from e
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
    return train_loader, val_loader


def build_stage1_dataloaders_from_manifest(
    dataset_json: Path,
    split_manifest_json: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    mode: str = "sequence",
    max_trajectory_len: int = 0,
):
    entries = load_dataset_entries(Path(dataset_json))
    entries, filter_meta = filter_entries_by_max_trajectory_len(entries, int(max_trajectory_len))
    if not entries:
        raise RuntimeError(f"No trajectories left after filtering with max_trajectory_len={int(max_trajectory_len)}")
    manifest = _read_json(Path(split_manifest_json))
    if not isinstance(manifest, dict):
        raise RuntimeError("split manifest 必须是对象。")
    train_ids = [str(x) for x in manifest.get("train_trajectory_ids", [])]
    val_ids = [str(x) for x in manifest.get("val_trajectory_ids", [])]
    train_dataset, val_dataset, collate_fn = _build_datasets(entries, train_ids, val_ids, mode=mode)
    train_loader, val_loader = _build_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
    )
    split_meta = {
        "source": str(Path(dataset_json)),
        "manifest": str(Path(split_manifest_json)),
        "total_trajectories": int(manifest.get("total_trajectories", len(train_ids) + len(val_ids))),
        "train_trajectories": len(train_ids),
        "val_trajectories": len(val_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "mode": mode,
        "batch_size": int(batch_size),
        "split_strategy": "fixed_manifest",
        "filter": filter_meta,
    }
    return train_loader, val_loader, split_meta


def build_stage1_dataloaders_from_split_json(
    train_json: Path,
    val_json: Path,
    batch_size: int = 4,
    num_workers: int = 0,
    mode: str = "sequence",
    max_trajectory_len: int = 0,
):
    train_entries = load_dataset_entries(Path(train_json))
    val_entries = load_dataset_entries(Path(val_json))
    train_entries, train_filter_meta = filter_entries_by_max_trajectory_len(train_entries, int(max_trajectory_len))
    val_entries, val_filter_meta = filter_entries_by_max_trajectory_len(val_entries, int(max_trajectory_len))
    if not train_entries:
        raise RuntimeError(f"No train trajectories left after filtering with max_trajectory_len={int(max_trajectory_len)}")
    if not val_entries:
        raise RuntimeError(f"No val trajectories left after filtering with max_trajectory_len={int(max_trajectory_len)}")
    merged_entries = list(train_entries) + list(val_entries)
    train_groups = group_by_trajectory(train_entries)
    val_groups = group_by_trajectory(val_entries)
    train_ids = sorted(train_groups.keys())
    val_ids = sorted(val_groups.keys())
    train_dataset, val_dataset, collate_fn = _build_datasets(
        entries=merged_entries,
        train_ids=train_ids,
        val_ids=val_ids,
        mode=mode,
    )
    train_loader, val_loader = _build_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
    )
    split_meta = {
        "train_json": str(Path(train_json)),
        "val_json": str(Path(val_json)),
        "train_trajectories": len(train_ids),
        "val_trajectories": len(val_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "mode": mode,
        "batch_size": int(batch_size),
        "split_strategy": "fixed_json",
        "filter": {
            "max_trajectory_len": int(max_trajectory_len),
            "train": train_filter_meta,
            "val": val_filter_meta,
        },
    }
    return train_loader, val_loader, split_meta


def build_stage1_dataloaders(
    dataset_json: Path,
    batch_size: int = 4,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    mode: str = "sequence",
    max_trajectory_len: int = 0,
):
    entries = load_dataset_entries(Path(dataset_json))
    entries, filter_meta = filter_entries_by_max_trajectory_len(entries, int(max_trajectory_len))
    if not entries:
        raise RuntimeError(f"No trajectories left after filtering with max_trajectory_len={int(max_trajectory_len)}")
    groups = group_by_trajectory(entries)
    all_ids = sorted(groups.keys())
    train_ids, val_ids = split_trajectory_ids(all_ids, val_ratio=val_ratio, seed=seed)
    train_dataset, val_dataset, collate_fn = _build_datasets(entries, train_ids, val_ids, mode=mode)
    train_loader, val_loader = _build_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
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
        "filter": filter_meta,
    }
    return train_loader, val_loader, split_meta
