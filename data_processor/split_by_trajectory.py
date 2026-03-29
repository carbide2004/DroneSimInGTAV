import argparse
import json
import random
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _trajectory_id(entry):
    if "trajectory_id" not in entry:
        raise KeyError("Missing required field: trajectory_id")
    value = str(entry["trajectory_id"]).strip()
    if not value:
        raise ValueError("trajectory_id must be non-empty")
    return value


def _group_by_trajectory(entries):
    groups = {}
    for entry in entries:
        tid = _trajectory_id(entry)
        groups.setdefault(tid, []).append(entry)
    return groups


def _split_trajectory_ids(trajectory_ids, val_ratio, seed):
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


def main():
    parser = argparse.ArgumentParser(description="Split train_data_all.json by trajectory_id")
    parser.add_argument(
        "--input_json",
        default=str(_repo_root() / "dataset" / "train_data_all.json"),
        help="Input JSON path",
    )
    parser.add_argument(
        "--train_output",
        default=str(_repo_root() / "dataset" / "train_data_train.json"),
        help="Train split output JSON path",
    )
    parser.add_argument(
        "--val_output",
        default=str(_repo_root() / "dataset" / "train_data_val.json"),
        help="Validation split output JSON path",
    )
    parser.add_argument(
        "--manifest_output",
        default=str(_repo_root() / "dataset" / "train_val_split_manifest.json"),
        help="Split manifest JSON path",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation ratio by trajectories",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = _read_json(input_path)
    if not isinstance(data, list):
        raise RuntimeError("Input JSON must be a list.")

    groups = _group_by_trajectory(data)
    trajectory_ids = sorted(groups.keys())
    train_ids, val_ids = _split_trajectory_ids(trajectory_ids, args.val_ratio, args.seed)

    train_data = []
    for tid in train_ids:
        train_data.extend(groups[tid])
    val_data = []
    for tid in val_ids:
        val_data.extend(groups[tid])

    train_path = Path(args.train_output)
    val_path = Path(args.val_output)
    manifest_path = Path(args.manifest_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    _write_json(train_path, train_data)
    _write_json(val_path, val_data)

    manifest = {
        "source": str(input_path),
        "total_entries": len(data),
        "total_trajectories": len(trajectory_ids),
        "train_entries": len(train_data),
        "val_entries": len(val_data),
        "train_trajectories": len(train_ids),
        "val_trajectories": len(val_ids),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "train_trajectory_ids": train_ids,
        "val_trajectory_ids": val_ids,
    }
    _write_json(manifest_path, manifest)

    print(f"Total trajectories: {len(trajectory_ids)}")
    print(f"Train trajectories: {len(train_ids)}, entries: {len(train_data)}")
    print(f"Val trajectories: {len(val_ids)}, entries: {len(val_data)}")
    print(f"Train output: {train_path}")
    print(f"Val output: {val_path}")
    print(f"Manifest output: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
