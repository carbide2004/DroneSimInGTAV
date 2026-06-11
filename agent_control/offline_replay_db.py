import argparse
import io
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


SCHEMA_VERSION = 1


@dataclass
class ReplayState:
    state_id: int
    sample_id: str
    trajectory_id: str
    step_index: int
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    task: Optional[str]
    rgb_path: Optional[str]
    depth_path: Optional[str]
    rgb_blob: Optional[bytes] = None
    depth_blob: Optional[bytes] = None

    @property
    def pose_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return self.x, self.y, self.z, self.rx, self.ry, self.rz


@dataclass
class NearestMatch:
    state: ReplayState
    distance_xyz: float
    distance_yaw: float
    score: float
    within_threshold: bool


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection):
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id TEXT NOT NULL UNIQUE,
            trajectory_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            rx REAL NOT NULL,
            ry REAL NOT NULL,
            rz REAL NOT NULL,
            task TEXT,
            action TEXT,
            rgb_path TEXT,
            depth_path TEXT,
            rgb_width INTEGER,
            rgb_height INTEGER,
            depth_width INTEGER,
            depth_height INTEGER,
            rgb_blob BLOB,
            depth_blob BLOB,
            source_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS transitions (
            from_state_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            to_state_id INTEGER NOT NULL,
            PRIMARY KEY (from_state_id, action),
            FOREIGN KEY (from_state_id) REFERENCES states(id),
            FOREIGN KEY (to_state_id) REFERENCES states(id)
        );
        CREATE INDEX IF NOT EXISTS idx_states_traj_step ON states(trajectory_id, step_index);
        CREATE INDEX IF NOT EXISTS idx_states_xyz ON states(x, y, z);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )


def load_json_entries(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(f"数据集 JSON 必须是 list: {path}")
    return data


def _pose(entry: Dict) -> Dict[str, float]:
    pose = entry.get("pose")
    if not isinstance(pose, dict):
        raise RuntimeError(f"缺少 pose: {entry.get('sample_id')}")
    return {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose["z"]),
        "rx": float(pose.get("rx", 0.0)),
        "ry": float(pose.get("ry", 0.0)),
        "rz": float(pose["rz"]),
    }


def _action(entry: Dict) -> Optional[str]:
    action = entry.get("action")
    if isinstance(action, dict):
        name = str(action.get("name", "")).strip()
        return name or None
    name = str(entry.get("action_name", "")).strip()
    return name or None


def _image_info(entry: Dict) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int], Optional[int], Optional[int]]:
    observations = entry.get("observations")
    if not isinstance(observations, dict):
        return None, None, None, None, None, None
    rgb = observations.get("rgb")
    depth = observations.get("depth")
    if not isinstance(rgb, dict) or not isinstance(depth, dict):
        return None, None, None, None, None, None
    return (
        str(rgb.get("path", "")).strip() or None,
        str(depth.get("path", "")).strip() or None,
        int(rgb["width"]) if "width" in rgb else None,
        int(rgb["height"]) if "height" in rgb else None,
        int(depth["width"]) if "width" in depth else None,
        int(depth["height"]) if "height" in depth else None,
    )


def _read_blob(dataset_root: Path, rel_path: Optional[str]) -> Optional[bytes]:
    if not rel_path:
        return None
    path = dataset_root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"图片不存在，无法打包进数据库: {path}")
    return path.read_bytes()


def build_db_from_entries(
    db_path: Path,
    entries: Sequence[Dict],
    dataset_root: Path,
    store_images: bool = False,
    overwrite: bool = False,
) -> Dict:
    db_path = Path(db_path)
    if overwrite and db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _connect(db_path)
    init_db(conn)

    inserted = 0
    transitions = 0
    skipped = 0
    groups: Dict[str, List[Dict]] = {}

    try:
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                skipped += 1
                continue

            trajectory_id = str(entry.get("trajectory_id", "")).strip()
            if not trajectory_id:
                skipped += 1
                continue
            step_index = int(entry.get("step_index", i))
            sample_id = str(entry.get("sample_id", f"{trajectory_id}:{step_index:06d}"))
            pose = _pose(entry)
            rgb_path, depth_path, rgb_w, rgb_h, depth_w, depth_h = _image_info(entry)
            rgb_blob = _read_blob(dataset_root, rgb_path) if store_images else None
            depth_blob = _read_blob(dataset_root, depth_path) if store_images else None

            conn.execute(
                """
                INSERT OR REPLACE INTO states(
                    sample_id, trajectory_id, step_index,
                    x, y, z, rx, ry, rz,
                    task, action,
                    rgb_path, depth_path,
                    rgb_width, rgb_height, depth_width, depth_height,
                    rgb_blob, depth_blob, source_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_id,
                    trajectory_id,
                    step_index,
                    pose["x"],
                    pose["y"],
                    pose["z"],
                    pose["rx"],
                    pose["ry"],
                    pose["rz"],
                    entry.get("task") or entry.get("task_description"),
                    _action(entry),
                    rgb_path,
                    depth_path,
                    rgb_w,
                    rgb_h,
                    depth_w,
                    depth_h,
                    rgb_blob,
                    depth_blob,
                    json.dumps(entry, ensure_ascii=False),
                ),
            )
            groups.setdefault(trajectory_id, []).append(entry)
            inserted += 1

        state_ids = {
            (row["trajectory_id"], int(row["step_index"])): int(row["id"])
            for row in conn.execute("SELECT id, trajectory_id, step_index FROM states")
        }
        for trajectory_id, items in groups.items():
            ordered = sorted(items, key=lambda item: int(item.get("step_index", 0)))
            for current, nxt in zip(ordered, ordered[1:]):
                action = _action(current)
                if not action:
                    continue
                from_key = (trajectory_id, int(current.get("step_index", 0)))
                to_key = (trajectory_id, int(nxt.get("step_index", 0)))
                from_id = state_ids.get(from_key)
                to_id = state_ids.get(to_key)
                if from_id is None or to_id is None:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO transitions(from_state_id, action, to_state_id) VALUES (?, ?, ?)",
                    (from_id, action, to_id),
                )
                transitions += 1

        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("dataset_root", str(dataset_root)))
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("store_images", str(bool(store_images))))
        conn.commit()
    finally:
        conn.close()

    return {
        "db_path": str(db_path),
        "states": inserted,
        "transitions": transitions,
        "skipped": skipped,
        "store_images": bool(store_images),
    }


def _yaw_delta_deg(a: float, b: float) -> float:
    delta = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(delta)


def _row_to_state(row: sqlite3.Row, include_blobs: bool = False) -> ReplayState:
    return ReplayState(
        state_id=int(row["id"]),
        sample_id=str(row["sample_id"]),
        trajectory_id=str(row["trajectory_id"]),
        step_index=int(row["step_index"]),
        x=float(row["x"]),
        y=float(row["y"]),
        z=float(row["z"]),
        rx=float(row["rx"]),
        ry=float(row["ry"]),
        rz=float(row["rz"]),
        task=row["task"],
        rgb_path=row["rgb_path"],
        depth_path=row["depth_path"],
        rgb_blob=row["rgb_blob"] if include_blobs else None,
        depth_blob=row["depth_blob"] if include_blobs else None,
    )


class OfflineReplayDB:
    def __init__(
        self,
        db_path: Path,
        dataset_root: Optional[Path] = None,
        xyz_threshold: float = 5.0,
        yaw_threshold: float = 15.0,
    ):
        self.db_path = Path(db_path)
        self.conn = _connect(self.db_path)
        self.xyz_threshold = float(xyz_threshold)
        self.yaw_threshold = float(yaw_threshold)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else self._dataset_root_from_meta()
        self._states = [
            _row_to_state(row)
            for row in self.conn.execute(
                "SELECT id, sample_id, trajectory_id, step_index, x, y, z, rx, ry, rz, task, rgb_path, depth_path FROM states"
            )
        ]
        if not self._states:
            raise RuntimeError(f"Replay DB 没有 states: {self.db_path}")

    def close(self):
        self.conn.close()

    def _dataset_root_from_meta(self) -> Optional[Path]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = 'dataset_root'").fetchone()
        return Path(row["value"]) if row and row["value"] else None

    def get_state(self, state_id: int, include_blobs: bool = False) -> Optional[ReplayState]:
        columns = (
            "id, sample_id, trajectory_id, step_index, x, y, z, rx, ry, rz, task, rgb_path, depth_path"
            + (", rgb_blob, depth_blob" if include_blobs else "")
        )
        row = self.conn.execute(f"SELECT {columns} FROM states WHERE id = ?", (int(state_id),)).fetchone()
        return _row_to_state(row, include_blobs=include_blobs) if row else None

    def nearest(self, pose: Dict) -> NearestMatch:
        x = float(pose["x"])
        y = float(pose["y"])
        z = float(pose.get("z", 0.0))
        rz = float(pose.get("rz", 0.0))
        best = None

        for state in self._states:
            dxyz = math.sqrt((state.x - x) ** 2 + (state.y - y) ** 2 + (state.z - z) ** 2)
            dyaw = _yaw_delta_deg(state.rz, rz)
            score = (
                (dxyz / max(self.xyz_threshold, 1e-6)) ** 2
                + (dyaw / max(self.yaw_threshold, 1e-6)) ** 2
            )
            if best is None or score < best[0]:
                best = (score, state, dxyz, dyaw)

        score, state, dxyz, dyaw = best
        within = dxyz < self.xyz_threshold and dyaw < self.yaw_threshold
        return NearestMatch(
            state=state,
            distance_xyz=dxyz,
            distance_yaw=dyaw,
            score=score,
            within_threshold=within,
        )

    def transition(self, state_id: int, action: str) -> Optional[ReplayState]:
        row = self.conn.execute(
            "SELECT to_state_id FROM transitions WHERE from_state_id = ? AND action = ?",
            (int(state_id), str(action).upper()),
        ).fetchone()
        if not row:
            return None
        return self.get_state(int(row["to_state_id"]))

    def load_images(self, state: ReplayState) -> Tuple[Image.Image, Image.Image]:
        with_blobs = self.get_state(state.state_id, include_blobs=True)
        if with_blobs and with_blobs.rgb_blob and with_blobs.depth_blob:
            rgb = Image.open(io.BytesIO(with_blobs.rgb_blob)).convert("RGB")
            depth = Image.open(io.BytesIO(with_blobs.depth_blob)).convert("RGB")
            return rgb, depth

        if self.dataset_root is None:
            raise RuntimeError("数据库没有图片 BLOB，也没有可用 dataset_root。")
        if not state.rgb_path or not state.depth_path:
            raise RuntimeError(f"state 缺少图片路径: {state.sample_id}")
        rgb_path = self.dataset_root / state.rgb_path
        depth_path = self.dataset_root / state.depth_path
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB 图片不存在: {rgb_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth 图片不存在: {depth_path}")
        with Image.open(rgb_path) as rgb_src:
            rgb = rgb_src.convert("RGB")
        with Image.open(depth_path) as depth_src:
            depth = depth_src.convert("RGB")
        return rgb, depth


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline replay SQLite DB from schema v2 trajectory JSON")
    parser.add_argument("--dataset_json", required=True, help="例如 dataset/train_data_all.json")
    parser.add_argument("--dataset_root", required=True, help="图片根目录，例如 dataset")
    parser.add_argument("--db_path", required=True, help="输出 SQLite DB 路径")
    parser.add_argument("--store_images", action="store_true", help="把 RGB/Depth 图片作为 BLOB 写入 DB，方便只上传单个 DB")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 DB")
    return parser


def main():
    args = build_arg_parser().parse_args()
    entries = load_json_entries(Path(args.dataset_json))
    stats = build_db_from_entries(
        db_path=Path(args.db_path),
        entries=entries,
        dataset_root=Path(args.dataset_root),
        store_images=bool(args.store_images),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
