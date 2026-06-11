import argparse
import hashlib
import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dronesim_client import DroneSimClient
from offline_replay_db import _connect, init_db
from rgbd_utils import depth_bytes_to_pil, rgb_bytes_to_pil
from verification_runtime import read_jsonl


def _safe_id(value: str) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)[:80]


def _pose_dict(raw: Dict) -> Dict[str, float]:
    return {
        "x": float(raw.get("x", 0.0)),
        "y": float(raw.get("y", 0.0)),
        "z": float(raw.get("z", 0.0)),
        "rx": float(raw.get("rx", 0.0)),
        "ry": float(raw.get("ry", 0.0)),
        "rz": float(raw.get("rz", 0.0)),
    }


def _scenario_id(row: Dict) -> Optional[str]:
    value = row.get("scenario_id")
    if value:
        return str(value)
    scenario_ids = row.get("scenario_ids")
    if isinstance(scenario_ids, list) and scenario_ids:
        return str(scenario_ids[0])
    return None


def _miss_hash(row: Dict) -> str:
    wanted = _pose_dict(row.get("wanted_pose") or {})
    key = {
        "scenario_id": _scenario_id(row),
        "reason": row.get("reason"),
        "action": row.get("action"),
        "wanted_pose": {
            "x": round(wanted["x"], 3),
            "y": round(wanted["y"], 3),
            "z": round(wanted["z"], 3),
            "rz": round(wanted["rz"], 3),
        },
        "nearest_state_id": ((row.get("nearest_state") or {}).get("state_id")),
    }
    raw = json.dumps(key, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _image_bytes(image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _load_samples(path: Optional[Path]) -> Dict[str, Dict]:
    if path is None:
        return {}
    samples = {}
    for sample in read_jsonl(Path(path)):
        for key in ("scenario_id", "session_id", "sample_id"):
            value = sample.get(key)
            if value:
                samples[str(value)] = sample
    return samples


def _db_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row and row["value"] is not None else None


def _resolve_image_storage(conn: sqlite3.Connection, args) -> Tuple[str, Optional[Path]]:
    mode = str(args.image_storage)
    if mode == "auto":
        mode = "blob" if _db_meta(conn, "store_images") == "True" else "files"

    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    if dataset_root is None:
        meta_root = _db_meta(conn, "dataset_root")
        dataset_root = Path(meta_root) if meta_root else None

    if mode == "files" and dataset_root is None:
        raise RuntimeError("image_storage=files 需要 --dataset_root，或 DB meta 中已有 dataset_root。")
    return mode, dataset_root


def _create_anomaly_at_position(cli: DroneSimClient, sample: Dict):
    anomaly_type = str(sample.get("anomaly_type", "")).strip().lower()
    pos = sample.get("anomaly_position") or {}
    x = float(pos["x"])
    y = float(pos["y"])
    z = float(pos["z"])

    cli.clear_scene()
    time.sleep(0.5)

    cli.stop_camera()
    time.sleep(1.0)
    cli.teleport_player(x, y, z)
    time.sleep(1.0)
    cli.create_camera()
    time.sleep(2.0)

    original_pose = cli.get_pose()
    cli.set_posture(x, y, z + 2.0, 0.0, 0.0, 0.0)
    time.sleep(0.5)

    if anomaly_type == "fire":
        result = cli.create_fire()
    elif anomaly_type == "arrest":
        result = cli.create_arrest()
    elif anomaly_type == "accident":
        result = cli.create_accident()
    else:
        raise RuntimeError(f"不支持的 anomaly_type: {anomaly_type}")

    if original_pose is not None:
        ox, oy, oz, orx, ory, orz = original_pose
        cli.set_posture(ox, oy, oz, orx, ory, orz)
    return result


def _capture_rgbd(cli: DroneSimClient, retries: int):
    last_error = None
    for _ in range(max(1, int(retries))):
        cap = cli.capture()
        if cap is None:
            last_error = RuntimeError("capture returned None")
            time.sleep(0.5)
            continue
        w, h, rgb_bytes, depth_bytes = cap
        try:
            rgb_img = rgb_bytes_to_pil(w, h, rgb_bytes)
            depth_img = depth_bytes_to_pil(w, h, depth_bytes)
            return int(w), int(h), rgb_img, depth_img
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"采集 RGBD 失败: {last_error}")


def _state_id_by_sample_id(conn: sqlite3.Connection, sample_id: str) -> Optional[int]:
    row = conn.execute("SELECT id FROM states WHERE sample_id = ?", (sample_id,)).fetchone()
    return int(row["id"]) if row else None


def _insert_capture_state(
    conn: sqlite3.Connection,
    miss: Dict,
    sample: Optional[Dict],
    state_pose: Dict[str, float],
    width: int,
    height: int,
    rgb_blob: Optional[bytes],
    depth_blob: Optional[bytes],
    rgb_rel: Optional[str],
    depth_rel: Optional[str],
    step_index: int,
    overwrite: bool,
) -> int:
    scenario_id = _scenario_id(miss) or "unknown"
    suffix = _miss_hash(miss)
    sample_id = f"miss_{suffix}"
    trajectory_id = f"miss_collect_{_safe_id(scenario_id)}"
    source = {
        "schema_version": "offline_miss_capture_v1",
        "sample_id": sample_id,
        "trajectory_id": trajectory_id,
        "step_index": int(step_index),
        "pose": state_pose,
        "task": (sample or {}).get("task_description") or (sample or {}).get("task"),
        "miss": miss,
        "sample": sample,
        "captured_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    existing_id = _state_id_by_sample_id(conn, sample_id)
    if existing_id is not None and not overwrite:
        return existing_id

    sql = "INSERT OR REPLACE" if overwrite else "INSERT"
    conn.execute(
        f"""
        {sql} INTO states(
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
            int(step_index),
            state_pose["x"],
            state_pose["y"],
            state_pose["z"],
            state_pose["rx"],
            state_pose["ry"],
            state_pose["rz"],
            source["task"],
            None,
            rgb_rel,
            depth_rel,
            int(width),
            int(height),
            int(width),
            int(height),
            rgb_blob,
            depth_blob,
            json.dumps(source, ensure_ascii=False),
        ),
    )
    state_id = _state_id_by_sample_id(conn, sample_id)
    if state_id is None:
        raise RuntimeError(f"插入 state 后无法查询 sample_id={sample_id}")
    return state_id


def _add_transition(conn: sqlite3.Connection, miss: Dict, to_state_id: int, replace: bool):
    current_state = miss.get("current_state") or {}
    from_state_id = current_state.get("state_id")
    action = str(miss.get("action") or "").strip().upper()
    if not from_state_id or not action:
        return False
    sql = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    conn.execute(
        f"{sql} INTO transitions(from_state_id, action, to_state_id) VALUES (?, ?, ?)",
        (int(from_state_id), action, int(to_state_id)),
    )
    return True


def _save_images(dataset_root: Path, rel_dir: str, sample_id: str, rgb_img, depth_img) -> Tuple[str, str]:
    rel_base = Path(rel_dir)
    out_dir = dataset_root / rel_base
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_rel = (rel_base / f"{sample_id}_rgb.png").as_posix()
    depth_rel = (rel_base / f"{sample_id}_depth.png").as_posix()
    rgb_img.save(dataset_root / rgb_rel)
    depth_img.save(dataset_root / depth_rel)
    return rgb_rel, depth_rel


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect coverage miss frames in GTA V and append them to offline replay DB")
    parser.add_argument("--misses_file", required=True, help="run_offline_verification.py 输出的 coverage_misses.jsonl")
    parser.add_argument("--db_path", required=True, help="要补充的 offline replay SQLite DB")
    parser.add_argument("--verification_file", default=None, help="用于按 scenario_id 重建异常的 samples.jsonl")
    parser.add_argument("--dataset_root", default=None, help="image_storage=files 时保存图片的 dataset 根目录")
    parser.add_argument("--output_image_dir", default="imgs/offline_miss", help="相对 dataset_root 的补采图片目录")
    parser.add_argument("--image_storage", choices=("auto", "blob", "files"), default="auto")
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep_s", type=float, default=3.0)
    parser.add_argument("--settle_s", type=float, default=0.5)
    parser.add_argument("--capture_retries", type=int, default=3)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--time_hour", type=int, default=12)
    parser.add_argument("--time_minute", type=int, default=0)
    parser.add_argument("--time_second", type=int, default=0)
    parser.add_argument("--weather", default=None)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 miss state")
    parser.add_argument("--no_add_transitions", action="store_true", help="只补 state，不添加 current_state/action -> miss_state transition")
    parser.add_argument("--replace_transitions", action="store_true", help="替换已有同 from_state/action transition")
    parser.add_argument("--no_recreate_anomaly", action="store_true", help="不按 samples.jsonl 重建异常场景")
    parser.add_argument("--no_restore_player", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    misses = read_jsonl(Path(args.misses_file))
    if int(args.limit) > 0:
        misses = misses[: int(args.limit)]
    if not misses:
        print(f"No misses found: {args.misses_file}")
        return 1

    samples = _load_samples(Path(args.verification_file)) if args.verification_file else {}
    conn = _connect(Path(args.db_path))
    init_db(conn)
    image_storage, dataset_root = _resolve_image_storage(conn, args)

    cli = DroneSimClient(host=args.host, port=int(args.port))
    print(f"Waiting {float(args.sleep_s)} seconds...")
    time.sleep(float(args.sleep_s))

    inserted = 0
    skipped = 0
    transitions = 0
    current_scenario = None

    try:
        cli.create_camera()
        cli.set_time(args.time_hour, args.time_minute, args.time_second)
        if args.weather:
            cli.set_weather(str(args.weather))
        if args.fov is not None:
            cli.set_fov(float(args.fov))

        for i, miss in enumerate(misses):
            wanted_pose = miss.get("wanted_pose")
            if not isinstance(wanted_pose, dict):
                print(f"[{i + 1}/{len(misses)}] skip: missing wanted_pose")
                skipped += 1
                continue

            scenario_id = _scenario_id(miss)
            sample = samples.get(str(scenario_id)) if scenario_id else None
            if not args.no_recreate_anomaly and scenario_id != current_scenario:
                if sample is None:
                    print(f"[{i + 1}/{len(misses)}] warning: no sample context for scenario_id={scenario_id}, capture without recreating anomaly")
                else:
                    print(f"[{i + 1}/{len(misses)}] recreate anomaly for scenario_id={scenario_id}")
                    _create_anomaly_at_position(cli, sample)
                    if args.fov is not None:
                        cli.set_fov(float(args.fov))
                current_scenario = scenario_id

            pose = _pose_dict(wanted_pose)
            suffix = _miss_hash(miss)
            sample_id = f"miss_{suffix}"
            existing_id = _state_id_by_sample_id(conn, sample_id)
            if existing_id is not None and not args.overwrite:
                if not args.no_add_transitions and _add_transition(conn, miss, existing_id, bool(args.replace_transitions)):
                    transitions += 1
                print(f"[{i + 1}/{len(misses)}] skip existing {sample_id}")
                skipped += 1
                continue

            print(
                f"[{i + 1}/{len(misses)}] capture {sample_id}: "
                f"x={pose['x']:.2f}, y={pose['y']:.2f}, z={pose['z']:.2f}, rz={pose['rz']:.2f}"
            )
            cli.set_posture(pose["x"], pose["y"], pose["z"], pose["rx"], pose["ry"], pose["rz"])
            time.sleep(float(args.settle_s))
            actual = cli.get_pose()
            state_pose = _pose_dict({
                "x": actual[0],
                "y": actual[1],
                "z": actual[2],
                "rx": actual[3],
                "ry": actual[4],
                "rz": actual[5],
            }) if actual is not None else pose

            width, height, rgb_img, depth_img = _capture_rgbd(cli, int(args.capture_retries))
            if image_storage == "blob":
                rgb_blob = _image_bytes(rgb_img)
                depth_blob = _image_bytes(depth_img)
                rgb_rel = None
                depth_rel = None
            else:
                rgb_blob = None
                depth_blob = None
                rgb_rel, depth_rel = _save_images(Path(dataset_root), str(args.output_image_dir), sample_id, rgb_img, depth_img)

            state_id = _insert_capture_state(
                conn=conn,
                miss=miss,
                sample=sample,
                state_pose=state_pose,
                width=width,
                height=height,
                rgb_blob=rgb_blob,
                depth_blob=depth_blob,
                rgb_rel=rgb_rel,
                depth_rel=depth_rel,
                step_index=i,
                overwrite=bool(args.overwrite),
            )
            inserted += 1
            if not args.no_add_transitions and _add_transition(conn, miss, state_id, bool(args.replace_transitions)):
                transitions += 1
            conn.commit()
    finally:
        try:
            cli.stop_camera()
        except Exception:
            pass
        try:
            cli.clear_scene()
        except Exception:
            pass
        try:
            if not args.no_restore_player:
                cli.restore_player()
        except Exception:
            pass
        conn.close()

    print("\n补采完成")
    print(f"inserted={inserted} skipped={skipped} transitions={transitions} image_storage={image_storage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
