"""Bounded on-disk writer for successful Stage 2E expert episodes."""

import dataclasses
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


DEPTH_STORAGE_DTYPE = np.float16


def _json_value(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if hasattr(value, "value"):
        return _json_value(value.value)
    return value


def action_record(action):
    result = {"type": type(action).__name__.removesuffix("Action").upper()}
    if hasattr(action, "event_estimate_local"):
        result["event_estimate_local"] = [
            float(value) for value in action.event_estimate_local
        ]
    return result


class ExpertEpisodeRecorder:
    def __init__(
        self,
        output_root,
        episode_name,
        jpeg_quality=95,
    ):
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.final_path = self.output_root / str(episode_name)
        self.partial_path = self.output_root / (
            str(episode_name) + ".partial"
        )
        if self.final_path.exists() or self.partial_path.exists():
            raise FileExistsError(
                f"Episode output already exists: {self.final_path}"
            )
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be in [1, 95]")
        self.partial_path.mkdir()
        for relative in (
            "agent/rgb",
            "agent/depth",
            "teacher",
            "evaluation_truth",
        ):
            (self.partial_path / relative).mkdir(parents=True)
        self._agent_labels = open(
            self.partial_path / "agent" / "steps.jsonl",
            "x",
            encoding="utf-8",
        )
        self._teacher_labels = open(
            self.partial_path / "teacher" / "awareness.jsonl",
            "x",
            encoding="utf-8",
        )
        self._truth_labels = open(
            self.partial_path
            / "evaluation_truth"
            / "steps.jsonl",
            "x",
            encoding="utf-8",
        )
        self._beliefs = []
        self._closed = False

    def write_metadata(self, agent, teacher, evaluation_truth):
        agent = dict(agent)
        agent["depth_storage"] = {
            "dtype": np.dtype(DEPTH_STORAGE_DTYPE).name,
            "units": "meters",
            "encoding": "IEEE 754 binary16",
            "lossy": True,
        }
        for relative, payload in (
            ("agent/episode.json", agent),
            ("teacher/episode.json", teacher),
            ("evaluation_truth/episode.json", evaluation_truth),
        ):
            with open(
                self.partial_path / relative,
                "x",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    _json_value(payload),
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
                stream.write("\n")

    def record_step(
        self,
        step_index,
        pair,
        observation,
        grounded,
        decision,
        evaluation_truth,
    ):
        if self._closed:
            raise RuntimeError("Episode recorder is closed")
        step_index = int(step_index)
        stem = f"{step_index:03d}"
        for name, frame in (
            ("oblique", pair.oblique),
            ("nadir", pair.nadir),
        ):
            Image.fromarray(frame.rgb_array()).save(
                self.partial_path
                / "agent"
                / "rgb"
                / f"{stem}_{name}.jpg",
                quality=self.jpeg_quality,
                subsampling=0,
            )
            depth = frame.depth_array()
            if (
                not np.isfinite(depth).all()
                or np.any(depth < 0.0)
                or np.any(depth > np.finfo(DEPTH_STORAGE_DTYPE).max)
            ):
                raise RuntimeError(
                    f"{name} depth cannot be represented as finite "
                    "non-negative float16 metres"
                )
            np.savez_compressed(
                self.partial_path
                / "agent"
                / "depth"
                / f"{stem}_{name}.npz",
                depth=depth.astype(DEPTH_STORAGE_DTYPE),
            )
        agent_record = {
            "step_index": step_index,
            "clock": pair.clock,
            "frame_ids": {
                "oblique": pair.oblique.frame_id,
                "nadir": pair.nadir.frame_id,
            },
            "odometry": observation.odometry,
            "action": action_record(decision.action),
        }
        teacher_record = {
            "step_index": step_index,
            "grounded_tracks": grounded.tracks,
            "awareness": decision.awareness,
        }
        for stream, payload in (
            (self._agent_labels, agent_record),
            (self._teacher_labels, teacher_record),
            (self._truth_labels, evaluation_truth),
        ):
            stream.write(
                json.dumps(
                    _json_value(payload),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            stream.flush()
        self._beliefs.append(
            np.asarray(decision.belief, dtype=np.float32)
        )

    def finish(self, summary):
        if self._closed:
            raise RuntimeError("Episode recorder is closed")
        for stream in (
            self._agent_labels,
            self._teacher_labels,
            self._truth_labels,
        ):
            stream.close()
        np.savez_compressed(
            self.partial_path / "teacher" / "beliefs.npz",
            belief=np.stack(self._beliefs, axis=0),
        )
        with open(
            self.partial_path / "summary.json",
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                _json_value(summary),
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")
        os.replace(self.partial_path, self.final_path)
        self._closed = True
        return self.final_path

    def abort(self):
        if self._closed:
            return
        for stream in (
            self._agent_labels,
            self._teacher_labels,
            self._truth_labels,
        ):
            if not stream.closed:
                stream.close()
        resolved = self.partial_path.resolve()
        if resolved.parent != self.output_root:
            raise RuntimeError(
                "Refusing to remove a partial directory outside output_root"
            )
        if resolved.exists():
            shutil.rmtree(resolved)
        self._closed = True


def _append_jsonl(output_root, filename, payload):
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with open(
        output_root / filename,
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write(
            json.dumps(
                _json_value(payload),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )


def append_failure(output_root, payload):
    _append_jsonl(output_root, "failures.jsonl", payload)


def append_attempt_timing(output_root, payload):
    _append_jsonl(output_root, "timings.jsonl", payload)
