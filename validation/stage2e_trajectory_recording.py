"""Optional compact recordings for Stage 2E expert validation.

The recorder deliberately omits metric depth. It stores compressed RGB,
structured decisions, compact evaluation truth, and belief grids only when an
explicit recording directory is supplied by the validator.
"""

import dataclasses
import enum
import json
import os
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def _json_value(value):
    if isinstance(value, enum.Enum):
        return value.name
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
    return value


def _clock_record(clock):
    return {
        "session_id": int(clock.session_id),
        "step_index": int(clock.step_index),
        "game_timer_ms": int(clock.game_timer_ms),
        "frame_count": int(clock.frame_count),
        "actual_elapsed_ms": int(clock.actual_elapsed_ms),
    }


def _action_record(action):
    result = {
        "type": type(action).__name__.removesuffix("Action").upper()
    }
    if hasattr(action, "event_estimate_local"):
        result["event_estimate_local"] = [
            float(value) for value in action.event_estimate_local
        ]
    return result


class Stage2EValidationRecorder:
    @staticmethod
    def require_dependencies():
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Stage 2E trajectory recording requires Pillow"
            ) from error
        return Image

    def __init__(self, root, jpeg_quality=85):
        self._image_type = self.require_dependencies()
        self.root = Path(root).resolve()
        self.partial_root = self.root.with_name(
            self.root.name + ".partial"
        )
        if self.root.exists() or self.partial_root.exists():
            raise FileExistsError(
                f"Stage 2E recording path already exists: {self.root}"
            )
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be in [1, 95]")
        self.partial_root.mkdir(parents=True)
        (self.partial_root / "oblique").mkdir()
        (self.partial_root / "nadir").mkdir()
        self._metadata = None
        self._blueprint = None
        self._frames = []
        self._beliefs = []
        self._finished = False

    def write_metadata(self, agent, teacher, evaluation_truth):
        if self._metadata is not None:
            raise RuntimeError("Stage 2E recording metadata already exists")
        self._blueprint = evaluation_truth["start_blueprint"]
        self._metadata = {
            "agent": _json_value(agent),
            "teacher": _json_value(teacher),
            "evaluation_truth": _json_value(evaluation_truth),
        }

    def _save_rgb(self, view_name, index, frame):
        relative = Path(view_name) / f"{index:04d}.jpg"
        rgb = np.asarray(frame.rgb_array(), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(
                f"{view_name} RGB has invalid shape {rgb.shape}"
            )
        self._image_type.fromarray(rgb).save(
            self.partial_root / relative,
            format="JPEG",
            quality=self.jpeg_quality,
            subsampling=0,
        )
        return relative.as_posix()

    def record_step(
        self,
        step_index,
        pair,
        observation,
        grounded,
        decision,
        evaluation_truth,
    ):
        if self._finished:
            raise RuntimeError("Stage 2E trajectory recorder is closed")
        if self._blueprint is None:
            raise RuntimeError("Stage 2E recording metadata is missing")
        index = len(self._frames)
        rgb = {}
        for view_name, frame in (
            ("oblique", pair.oblique),
            ("nadir", pair.nadir),
        ):
            rgb[view_name] = {
                "path": self._save_rgb(view_name, index, frame),
                "frame_id": int(frame.frame_id),
                "width": int(frame.width),
                "height": int(frame.height),
            }

        local_position = tuple(
            float(value) for value in observation.odometry.position_local
        )
        world_position = self._blueprint.local_to_world(local_position)
        yaw_world = (
            float(self._blueprint.absolute_pose[5])
            + float(observation.odometry.yaw_from_start_degrees)
        )
        event_world = evaluation_truth["event_position"]
        entities = []
        for entity in evaluation_truth["entities"]:
            item = _json_value(entity)
            item["position_local"] = list(
                self._blueprint.world_to_local(entity.position)
            )
            entities.append(item)

        self._frames.append(
            {
                "index": index,
                "observation_step": int(step_index),
                "action_index": index + 1,
                "action": _action_record(decision.action),
                "action_execution": "PROPOSED",
                "clock": _clock_record(pair.clock),
                "rgb": rgb,
                "camera_pose_world": [
                    *world_position,
                    float(self._blueprint.absolute_pose[3]),
                    0.0,
                    yaw_world,
                ],
                "odometry": _json_value(observation.odometry),
                "grounded_tracks": _json_value(grounded.tracks),
                "awareness": _json_value(decision.awareness),
                "evaluation_truth": {
                    "event_active": bool(
                        evaluation_truth["event_active"]
                    ),
                    "event_position": list(event_world),
                    "event_position_local": list(
                        self._blueprint.world_to_local(event_world)
                    ),
                    "entities": entities,
                    "valid_dynamic_cue_so_far": bool(
                        evaluation_truth["valid_dynamic_cue_so_far"]
                    ),
                },
            }
        )
        self._beliefs.append(
            np.asarray(decision.belief, dtype=np.float32).copy()
        )

    def mark_last_action_executed(self):
        if self._finished:
            raise RuntimeError("Stage 2E trajectory recorder is closed")
        if not self._frames:
            raise RuntimeError("No Stage 2E action is available to mark")
        frame = self._frames[-1]
        if frame["action_execution"] != "PROPOSED":
            raise RuntimeError("Stage 2E action was already marked executed")
        frame["action_execution"] = "EXECUTED"

    def finish(self, status, result=None, error=None):
        if self._finished:
            raise RuntimeError("Stage 2E trajectory recorder is closed")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": str(status),
            "error": None if error is None else str(error),
            "result": None if result is None else _json_value(result),
            "metadata": self._metadata,
            "frames": self._frames,
        }
        with (self.partial_root / "trajectory.json").open(
            "x",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        if self._beliefs:
            belief = np.stack(self._beliefs, axis=0)
        else:
            belief = np.empty((0, 0, 0), dtype=np.float32)
        np.savez_compressed(
            self.partial_root / "beliefs.npz",
            belief=belief,
        )
        os.replace(self.partial_root, self.root)
        self._finished = True
        return self.root

    def size_bytes(self):
        base = self.root if self._finished else self.partial_root
        return sum(
            path.stat().st_size
            for path in base.rglob("*")
            if path.is_file()
        )
