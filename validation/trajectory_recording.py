"""Optional on-disk visualization records for Stage 2D validation.

Only compressed RGB images and compact JSON metadata are written. Raw depth,
metric RGB-D payloads, and world-state binary dumps are deliberately omitted.
"""

import json
from pathlib import Path

import numpy as np

from agent_control.research_actions import (
    HoldAction,
    RotateAction,
    StopAction,
    TranslateAction,
)


SCHEMA_VERSION = 1


def serialize_action(action):
    if isinstance(action, TranslateAction):
        return {
            "type": "TRANSLATE",
            "dx_body": action.dx_body,
            "dy_body": action.dy_body,
            "dz_world": action.dz_world,
        }
    if isinstance(action, RotateAction):
        return {
            "type": "ROTATE",
            "dyaw": action.dyaw,
        }
    if isinstance(action, HoldAction):
        return {"type": "HOLD"}
    if isinstance(action, StopAction):
        return {
            "type": "STOP",
            "event_estimate_local": list(
                action.event_estimate_local
            ),
        }
    raise TypeError(f"Unsupported research action {action!r}")


def _serialize_view(view):
    return {
        "in_frustum_samples": view.in_frustum_samples,
        "clear_in_frustum_samples": view.clear_in_frustum_samples,
        "projected_bbox": (
            None
            if view.projected_bbox is None
            else list(view.projected_bbox)
        ),
        "projected_span_pixels": view.projected_span_pixels,
        "inside_image_margin": view.inside_image_margin,
        "task_observable": view.task_observable,
    }


def _serialize_assessment(assessment):
    return {
        "event_task_observable": assessment.event_task_observable,
        "cue_task_observable": assessment.cue_task_observable,
        "targets": [
            {
                "stable_id": target.stable_id,
                "role": target.role.name,
                "oblique": _serialize_view(target.oblique),
                "nadir": _serialize_view(target.nadir),
            }
            for target in assessment.targets
        ],
    }


def _serialize_entity(entity):
    return {
        "stable_id": entity.stable_id,
        "gta_handle": entity.gta_handle,
        "model_hash": entity.model_hash,
        "kind": entity.kind.name,
        "role": entity.role.name,
        "event_id": entity.event_id,
        "task_state": entity.task_state.name,
        "exists": entity.exists,
        "position": list(entity.position),
        "velocity": list(entity.velocity),
        "speed": entity.speed,
        "heading": entity.heading,
        "task_target": list(entity.task_target),
    }


def _serialize_clock(clock):
    return {
        "session_id": clock.session_id,
        "step_index": clock.step_index,
        "game_timer_ms": clock.game_timer_ms,
        "frame_count": clock.frame_count,
        "actual_elapsed_ms": clock.actual_elapsed_ms,
    }


def _serialize_witness(witness):
    return {
        "cue": {
            "stable_id": witness.cue.stable_id,
            "role": witness.cue.role.name,
            "first_step": witness.cue.first_step,
            "second_step": witness.cue.second_step,
            "first_pose": list(witness.cue.first_pose),
            "second_pose": list(witness.cue.second_pose),
            "transition_action": serialize_action(
                witness.cue.transition_action
            ),
            "horizontal_displacement_m": (
                witness.cue.horizontal_displacement_m
            ),
            "direction_cosine": witness.cue.direction_cosine,
        },
        "goal": {
            "stable_id": witness.goal.stable_id,
            "pose": list(witness.goal.pose),
            "node_index": witness.goal.node_index,
        },
        "actions": [
            serialize_action(action)
            for action in witness.actions
        ],
        "translate_actions": witness.translate_actions,
        "rotate_actions": witness.rotate_actions,
        "hold_actions": witness.hold_actions,
        "stop_actions": witness.stop_actions,
        "total_actions": witness.total_actions,
        "remaining_actions": witness.remaining_actions,
    }


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(path)


class TrajectoryRecorder:
    @staticmethod
    def require_dependencies():
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError(
                "Trajectory recording requires Pillow"
            ) from error
        return Image

    def __init__(
        self,
        root,
        report,
        generated_start,
        jpeg_quality=85,
    ):
        self._image_type = self.require_dependencies()
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(
                f"Trajectory directory already exists: {self.root}"
            )
        self.root.mkdir(parents=True)
        (self.root / "oblique").mkdir()
        (self.root / "nadir").mkdir()
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be in [1, 95]")
        self.report = report
        self.generated_start = generated_start
        self.frames = []
        self._finished = False

    def _save_rgb(self, view_name, index, frame):
        relative = Path(view_name) / f"{index:04d}.jpg"
        output = self.root / relative
        rgb = np.asarray(frame.rgb_array(), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(
                f"{view_name} RGB has invalid shape {rgb.shape}"
            )
        self._image_type.fromarray(rgb).save(
            output,
            format="JPEG",
            quality=self.jpeg_quality,
            subsampling=0,
        )
        return relative.as_posix()

    def record(
        self,
        observation_step,
        action,
        pair,
        pose,
        odometry,
        scenario,
        assessment,
    ):
        if self._finished:
            raise RuntimeError("Trajectory recorder is already finished")
        index = len(self.frames)
        oblique_path = self._save_rgb(
            "oblique",
            index,
            pair.oblique,
        )
        nadir_path = self._save_rgb(
            "nadir",
            index,
            pair.nadir,
        )
        witness = self.report.witness
        self.frames.append(
            {
                "index": index,
                "observation_step": int(observation_step),
                "action_index": int(observation_step) + 1,
                "action": serialize_action(action),
                "is_witness_cue_observation": (
                    int(observation_step)
                    in (
                        witness.cue.first_step,
                        witness.cue.second_step,
                    )
                ),
                "is_terminal_observation": isinstance(
                    action,
                    StopAction,
                ),
                "camera_pose_world": [
                    float(value) for value in pose
                ],
                "camera_position_start_local": list(
                    odometry.position_local
                ),
                "yaw_from_start_degrees": (
                    odometry.yaw_from_start_degrees
                ),
                "clock": _serialize_clock(pair.clock),
                "rgb": {
                    "oblique": {
                        "path": oblique_path,
                        "frame_id": pair.oblique.frame_id,
                        "width": pair.oblique.width,
                        "height": pair.oblique.height,
                    },
                    "nadir": {
                        "path": nadir_path,
                        "frame_id": pair.nadir.frame_id,
                        "width": pair.nadir.width,
                        "height": pair.nadir.height,
                    },
                },
                "scenario": {
                    "scenario_id": scenario.scenario_id,
                    "blueprint_id": scenario.blueprint_id,
                    "seed": scenario.seed,
                    "game_timer_ms": scenario.game_timer_ms,
                    "frame_count": scenario.frame_count,
                    "event_position": list(
                        scenario.event_position
                    ),
                    "event_active": scenario.event_active,
                    "entities": [
                        _serialize_entity(entity)
                        for entity in scenario.entities
                    ],
                },
                "visibility": _serialize_assessment(assessment),
            }
        )

    def finish(
        self,
        status,
        cue_reproduced=None,
        error=None,
    ):
        if self._finished:
            return
        self._finished = True
        blueprint = self.generated_start.blueprint
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": str(status),
            "error": None if error is None else str(error),
            "visibility_stratum": (
                self.report.visibility_stratum.name
            ),
            "start": {
                "start_id": blueprint.start_id,
                "scenario_blueprint_id": (
                    blueprint.scenario_blueprint_id
                ),
                "start_seed": blueprint.start_seed,
                "candidate_index": blueprint.candidate_index,
                "absolute_pose": list(blueprint.absolute_pose),
                "event_distance": blueprint.event_distance,
                "event_bearing_body_degrees": (
                    blueprint.event_bearing_body_degrees
                ),
            },
            "cue_reproduced": cue_reproduced,
            "witness": _serialize_witness(self.report.witness),
            "frames": self.frames,
        }
        write_json(self.root / "trajectory.json", payload)

    def size_bytes(self):
        return sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file()
        )
