"""Planner-free online agent for the Stage 3C explicit-belief policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time

from learning.belief_features import GroundedTrackFeatureConfig
from learning.online_belief_policy import OnlineExplicitBeliefPolicyRuntime
from learning.policy_dataset import ACTION_HISTORY_PAD, ACTION_NAMES

from .research_actions import (
    AscendAction,
    DescendAction,
    ForwardAction,
    HoldAction,
    StopAction,
    TurnLeftAction,
    TurnRightAction,
)


_ACTION_CONSTRUCTORS = {
    "FORWARD": ForwardAction,
    "ASCEND": AscendAction,
    "DESCEND": DescendAction,
    "TURNLEFT": TurnLeftAction,
    "TURNRIGHT": TurnRightAction,
    "HOLD": HoldAction,
}


def strict_action_name(action) -> str:
    name = type(action).__name__.removesuffix("Action").upper()
    if name not in ACTION_NAMES:
        raise RuntimeError(f"INVALID_STRICT_ACTION_TYPE: {type(action).__name__}")
    return name


def action_from_policy_snapshot(snapshot):
    name = snapshot.selected_action_name
    if name == "STOP":
        return StopAction(snapshot.event_estimate_local)
    try:
        return _ACTION_CONSTRUCTORS[name]()
    except KeyError as error:
        raise RuntimeError(f"UNKNOWN_POLICY_ACTION: {name}") from error


@dataclass(frozen=True)
class ExplicitBeliefPolicyAwareness:
    mode: str
    checkpoint_name: str
    checkpoint_epoch: int
    dagger_iteration: int
    source_seen: bool
    source_visible_now: bool
    source_track_id: int | None
    source_age: int
    inference_started: bool
    belief_updated: bool
    evidence_track_ids: tuple[int, ...]
    belief_entropy: float
    map_cell: tuple[int, int]
    map_local_xy: tuple[float, float]
    credible_areas_m2: tuple[float, float, float]
    raw_action_probabilities: tuple[float, ...]
    legal_action_probabilities: tuple[float, ...]
    proposed_action: str
    executed_action: str | None
    executed_by: str | None
    expert_action: str | None
    expert_label_available: bool
    expert_error: str | None
    remaining_value: float
    event_estimate_local: tuple[float, float, float]
    action_history: tuple[str, ...]
    model_seconds: float


@dataclass(frozen=True)
class ExplicitBeliefPolicyDecision:
    action: object
    awareness: ExplicitBeliefPolicyAwareness
    belief: object
    policy_snapshot: object
    model_seconds: float


class OnlineExplicitBeliefPolicyAgent:
    """Wrap the strict model runtime without a planner or geometry oracle."""

    def __init__(
        self,
        episode_spec,
        observation_spec,
        checkpoint_path,
        mode="control",
        device="auto",
    ):
        mode = str(mode).lower()
        if mode not in ("shadow", "control", "dagger"):
            raise ValueError("mode must be shadow, control, or dagger")
        self.mode = mode
        self.runtime = OnlineExplicitBeliefPolicyRuntime(
            checkpoint_path,
            GroundedTrackFeatureConfig(
                radius_m=120.0,
                vertical_bound_m=float(episode_spec.activity_vertical_m),
                horizon_steps=int(episode_spec.horizon_steps),
                width=int(observation_spec.width),
                height=int(observation_spec.height),
            ),
            horizon_steps=int(episode_spec.horizon_steps),
            vertical_bound_m=float(episode_spec.activity_vertical_m),
            device=device,
        )
        self._validate_execution_contract(episode_spec, observation_spec)

    def _validate_execution_contract(self, episode_spec, observation_spec):
        contract = self.runtime.checkpoint["episode_contract"]
        episode_fields = {
            "horizon_steps": int,
            "activity_radius_m": float,
            "activity_vertical_m": float,
            "forward_step_m": float,
            "vertical_step_m": float,
            "yaw_step_degrees": float,
            "simulation_step_ms": int,
        }
        observation_fields = {
            "width": int,
            "height": int,
            "fov_degrees": float,
            "near_clip": float,
            "far_clip": float,
            "oblique_pitch_degrees": float,
            "nadir_pitch_degrees": float,
        }
        for group_name, source, fields in (
            ("episode_spec", episode_spec, episode_fields),
            ("observation_spec", observation_spec, observation_fields),
        ):
            expected = contract.get(group_name, {})
            for name, conversion in fields.items():
                if name not in expected:
                    raise RuntimeError(
                        f"Stage 3C checkpoint is missing {group_name}.{name}"
                    )
                actual_value = conversion(getattr(source, name))
                expected_value = conversion(expected[name])
                equal = (
                    actual_value == expected_value
                    if conversion is int
                    else abs(actual_value - expected_value) <= 1.0e-6
                )
                if not equal:
                    raise RuntimeError(
                        f"Checkpoint/online {group_name}.{name} mismatch: "
                        f"{expected_value} != {actual_value}"
                    )

    @property
    def checkpoint(self):
        return self.runtime.checkpoint

    @staticmethod
    def _history_names(indices):
        return tuple(
            "PAD" if value == ACTION_HISTORY_PAD else ACTION_NAMES[value]
            for value in indices
        )

    def predict(self, pair, observation, grounded, action_count):
        started = time.perf_counter()
        snapshot = self.runtime.update(
            grounded, observation, pair, action_count
        )
        elapsed = time.perf_counter() - started
        action = action_from_policy_snapshot(snapshot)
        awareness = ExplicitBeliefPolicyAwareness(
            mode=self.mode,
            checkpoint_name=self.runtime.checkpoint_path.name,
            checkpoint_epoch=int(self.checkpoint.get("epoch", -1)),
            dagger_iteration=int(self.checkpoint.get("dagger_iteration", 0)),
            source_seen=snapshot.source_seen,
            source_visible_now=snapshot.source_visible_now,
            source_track_id=snapshot.source_track_id,
            source_age=snapshot.source_age,
            inference_started=snapshot.inference_started,
            belief_updated=snapshot.belief_updated,
            evidence_track_ids=snapshot.evidence_track_ids,
            belief_entropy=snapshot.entropy,
            map_cell=snapshot.map_cell,
            map_local_xy=snapshot.map_local_xy,
            credible_areas_m2=snapshot.credible_areas_m2,
            raw_action_probabilities=snapshot.raw_action_probabilities,
            legal_action_probabilities=snapshot.legal_action_probabilities,
            proposed_action=snapshot.selected_action_name,
            executed_action=None,
            executed_by=None,
            expert_action=None,
            expert_label_available=False,
            expert_error=None,
            remaining_value=snapshot.remaining_value,
            event_estimate_local=snapshot.event_estimate_local,
            action_history=self._history_names(snapshot.action_history),
            model_seconds=elapsed,
        )
        return ExplicitBeliefPolicyDecision(
            action=action,
            awareness=awareness,
            belief=snapshot.belief.copy(),
            policy_snapshot=snapshot,
            model_seconds=elapsed,
        )

    def bind_execution(
        self,
        decision,
        executed_action,
        executed_by,
        expert_action=None,
        expert_error=None,
    ):
        executed_name = strict_action_name(executed_action)
        expert_name = (
            None if expert_action is None else strict_action_name(expert_action)
        )
        awareness = replace(
            decision.awareness,
            executed_action=executed_name,
            executed_by=str(executed_by),
            expert_action=expert_name,
            expert_label_available=expert_action is not None,
            expert_error=None if expert_error is None else str(expert_error),
        )
        return replace(decision, action=executed_action, awareness=awareness)

    def bind_no_execution(
        self,
        decision,
        expert_action=None,
        expert_error=None,
    ):
        """Attach diagnostics without claiming that any action was executed."""
        expert_name = (
            None if expert_action is None else strict_action_name(expert_action)
        )
        awareness = replace(
            decision.awareness,
            executed_action=None,
            executed_by=None,
            expert_action=expert_name,
            expert_label_available=expert_action is not None,
            expert_error=None if expert_error is None else str(expert_error),
        )
        return replace(decision, awareness=awareness)

    def commit_executed_action(self, action):
        self.runtime.commit_action_name(strict_action_name(action))
