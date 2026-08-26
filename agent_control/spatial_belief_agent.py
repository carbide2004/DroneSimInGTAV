"""Online Spatial RNN belief agent using the shared fixed action policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from learning.belief_features import GroundedTrackFeatureConfig
from learning.online_spatial_belief import (
    OnlineBeliefSnapshot,
    OnlineSpatialBeliefRuntime,
)

from .expert_teacher import (
    BeliefNavigationController,
    SpatialBelief,
)


NAVIGATION_MAX_ENTROPY = 6.5
NAVIGATION_MAX_CREDIBLE_80_AREA_M2 = 8000.0


class ExternalSpatialBelief(SpatialBelief):
    """Planner-facing normalized belief whose update is owned by the RNN."""

    def replace(self, probability) -> None:
        values = np.asarray(probability, dtype=np.float64)
        if values.shape != self.probability.shape:
            raise RuntimeError("ONLINE_BELIEF_GRID_MISMATCH")
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values[~self.valid] != 0.0)
        ):
            raise RuntimeError("ONLINE_BELIEF_INVALID")
        total = float(values.sum())
        if not math.isfinite(total) or abs(total - 1.0) > 2.0e-5:
            raise RuntimeError("ONLINE_BELIEF_NOT_NORMALIZED")
        self.probability = values.copy()
        self.ambiguous = False


@dataclass(frozen=True)
class OnlineSpatialAwareness:
    mode: str
    checkpoint_name: str
    checkpoint_epoch: int
    source_seen: bool
    source_visible_now: bool
    inference_started: bool
    belief_updated: bool
    evidence_track_ids: tuple[int, ...]
    belief_entropy: float
    map_cell: tuple[int, int]
    map_local_xy: tuple[float, float]
    credible_areas_m2: tuple[float, float, float]
    belief_navigation_ready: bool
    belief_navigation_reason: str
    model_seconds: float
    planner_seconds: float
    navigation: object


@dataclass(frozen=True)
class OnlineSpatialDecision:
    action: object
    awareness: OnlineSpatialAwareness
    belief: np.ndarray
    belief_snapshot: OnlineBeliefSnapshot
    model_seconds: float
    planner_seconds: float


class OnlineSpatialBeliefAgent:
    """Compose a source-blind RNN posterior with the Stage 2 action policy."""

    def __init__(
        self,
        episode_spec,
        observation_spec,
        geometry,
        checkpoint_path,
        mode="control",
        device="auto",
    ):
        mode = str(mode).lower()
        if mode not in ("shadow", "control"):
            raise ValueError("mode must be shadow or control")
        self.mode = mode
        self.runtime = OnlineSpatialBeliefRuntime(
            checkpoint_path,
            GroundedTrackFeatureConfig(
                radius_m=120.0,
                vertical_bound_m=float(episode_spec.activity_vertical_m),
                horizon_steps=int(episode_spec.horizon_steps),
                width=int(observation_spec.width),
                height=int(observation_spec.height),
            ),
            device=device,
        )
        self._external_belief = ExternalSpatialBelief()
        self._belief_navigation_ready = False
        controller_belief = (
            self._external_belief if mode == "control" else SpatialBelief()
        )
        self._controller = BeliefNavigationController(
            episode_spec,
            geometry,
            belief=controller_belief,
            update_belief=(mode == "shadow"),
            belief_navigation_allowed=(
                (lambda: self._belief_navigation_ready)
                if mode == "control"
                else None
            ),
        )

    @property
    def checkpoint(self):
        return self.runtime.checkpoint

    def reset(self) -> None:
        raise RuntimeError(
            "OnlineSpatialBeliefAgent is episode-scoped; construct a new agent on reset"
        )

    @staticmethod
    def _navigation_status(snapshot):
        if not snapshot.inference_started:
            return False, "NO_DYNAMIC_EVIDENCE"
        if snapshot.source_seen:
            return False, "SOURCE_CONFIRMATION"
        if snapshot.entropy > NAVIGATION_MAX_ENTROPY:
            return (
                False,
                f"ENTROPY_ABOVE_{NAVIGATION_MAX_ENTROPY:.1f}",
            )
        credible_80_area = float(snapshot.credible_areas_m2[1])
        if credible_80_area > NAVIGATION_MAX_CREDIBLE_80_AREA_M2:
            return (
                False,
                "CREDIBLE_80_AREA_ABOVE_"
                f"{NAVIGATION_MAX_CREDIBLE_80_AREA_M2:.0f}M2",
            )
        return True, "CONFIDENT_SOURCE_BLIND_BELIEF"

    def decide(self, observation, grounded) -> OnlineSpatialDecision:
        started = time.perf_counter()
        snapshot = self.runtime.update(grounded)
        if self.mode == "control":
            self._external_belief.replace(snapshot.belief)
            (
                self._belief_navigation_ready,
                belief_navigation_reason,
            ) = self._navigation_status(snapshot)
        else:
            belief_navigation_reason = "SHADOW_MODE"
        model_seconds = time.perf_counter() - started

        started = time.perf_counter()
        navigation = self._controller.decide(observation, grounded)
        planner_seconds = time.perf_counter() - started
        awareness = OnlineSpatialAwareness(
            mode=self.mode,
            checkpoint_name=self.runtime.checkpoint_path.name,
            checkpoint_epoch=int(self.checkpoint["epoch"]),
            source_seen=snapshot.source_seen,
            source_visible_now=snapshot.source_visible_now,
            inference_started=snapshot.inference_started,
            belief_updated=snapshot.belief_updated,
            evidence_track_ids=snapshot.evidence_track_ids,
            belief_entropy=snapshot.entropy,
            map_cell=snapshot.map_cell,
            map_local_xy=snapshot.map_local_xy,
            credible_areas_m2=snapshot.credible_areas_m2,
            belief_navigation_ready=self._belief_navigation_ready,
            belief_navigation_reason=belief_navigation_reason,
            model_seconds=model_seconds,
            planner_seconds=planner_seconds,
            navigation=navigation.awareness,
        )
        return OnlineSpatialDecision(
            action=navigation.action,
            awareness=awareness,
            belief=snapshot.belief.copy(),
            belief_snapshot=snapshot,
            model_seconds=model_seconds,
            planner_seconds=planner_seconds,
        )
