"""Stage 2E potential-cue-visible starts with a static goal certificate."""

from dataclasses import dataclass

from .dronesim_client import ScenarioEntityRole
from .expert_teacher import VisibleTrackGrounder
from .feasibility import SpatiotemporalFeasibilityAuditor
from .task_starts import (
    StartVisibilityStratum,
    TASK_HORIZON_STEPS,
    TaskStartGenerationError,
    generate_task_start,
)


@dataclass(frozen=True)
class CertifiedTaskStart:
    generated: object
    certificate: object
    attempt_index: int
    path_cost_bin: str
    bearing_bin: int
    altitude_bin: str
    initial_grounded_response_count: int


def _path_cost_bin(cost):
    if 20 <= cost <= 27:
        return "20-27"
    if 28 <= cost <= 35:
        return "28-35"
    if 36 <= cost <= 44:
        return "36-44"
    raise ValueError(f"Path cost {cost} is outside [20, 44]")


def _bearing_bin(degrees):
    return int(((float(degrees) + 180.0) % 360.0) // 45.0)


def _altitude_bin(altitude_agl):
    altitude_agl = float(altitude_agl)
    return "25-42.5" if altitude_agl < 42.5 else "42.5-60"


def generate_certified_task_start(
    client,
    session,
    scenario,
    observation_spec,
    start_seed,
    maximum_attempts=16,
    maximum_candidates_per_attempt=64,
    search_timeout_seconds=120.0,
    progress_callback=None,
    horizon_steps=TASK_HORIZON_STEPS,
):
    maximum_attempts = int(maximum_attempts)
    if not 1 <= maximum_attempts <= 256:
        raise ValueError("maximum_attempts must be in [1, 256]")
    horizon_steps = int(horizon_steps)
    if not 21 <= horizon_steps <= 256:
        raise ValueError("horizon_steps must be in [21, 256]")
    maximum_certificate_actions = min(44, horizon_steps - 1)
    failures = []
    for attempt_index in range(maximum_attempts):
        attempt_seed = (
            int(start_seed) + attempt_index
        ) & 0xFFFFFFFFFFFFFFFF
        try:
            generated = generate_task_start(
                client,
                session,
                scenario,
                observation_spec,
                StartVisibilityStratum.POTENTIAL_CUE_VISIBLE,
                attempt_seed,
                max_candidates=maximum_candidates_per_attempt,
                horizon_steps=horizon_steps,
            )
            grounded = VisibleTrackGrounder(
                generated.blueprint,
                observation_spec,
            )
            grounded_frame = grounded.ground(
                generated.rgbd_pair,
                generated.visibility,
            )
            early_responder_ids = {
                int(entity.stable_id)
                for entity in scenario.entities
                if entity.exists
                and entity.role
                in (
                    ScenarioEntityRole.FIRE_TRUCK,
                    ScenarioEntityRole.FLEEING_PEDESTRIAN,
                )
                and entity.planned_activation_offset_ms <= 2000
            }
            response_tracks = tuple(
                track
                for track in grounded_frame.tracks
                if track.semantic_class
                in ("FIRE_TRUCK", "PEDESTRIAN")
                and grounded.evaluation_stable_id(track.track_id)
                in early_responder_ids
            )
            if not response_tracks:
                raise TaskStartGenerationError(
                    "POTENTIAL_CUE_VISIBLE start has no early response "
                    "track recoverable by the episode RGB-D grounder"
                )
            auditor = SpatiotemporalFeasibilityAuditor(
                client,
                session,
                scenario.scenario_id,
                generated,
                search_timeout_seconds=search_timeout_seconds,
                progress_callback=progress_callback,
            )
            certificate = auditor.certify_static_goal_path(
                minimum_actions=20,
                maximum_actions=maximum_certificate_actions,
            )
            return CertifiedTaskStart(
                generated=generated,
                certificate=certificate,
                attempt_index=attempt_index,
                path_cost_bin=_path_cost_bin(
                    certificate.path_cost
                ),
                bearing_bin=_bearing_bin(
                    generated.blueprint.event_bearing_body_degrees
                ),
                altitude_bin=_altitude_bin(
                    generated.blueprint.altitude_agl
                ),
                initial_grounded_response_count=len(response_tracks),
            )
        except (RuntimeError, TaskStartGenerationError) as error:
            failures.append(
                f"seed={attempt_seed}:{type(error).__name__}:{error}"
            )
    raise TaskStartGenerationError(
        "CERTIFIED_TASK_START_NOT_FOUND after "
        f"{maximum_attempts} attempts; "
        + " | ".join(failures[-4:])
    )
