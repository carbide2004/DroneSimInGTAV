"""Stage 2E potential-cue-visible starts with a light goal audit."""

import math
import time
from dataclasses import dataclass

from .dronesim_client import ScenarioEntityRole
from .expert_teacher import VisibleTrackGrounder
from .feasibility import (
    SpatiotemporalFeasibilityAuditor,
    StaticGoalBudgetAudit,
)
from .task_starts import (
    GeneratedTaskStart,
    StartVisibilityStratum,
    TASK_HORIZON_STEPS,
    TaskStartGenerationError,
    generate_task_start,
)


@dataclass(frozen=True)
class AuditedStartTiming:
    attempts: int
    task_start_generation_seconds: float
    rgbd_grounding_seconds: float
    static_goal_budget_audit_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class AuditedTaskStart:
    generated: GeneratedTaskStart
    goal_budget_audit: StaticGoalBudgetAudit
    attempt_index: int
    bearing_bin: int
    altitude_bin: str
    initial_grounded_response_count: int
    timing: AuditedStartTiming


def _bearing_bin(degrees):
    return int(((float(degrees) + 180.0) % 360.0) // 45.0)


def _altitude_bin(altitude_agl):
    altitude_agl = float(altitude_agl)
    return "25-42.5" if altitude_agl < 42.5 else "42.5-60"


def generate_audited_task_start(
    client,
    session,
    scenario,
    observation_spec,
    start_seed,
    maximum_attempts=16,
    maximum_candidates_per_attempt=64,
    audit_timeout_seconds=120.0,
    progress_callback=None,
    horizon_steps=TASK_HORIZON_STEPS,
    required_reserve_actions=15,
):
    maximum_attempts = int(maximum_attempts)
    if not 1 <= maximum_attempts <= 256:
        raise ValueError("maximum_attempts must be in [1, 256]")
    audit_timeout_seconds = float(audit_timeout_seconds)
    if (
        not math.isfinite(audit_timeout_seconds)
        or audit_timeout_seconds <= 0.0
    ):
        raise ValueError(
            "audit_timeout_seconds must be finite and positive"
        )
    horizon_steps = int(horizon_steps)
    if not 21 <= horizon_steps <= 256:
        raise ValueError("horizon_steps must be in [21, 256]")
    required_reserve_actions = int(required_reserve_actions)
    if not 0 <= required_reserve_actions < horizon_steps:
        raise ValueError(
            "required_reserve_actions must be in [0, horizon_steps)"
        )
    failures = []
    timing_started = time.perf_counter()
    generation_seconds = 0.0
    grounding_seconds = 0.0
    audit_seconds = 0.0
    for attempt_index in range(maximum_attempts):
        attempt_seed = (
            int(start_seed) + attempt_index
        ) & 0xFFFFFFFFFFFFFFFF
        attempt_started = time.perf_counter()
        phase = "generate"
        phase_started = attempt_started
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
            generation_seconds += time.perf_counter() - phase_started
            phase = "ground"
            phase_started = time.perf_counter()
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
            grounding_seconds += time.perf_counter() - phase_started
            phase = "budget_audit"
            phase_started = time.perf_counter()
            auditor = SpatiotemporalFeasibilityAuditor(
                client,
                session,
                scenario.scenario_id,
                generated,
                search_timeout_seconds=audit_timeout_seconds,
                progress_callback=progress_callback,
            )
            goal_budget_audit = auditor.audit_static_goal_budget(
                required_reserve_actions=required_reserve_actions,
            )
            audit_seconds += time.perf_counter() - phase_started
            phase = "complete"
            timing = AuditedStartTiming(
                attempts=attempt_index + 1,
                task_start_generation_seconds=generation_seconds,
                rgbd_grounding_seconds=grounding_seconds,
                static_goal_budget_audit_seconds=audit_seconds,
                total_seconds=time.perf_counter() - timing_started,
            )
            if progress_callback is not None:
                progress_callback(
                    "audited start PASS "
                    f"attempt={attempt_index + 1} "
                    "time[generate="
                    f"{generation_seconds:.1f}s ground="
                    f"{grounding_seconds:.1f}s budget_audit="
                    f"{audit_seconds:.1f}s total="
                    f"{timing.total_seconds:.1f}s]"
                )
            return AuditedTaskStart(
                generated=generated,
                goal_budget_audit=goal_budget_audit,
                attempt_index=attempt_index,
                bearing_bin=_bearing_bin(
                    generated.blueprint.event_bearing_body_degrees
                ),
                altitude_bin=_altitude_bin(
                    generated.blueprint.altitude_agl
                ),
                initial_grounded_response_count=len(response_tracks),
                timing=timing,
            )
        except (RuntimeError, TaskStartGenerationError) as error:
            if phase == "generate":
                generation_seconds += time.perf_counter() - phase_started
            elif phase == "ground":
                grounding_seconds += time.perf_counter() - phase_started
            elif phase == "budget_audit":
                audit_seconds += time.perf_counter() - phase_started
            failures.append(
                f"seed={attempt_seed}:{type(error).__name__}:{error}"
            )
            if progress_callback is not None:
                progress_callback(
                    "audited start attempt FAIL "
                    f"attempt={attempt_index + 1} "
                    f"wall={time.perf_counter() - attempt_started:.1f}s "
                    f"phase={phase} error={type(error).__name__}"
                )
    error = TaskStartGenerationError(
        "AUDITED_TASK_START_NOT_FOUND after "
        f"{maximum_attempts} attempts; "
        + " | ".join(failures[-4:])
    )
    error.timing = AuditedStartTiming(
        attempts=maximum_attempts,
        task_start_generation_seconds=generation_seconds,
        rgbd_grounding_seconds=grounding_seconds,
        static_goal_budget_audit_seconds=audit_seconds,
        total_seconds=time.perf_counter() - timing_started,
    )
    raise error
