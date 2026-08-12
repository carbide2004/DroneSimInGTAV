"""Stage 2E potential-cue-visible starts with a static goal certificate."""

from dataclasses import dataclass

from .feasibility import SpatiotemporalFeasibilityAuditor
from .task_starts import (
    StartVisibilityStratum,
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
):
    maximum_attempts = int(maximum_attempts)
    if not 1 <= maximum_attempts <= 256:
        raise ValueError("maximum_attempts must be in [1, 256]")
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
                maximum_actions=44,
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
