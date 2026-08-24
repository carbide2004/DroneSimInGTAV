"""Stage 2E real-camera start checks driven by a scene catalog."""

import time
from collections import Counter
from dataclasses import dataclass

from .dronesim_client import ScenarioEntityRole
from .expert_teacher import VisibleTrackGrounder
from .feasibility import StaticGoalBudgetAudit
from .scene_catalog import (
    SceneStartCatalog,
    certified_scene_start_catalog,
    materialize_scene_start,
    next_scene_start_candidate,
)
from .start_pool import StaticStartPool
from .task_starts import (
    GeneratedTaskStart,
    TASK_HORIZON_STEPS,
    TaskStartGenerationError,
)


@dataclass(frozen=True)
class AuditedStartTiming:
    attempts: int
    task_start_generation_seconds: float
    rgbd_grounding_seconds: float
    static_goal_budget_audit_seconds: float
    total_seconds: float
    dynamic_response_query_seconds: float = 0.0
    yaw_selection_seconds: float = 0.0
    real_camera_verify_seconds: float = 0.0


@dataclass(frozen=True)
class AuditedTaskStart:
    generated: GeneratedTaskStart
    goal_budget_audit: StaticGoalBudgetAudit
    attempt_index: int
    bearing_bin: int
    altitude_bin: str
    initial_grounded_response_count: int
    timing: AuditedStartTiming
    pool_start_id: int


def _bearing_bin(degrees):
    return int(((float(degrees) + 180.0) % 360.0) // 45.0)


def _altitude_bin(altitude_agl):
    return "25-42.5" if float(altitude_agl) < 42.5 else "42.5-60"


def _ground_expected_responses(
    generated,
    observation_spec,
    expected_responder_ids,
):
    grounder = VisibleTrackGrounder(
        generated.blueprint, observation_spec
    )
    grounded = grounder.ground(
        generated.rgbd_pair, generated.visibility
    )
    if any(
        track.semantic_class == "FIRE_SOURCE"
        for track in grounded.tracks
    ):
        raise TaskStartGenerationError(
            "START_CANDIDATE_REJECTED: fire source was recovered "
            "from the initial RGB-D"
        )
    expected = {int(value) for value in expected_responder_ids}
    return tuple(
        track
        for track in grounded.tracks
        if track.semantic_class in ("FIRE_TRUCK", "PEDESTRIAN")
        and grounder.evaluation_stable_id(track.track_id) in expected
    )


def certify_scene_start_catalog_rgbd(
    client,
    session,
    scenario,
    observation_spec,
    start_pool,
    scene_catalog,
    minimum_entries,
    required_entries=None,
    horizon_steps=TASK_HORIZON_STEPS,
    progress_callback=None,
):
    """Filter a projected catalog through one real RGB-D check per entry."""
    target_entries = int(minimum_entries)
    required_entries = (
        target_entries
        if required_entries is None
        else int(required_entries)
    )
    if not 1 <= required_entries <= len(scene_catalog.candidates):
        raise ValueError(
            "required_entries must fit the projected scene catalog"
        )
    target_entries = max(
        required_entries, min(target_entries, len(scene_catalog.candidates))
    )
    if scene_catalog.real_rgbd_certified:
        if len(scene_catalog.candidates) < required_entries:
            raise TaskStartGenerationError(
                "SCENE_START_CATALOG_INSUFFICIENT: certified catalog is "
                "smaller than the required scene quota"
            )
        return scene_catalog
    passed = []
    rejection_counts = Counter()
    for index, candidate in enumerate(scene_catalog.candidates, 1):
        def validate(generated, _pool_start_id, _attempt, expected_ids):
            tracks = _ground_expected_responses(
                generated, observation_spec, expected_ids
            )
            return tracks or None

        try:
            materialize_scene_start(
                client,
                session,
                scenario,
                observation_spec,
                start_pool,
                scene_catalog,
                candidate,
                int(scenario.seed) ^ int(candidate.pool_start_id),
                horizon_steps=horizon_steps,
                candidate_validator=validate,
            )
        except TaskStartGenerationError as error:
            message = str(error)
            if not message.startswith("START_CANDIDATE_REJECTED:"):
                raise
            reason = message.split(":", 1)[1].strip()
            rejection_counts[reason] += 1
            if progress_callback is not None:
                progress_callback(
                    "scene catalog RGB-D REJECT "
                    f"pool_start={candidate.pool_start_id} "
                    f"checked={index}/{len(scene_catalog.candidates)} "
                    f"reason={reason}"
                )
            continue
        passed.append(candidate)
        if progress_callback is not None:
            progress_callback(
                "scene catalog RGB-D PASS "
                f"certified={len(passed)}/{target_entries} "
                f"pool_start={candidate.pool_start_id}"
            )
        if len(passed) >= target_entries:
            return certified_scene_start_catalog(scene_catalog, passed)
    if len(passed) >= required_entries:
        if progress_callback is not None:
            progress_callback(
                "SCENE_CATALOG_RESERVE_SHORTFALL "
                f"certified={len(passed)} required={required_entries} "
                f"target={target_entries} "
                f"rejections={dict(rejection_counts)}"
            )
        return certified_scene_start_catalog(scene_catalog, passed)
    error = TaskStartGenerationError(
        "SCENE_START_CATALOG_INSUFFICIENT: real RGB-D certified "
        f"candidates={len(passed)}, required={required_entries}, "
        f"target={target_entries}, "
        f"projected={len(scene_catalog.candidates)}, "
        f"rejections={dict(rejection_counts)}"
    )
    error.scene_start_catalog = scene_catalog
    raise error

def generate_audited_task_start(
    client,
    session,
    scenario,
    observation_spec,
    start_seed,
    start_pool,
    scene_catalog,
    attempted_pool_start_ids=(),
    progress_callback=None,
    horizon_steps=TASK_HORIZON_STEPS,
):
    """Consume one catalog candidate and perform one authoritative check."""
    if not isinstance(start_pool, StaticStartPool):
        raise TypeError("start_pool must be a StaticStartPool")
    if not isinstance(scene_catalog, SceneStartCatalog):
        raise TypeError("scene_catalog must be a SceneStartCatalog")
    if not scene_catalog.real_rgbd_certified:
        raise ValueError("scene_catalog must be real-RGB-D certified")
    horizon_steps = int(horizon_steps)
    if not 21 <= horizon_steps <= 256:
        raise ValueError("horizon_steps must be in [21, 256]")

    started = time.perf_counter()
    candidate = next_scene_start_candidate(
        scene_catalog, attempted_pool_start_ids
    )
    early_responder_ids = {
        int(entity.stable_id)
        for entity in scenario.entities
        if entity.exists
        and entity.role in (
            ScenarioEntityRole.FIRE_TRUCK,
            ScenarioEntityRole.FLEEING_PEDESTRIAN,
        )
        and entity.planned_activation_offset_ms <= 2000
    }

    def validate_grounded_candidate(
        generated, pool_start_id, _attempt, expected_responder_ids
    ):
        expected = {
            int(value) for value in expected_responder_ids
        } & early_responder_ids
        try:
            response_tracks = _ground_expected_responses(
                generated, observation_spec, expected
            )
        except TaskStartGenerationError as error:
            error.pool_start_id = int(pool_start_id)
            error.consume_candidate = True
            raise
        if not response_tracks:
            if progress_callback is not None:
                progress_callback(
                    "scene start grounder REJECT "
                    f"pool_start={pool_start_id}"
                )
            return None
        return response_tracks
    try:
        (
            generated,
            goal_budget_audit,
            response_tracks,
            materialization_timing,
        ) = materialize_scene_start(
            client,
            session,
            scenario,
            observation_spec,
            start_pool,
            scene_catalog,
            candidate,
            start_seed,
            horizon_steps=horizon_steps,
            candidate_validator=validate_grounded_candidate,
        )
    except TaskStartGenerationError as error:
        if not hasattr(error, "pool_start_id"):
            error.pool_start_id = int(candidate.pool_start_id)
            error.consume_candidate = True
        raise

    total_seconds = time.perf_counter() - started
    candidate_index = next(
        index
        for index, item in enumerate(scene_catalog.candidates)
        if item.pool_start_id == candidate.pool_start_id
    )
    timing = AuditedStartTiming(
        attempts=1,
        task_start_generation_seconds=(
            total_seconds
            - materialization_timing.candidate_validation
        ),
        rgbd_grounding_seconds=(
            materialization_timing.candidate_validation
        ),
        static_goal_budget_audit_seconds=0.0,
        total_seconds=total_seconds,
        real_camera_verify_seconds=(
            materialization_timing.real_camera_verify
        ),
    )
    if progress_callback is not None:
        progress_callback(
            "scene start PASS "
            f"pool_start={candidate.pool_start_id} "
            f"catalog_rank={candidate_index + 1}/"
            f"{len(scene_catalog.candidates)} "
            f"time[verify={timing.real_camera_verify_seconds:.1f}s "
            f"ground={timing.rgbd_grounding_seconds:.1f}s "
            f"total={timing.total_seconds:.1f}s]"
        )
    return AuditedTaskStart(
        generated=generated,
        goal_budget_audit=goal_budget_audit,
        attempt_index=candidate_index,
        bearing_bin=_bearing_bin(
            generated.blueprint.event_bearing_body_degrees
        ),
        altitude_bin=_altitude_bin(
            generated.blueprint.altitude_agl
        ),
        initial_grounded_response_count=len(response_tracks),
        timing=timing,
        pool_start_id=int(candidate.pool_start_id),
    )
