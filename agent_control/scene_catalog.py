"""Scene-level Stage 2E start candidates.

A catalog performs the expensive dynamic truth query once for a scenario
blueprint.  Episode attempts then materialize exactly one candidate with the
real camera and RGB-D grounder.  Fire-envelope visibility is retained in the
underlying diagnostics but never participates in start acceptance.
"""

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass

from .dronesim_client import (
    LockstepSession,
    ScenarioEntityRole,
    ScenarioLifecycle,
    TargetVisibilityBatchSnapshot,
    TargetVisibilityCase,
    VisibilityTargetRole,
)
from .feasibility import StaticGoalBudgetAudit
from .start_pool import (
    StaticStartPool,
    _best_goal_audit,
    _merged_visibility,
)
from .task_starts import (
    ActivityBounds,
    GeneratedTaskStart,
    ObservationSpec,
    StartVisibilityStratum,
    TASK_HORIZON_STEPS,
    TaskActionSpec,
    TaskStartBlueprint,
    TaskStartGenerationError,
    _event_bearing_body,
    _require_observation_spec,
    _require_visibility_instant,
    _select_cue_visible_yaw,
    _start_id,
    assess_visibility,
    pair_view_matrices,
    virtual_view_matrices,
)


SCENE_START_CATALOG_SCHEMA_VERSION = 2
SCENE_START_CATALOG_ALGORITHM = "source-only-early-response-v2"


@dataclass(frozen=True)
class SceneStartCandidate:
    pool_start_id: int
    yaw_degrees: float
    responder_ids: tuple
    goal_budget_audit: StaticGoalBudgetAudit
    projected_span_pixels: float
    clear_samples: int


@dataclass(frozen=True)
class SceneStartCatalogTiming:
    dynamic_response_query: float
    yaw_selection: float
    total: float


@dataclass(frozen=True)
class SceneStartCatalog:
    scenario_blueprint_id: int
    scenario_seed: int
    pool_digest: str
    step_index: int
    real_rgbd_certified: bool
    candidates: tuple
    timing: SceneStartCatalogTiming
    digest: str


@dataclass(frozen=True)
class SceneStartMaterializationTiming:
    real_camera_verify: float
    candidate_validation: float
    total: float


def _catalog_payload(catalog, include_digest):
    payload = {
        "schema_version": SCENE_START_CATALOG_SCHEMA_VERSION,
        "algorithm": SCENE_START_CATALOG_ALGORITHM,
        "scenario_blueprint_id": int(catalog.scenario_blueprint_id),
        "scenario_seed": int(catalog.scenario_seed),
        "pool_digest": str(catalog.pool_digest),
        "step_index": int(catalog.step_index),
        "real_rgbd_certified": bool(catalog.real_rgbd_certified),
        "candidates": [
            {
                "pool_start_id": int(candidate.pool_start_id),
                "yaw_degrees": float(candidate.yaw_degrees),
                "responder_ids": [
                    int(value) for value in candidate.responder_ids
                ],
                "goal_budget_audit": asdict(
                    candidate.goal_budget_audit
                ),
                "projected_span_pixels": float(
                    candidate.projected_span_pixels
                ),
                "clear_samples": int(candidate.clear_samples),
            }
            for candidate in catalog.candidates
        ],
        "timing": asdict(catalog.timing),
    }
    if include_digest:
        payload["digest"] = str(catalog.digest)
    return payload


def _digest_payload(payload):
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.blake2b(
        canonical, digest_size=16, person=b"SceneCatalog"
    ).hexdigest()


def _catalog_digest(catalog):
    payload = _catalog_payload(catalog, False)
    # Wall-clock timings and plugin-local runtime IDs are not catalog
    # semantics. Scene identity is validated separately by the immutable
    # blueprint signature in the collection manifest.
    payload.pop("timing", None)
    payload.pop("scenario_blueprint_id", None)
    return _digest_payload(payload)


def _with_digest(catalog):
    digest = _catalog_digest(catalog)
    return SceneStartCatalog(
        scenario_blueprint_id=catalog.scenario_blueprint_id,
        scenario_seed=catalog.scenario_seed,
        pool_digest=catalog.pool_digest,
        step_index=catalog.step_index,
        real_rgbd_certified=catalog.real_rgbd_certified,
        candidates=catalog.candidates,
        timing=catalog.timing,
        digest=digest,
    )


def _observable_responder_ids(assessment, responder_ids):
    allowed = {int(value) for value in responder_ids}
    return tuple(sorted(
        int(target.stable_id)
        for target in assessment.targets
        if int(target.stable_id) in allowed
        and target.role in (
            VisibilityTargetRole.FIRE_TRUCK,
            VisibilityTargetRole.FLEEING_PEDESTRIAN,
        )
        and (
            target.oblique.task_observable
            or target.nadir.task_observable
        )
    ))


def _response_quality(assessment, responder_ids):
    allowed = {int(value) for value in responder_ids}
    views = [
        view
        for target in assessment.targets
        if int(target.stable_id) in allowed
        and target.role in (
            VisibilityTargetRole.FIRE_TRUCK,
            VisibilityTargetRole.FLEEING_PEDESTRIAN,
        )
        for view in (target.oblique, target.nadir)
        if view.task_observable
    ]
    if not views:
        return None
    return (
        max(float(view.projected_span_pixels) for view in views),
        max(int(view.clear_in_frustum_samples) for view in views),
    )


def build_scene_start_catalog(
    client,
    session,
    scenario,
    observation_spec,
    start_pool,
    minimum_entries,
    horizon_steps=TASK_HORIZON_STEPS,
):
    """Build one deterministic truth-level start catalog for a scene."""
    if not isinstance(session, LockstepSession):
        raise TypeError("session must be a LockstepSession")
    if not isinstance(observation_spec, ObservationSpec):
        raise TypeError("observation_spec must be an ObservationSpec")
    if not isinstance(start_pool, StaticStartPool):
        raise TypeError("start_pool must be a StaticStartPool")
    if scenario.lifecycle != ScenarioLifecycle.RUNNING:
        raise ValueError("Scene catalogs require a RUNNING scenario")
    if not scenario.event_active:
        raise ValueError("Scene catalogs require an active fire")
    minimum_entries = int(minimum_entries)
    if not 1 <= minimum_entries <= len(start_pool.entries):
        raise ValueError(
            "minimum_entries must fit the available static start pool"
        )
    if tuple(round(value, 3) for value in scenario.event_position) != tuple(
        round(value, 3) for value in start_pool.event_position
    ):
        raise TaskStartGenerationError(
            "ANCHOR_POOL_MISMATCH: event position changed"
        )

    clock = session.refresh()
    responders = tuple(
        entity
        for entity in scenario.entities
        if entity.exists
        and entity.role in (
            ScenarioEntityRole.FIRE_TRUCK,
            ScenarioEntityRole.FLEEING_PEDESTRIAN,
        )
        and int(entity.planned_activation_offset_ms) <= 2000
    )
    if not responders:
        raise TaskStartGenerationError(
            "SCENE_START_CATALOG_INSUFFICIENT: no response entity is "
            "scheduled inside the two-second window"
        )

    started = time.perf_counter()
    dynamic_seconds = 0.0
    yaw_seconds = 0.0
    matches = []
    positions_per_batch = max(1, 64 // len(responders))
    for offset in range(0, len(start_pool.entries), positions_per_batch):
        entries = start_pool.entries[offset:offset + positions_per_batch]
        requested = [
            TargetVisibilityCase(entity.stable_id, entry.position)
            for entry in entries
            for entity in responders
        ]
        phase = time.perf_counter()
        response = client.query_target_visibility_batch(
            scenario.scenario_id,
            session.session_id,
            requested,
            timeout=30.0,
        )
        dynamic_seconds += time.perf_counter() - phase
        _require_visibility_instant(response, clock)
        for entry_index, entry in enumerate(entries):
            cases = response.cases[
                entry_index * len(responders):
                (entry_index + 1) * len(responders)
            ]
            local_batch = TargetVisibilityBatchSnapshot(
                response.scenario_id,
                response.lockstep_session_id,
                response.step_index,
                response.game_timer_ms,
                response.frame_count,
                tuple(cases),
            )
            phase = time.perf_counter()
            yaw = _select_cue_visible_yaw(
                entry.position,
                local_batch,
                observation_spec,
                responders,
                random.Random(
                    int(scenario.seed) ^ int(entry.pool_start_id)
                ),
            )
            yaw_seconds += time.perf_counter() - phase
            if yaw is None:
                continue
            assessment = assess_visibility(
                _merged_visibility(entry, cases, scenario, clock),
                virtual_view_matrices(
                    entry.position, yaw, observation_spec
                ),
                observation_spec,
            )
            # Deliberately source-only: the diagnostic smoke envelope is not
            # a stable rendering primitive and cannot reject a task start.
            if not assessment.event_initially_hidden:
                continue
            responder_ids = _observable_responder_ids(
                assessment,
                (entity.stable_id for entity in responders),
            )
            quality = _response_quality(assessment, responder_ids)
            if not responder_ids or quality is None:
                continue
            goal_audit = _best_goal_audit(
                entry, yaw, start_pool, int(horizon_steps)
            )
            if goal_audit is None:
                continue
            matches.append(SceneStartCandidate(
                pool_start_id=int(entry.pool_start_id),
                yaw_degrees=float(yaw),
                responder_ids=responder_ids,
                goal_budget_audit=goal_audit,
                projected_span_pixels=float(quality[0]),
                clear_samples=int(quality[1]),
            ))

    matches.sort(key=lambda candidate: (
        candidate.goal_budget_audit.lower_bound_total_actions,
        -candidate.projected_span_pixels,
        -candidate.clear_samples,
        candidate.pool_start_id,
    ))
    timing = SceneStartCatalogTiming(
        dynamic_response_query=dynamic_seconds,
        yaw_selection=yaw_seconds,
        total=time.perf_counter() - started,
    )
    catalog = _with_digest(SceneStartCatalog(
        scenario_blueprint_id=int(scenario.blueprint_id),
        scenario_seed=int(scenario.seed),
        pool_digest=str(start_pool.digest),
        step_index=int(clock.step_index),
        real_rgbd_certified=False,
        candidates=tuple(matches),
        timing=timing,
        digest="",
    ))
    if len(catalog.candidates) < minimum_entries:
        error = TaskStartGenerationError(
            "SCENE_START_CATALOG_INSUFFICIENT: projected "
            f"candidates={len(catalog.candidates)}, "
            f"required={minimum_entries}"
        )
        error.scene_start_catalog = catalog
        raise error
    return catalog


def next_scene_start_candidate(catalog, attempted_pool_start_ids=()):
    if not isinstance(catalog, SceneStartCatalog):
        raise TypeError("catalog must be a SceneStartCatalog")
    attempted = {int(value) for value in attempted_pool_start_ids}
    for candidate in catalog.candidates:
        if candidate.pool_start_id not in attempted:
            return candidate
    raise TaskStartGenerationError(
        "SCENE_START_CATALOG_EXHAUSTED: every projected candidate was "
        "attempted once"
    )


def certified_scene_start_catalog(catalog, candidates):
    """Return a digest-bound catalog containing only real RGB-D passes."""
    if not isinstance(catalog, SceneStartCatalog):
        raise TypeError("catalog must be a SceneStartCatalog")
    candidates = tuple(candidates)
    allowed = {candidate.pool_start_id for candidate in catalog.candidates}
    selected = [candidate.pool_start_id for candidate in candidates]
    if (
        not candidates
        or len(set(selected)) != len(selected)
        or not set(selected).issubset(allowed)
    ):
        raise ValueError(
            "Certified candidates must be a unique non-empty catalog subset"
        )
    return _with_digest(SceneStartCatalog(
        scenario_blueprint_id=catalog.scenario_blueprint_id,
        scenario_seed=catalog.scenario_seed,
        pool_digest=catalog.pool_digest,
        step_index=catalog.step_index,
        real_rgbd_certified=True,
        candidates=candidates,
        timing=catalog.timing,
        digest="",
    ))

def _candidate_error(candidate, message):
    error = TaskStartGenerationError(message)
    error.pool_start_id = int(candidate.pool_start_id)
    error.consume_candidate = True
    return error


def materialize_scene_start(
    client,
    session,
    scenario,
    observation_spec,
    start_pool,
    scene_catalog,
    candidate,
    start_seed,
    horizon_steps=TASK_HORIZON_STEPS,
    candidate_validator=None,
):
    """Validate exactly one catalog candidate with the real RGB-D camera."""
    if int(scene_catalog.scenario_seed) != int(scenario.seed):
        raise TaskStartGenerationError(
            "SCENE_START_CATALOG_MISMATCH: scenario seed changed"
        )
    if scene_catalog.pool_digest != start_pool.digest:
        raise TaskStartGenerationError(
            "SCENE_START_CATALOG_MISMATCH: static pool changed"
        )
    entries = {
        int(entry.pool_start_id): entry for entry in start_pool.entries
    }
    entry = entries.get(int(candidate.pool_start_id))
    if entry is None:
        raise TaskStartGenerationError(
            "SCENE_START_CATALOG_MISMATCH: candidate is absent from pool"
        )
    clock = session.refresh()
    if int(clock.step_index) != int(scene_catalog.step_index):
        raise TaskStartGenerationError(
            "SCENE_START_CATALOG_MISMATCH: lockstep instant changed"
        )

    started = time.perf_counter()
    verify_started = time.perf_counter()
    with client._operation_lock:
        client.set_camera_pose(
            *entry.position,
            candidate.yaw_degrees,
            collision_check=False,
        )
        pair = session.capture_rgbd_pair()
        _require_observation_spec(pair, observation_spec)
        actual_pose = client.get_pose()
        visibility = client.query_visibility(
            scenario.scenario_id,
            session.session_id,
            actual_pose[:3],
            timeout=30.0,
        )
    _require_visibility_instant(visibility, pair.clock)
    assessment = assess_visibility(
        visibility, pair_view_matrices(pair), observation_spec
    )
    verify_seconds = time.perf_counter() - verify_started
    if not assessment.event_initially_hidden:
        raise _candidate_error(
            candidate,
            "START_CANDIDATE_REJECTED: fire source vehicle is visible",
        )
    observed_ids = set(_observable_responder_ids(
        assessment, candidate.responder_ids
    ))
    if not observed_ids:
        raise _candidate_error(
            candidate,
            "START_CANDIDATE_REJECTED: projected early response is not "
            "observable with the real camera",
        )

    goal_audit = _best_goal_audit(
        entry, actual_pose[5], start_pool, int(horizon_steps)
    )
    if goal_audit is None:
        raise _candidate_error(
            candidate,
            "START_CANDIDATE_REJECTED: goal action margin changed",
        )
    measured_agl = float(actual_pose[2] - entry.ground_z)
    blueprint = TaskStartBlueprint(
        start_id=_start_id(
            scenario.blueprint_id,
            int(start_seed),
            StartVisibilityStratum.POTENTIAL_CUE_VISIBLE,
            int(entry.pool_start_id & 0x7FFFFFFF),
        ),
        scenario_blueprint_id=int(scenario.blueprint_id),
        start_seed=int(start_seed),
        candidate_index=int(entry.pool_start_id & 0x7FFFFFFF),
        absolute_pose=tuple(float(value) for value in actual_pose),
        altitude_agl=measured_agl,
        event_distance=math.hypot(
            float(actual_pose[0]) - float(scenario.event_position[0]),
            float(actual_pose[1]) - float(scenario.event_position[1]),
        ),
        event_bearing_body_degrees=_event_bearing_body(
            actual_pose[:3], actual_pose[5], scenario.event_position
        ),
        visibility_stratum=StartVisibilityStratum.POTENTIAL_CUE_VISIBLE,
        observation_spec=observation_spec,
        activity_bounds=ActivityBounds(),
        action_spec=TaskActionSpec(horizon_steps=int(horizon_steps)),
    )
    generated = GeneratedTaskStart(
        blueprint=blueprint,
        rgbd_pair=pair,
        visibility=visibility,
        assessment=assessment,
        rejection_counts=(),
    )
    validation_started = time.perf_counter()
    validation_result = None
    if candidate_validator is not None:
        validation_result = candidate_validator(
            generated,
            candidate.pool_start_id,
            1,
            candidate.responder_ids,
        )
        if validation_result is None:
            raise _candidate_error(
                candidate,
                "START_CANDIDATE_REJECTED: initial RGB-D did not ground "
                "the projected response entity",
            )
    validation_seconds = time.perf_counter() - validation_started
    return (
        generated,
        goal_audit,
        validation_result,
        SceneStartMaterializationTiming(
            real_camera_verify=verify_seconds,
            candidate_validation=validation_seconds,
            total=time.perf_counter() - started,
        ),
    )


def scene_catalog_to_json(catalog):
    if not isinstance(catalog, SceneStartCatalog):
        raise TypeError("catalog must be a SceneStartCatalog")
    expected = _catalog_digest(catalog)
    if catalog.digest != expected:
        raise ValueError("Scene-start catalog digest is invalid")
    return _catalog_payload(catalog, True)


def scene_catalog_from_json(payload):
    if not isinstance(payload, dict):
        raise ValueError("Scene-start catalog must be a JSON object")
    if int(payload.get("schema_version", -1)) != SCENE_START_CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported scene-start catalog schema")
    if payload.get("algorithm") != SCENE_START_CATALOG_ALGORITHM:
        raise ValueError("Unsupported scene-start catalog algorithm")
    candidates = []
    for item in payload.get("candidates", ()):
        audit_payload = item["goal_budget_audit"]
        audit = StaticGoalBudgetAudit(
            goal_pose=tuple(float(value) for value in audit_payload["goal_pose"]),
            lower_bound_nonterminal_actions=int(audit_payload["lower_bound_nonterminal_actions"]),
            stop_actions=int(audit_payload["stop_actions"]),
            lower_bound_total_actions=int(audit_payload["lower_bound_total_actions"]),
            required_reserve_actions=int(audit_payload["required_reserve_actions"]),
            remaining_actions=int(audit_payload["remaining_actions"]),
            candidate_ideals=int(audit_payload["candidate_ideals"]),
            clear_ideals=int(audit_payload["clear_ideals"]),
            observable_ideals=int(audit_payload["observable_ideals"]),
        )
        candidates.append(SceneStartCandidate(
            pool_start_id=int(item["pool_start_id"]),
            yaw_degrees=float(item["yaw_degrees"]),
            responder_ids=tuple(int(value) for value in item["responder_ids"]),
            goal_budget_audit=audit,
            projected_span_pixels=float(item["projected_span_pixels"]),
            clear_samples=int(item["clear_samples"]),
        ))
    timing_payload = payload["timing"]
    catalog = SceneStartCatalog(
        scenario_blueprint_id=int(payload["scenario_blueprint_id"]),
        scenario_seed=int(payload["scenario_seed"]),
        pool_digest=str(payload["pool_digest"]),
        step_index=int(payload["step_index"]),
        real_rgbd_certified=bool(payload["real_rgbd_certified"]),
        candidates=tuple(candidates),
        timing=SceneStartCatalogTiming(
            dynamic_response_query=float(timing_payload["dynamic_response_query"]),
            yaw_selection=float(timing_payload["yaw_selection"]),
            total=float(timing_payload["total"]),
        ),
        digest=str(payload.get("digest", "")),
    )
    expected = _catalog_digest(catalog)
    if catalog.digest != expected:
        raise ValueError("Scene-start catalog digest mismatch")
    return catalog
