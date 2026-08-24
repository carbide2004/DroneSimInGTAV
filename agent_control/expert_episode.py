"""Online Stage 2E expert rollout and strict acceptance checks."""

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from .dronesim_client import ScenarioEntityRole, ScenarioTaskState
from .expert_teacher import (
    AgentEpisodeSpec,
    CueGroundedExpert,
    ExpertGenerationError,
    LocalGeometryFacade,
    SOURCE_STOP_MAX_HORIZONTAL_RANGE_METERS,
    SOURCE_STOP_MIN_PROJECTED_SPAN_PIXELS,
    VisibleTrackGrounder,
)
from .research_actions import ResearchActionExecutor, StopAction
from .task_starts import make_agent_observation


@dataclass(frozen=True)
class CueSensitivityResult:
    passed: bool
    removed_track_id: int | None
    evidence_step: int | None
    divergence_step: int | None
    divergence_kind: str | None


@dataclass(frozen=True)
class ExpertEpisodeResult:
    success: bool
    actions: int
    planner_calls: int
    localization_error_m: float | None
    valid_dynamic_cue_observed: bool
    cue_sensitivity: CueSensitivityResult
    message: str
    timing: object


@dataclass(frozen=True)
class ExpertEpisodeTiming:
    observed_steps: int
    setup_seconds: float
    visibility_seconds: float
    scenario_snapshot_seconds: float
    grounding_seconds: float
    teacher_seconds: float
    recording_seconds: float
    action_pose_seconds: float
    action_advance_seconds: float
    action_capture_seconds: float
    action_total_seconds: float
    cue_sensitivity_seconds: float
    geometry_requested_segments: int
    geometry_queried_segments: int
    geometry_cache_hits: int
    geometry_batch_queries: int
    geometry_query_seconds: float
    total_seconds: float


def _entity_map(snapshot):
    return {
        entity.stable_id: entity
        for entity in snapshot.entities
        if entity.exists
    }


def _direction_cosine(previous, current, event_position):
    displacement = np.asarray(
        current.position[:2],
        dtype=np.float64,
    ) - np.asarray(
        previous.position[:2],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(displacement))
    if length <= 1.0e-9:
        return -1.0
    event = np.asarray(event_position[:2], dtype=np.float64)
    origin = np.asarray(previous.position[:2], dtype=np.float64)
    if previous.role == ScenarioEntityRole.FIRE_TRUCK:
        expected = event - origin
    elif previous.role == ScenarioEntityRole.FLEEING_PEDESTRIAN:
        expected = origin - event
    else:
        return -1.0
    expected_length = float(np.linalg.norm(expected))
    if expected_length <= 1.0e-9:
        return -1.0
    return float(
        np.dot(displacement, expected)
        / (length * expected_length)
    )


def _valid_evidence(
    awareness,
    grounder,
    previous_scenario,
    current_scenario,
):
    if previous_scenario is None:
        return False
    previous = _entity_map(previous_scenario)
    current = _entity_map(current_scenario)
    for evidence in awareness.motion_evidence:
        stable_id = grounder.evaluation_stable_id(
            evidence.track_id
        )
        left = previous.get(stable_id)
        right = current.get(stable_id)
        if (
            left is None
            or right is None
            or left.task_state != ScenarioTaskState.ACTIVE
            or right.task_state != ScenarioTaskState.ACTIVE
        ):
            continue
        displacement = math.dist(
            left.position[:2],
            right.position[:2],
        )
        if (
            displacement >= 0.4
            and _direction_cosine(
                left,
                right,
                current_scenario.event_position,
            )
            >= 0.5
        ):
            return True
    return False


def _cue_sensitivity(history, episode_spec, geometry):
    evidence_index = None
    removed_track_id = None
    for index, (_observation, _grounded, decision) in enumerate(
        history
    ):
        if decision.awareness.motion_evidence:
            evidence_index = index
            removed_track_id = decision.awareness.motion_evidence[
                0
            ].track_id
            break
    if evidence_index is None:
        return CueSensitivityResult(
            passed=False,
            removed_track_id=None,
            evidence_step=None,
            divergence_step=None,
            divergence_kind=None,
        )

    ablated = CueGroundedExpert(episode_spec, geometry)
    for index, (observation, grounded, original) in enumerate(history):
        filtered = replace(
            grounded,
            tracks=tuple(
                track
                for track in grounded.tracks
                if track.track_id != removed_track_id
            ),
        )
        try:
            counterfactual = ablated.decide(
                observation,
                filtered,
            )
        except ExpertGenerationError:
            if evidence_index <= index <= evidence_index + 4:
                return CueSensitivityResult(
                    passed=True,
                    removed_track_id=removed_track_id,
                    evidence_step=history[evidence_index][1].step_index,
                    divergence_step=grounded.step_index,
                    divergence_kind="ABLATED_PLANNER_FAILURE",
                )
            raise
        if index < evidence_index:
            continue
        if index > evidence_index + 4:
            break
        if (
            counterfactual.awareness.primary_mode_id
            != original.awareness.primary_mode_id
        ):
            kind = "BELIEF_MODE"
        elif (
            counterfactual.awareness.intent
            != original.awareness.intent
        ):
            kind = "INTENT"
        elif (
            type(counterfactual.action)
            is not type(original.action)
        ):
            kind = "ACTION"
        else:
            continue
        return CueSensitivityResult(
            passed=True,
            removed_track_id=removed_track_id,
            evidence_step=history[evidence_index][1].step_index,
            divergence_step=grounded.step_index,
            divergence_kind=kind,
        )
    return CueSensitivityResult(
        passed=False,
        removed_track_id=removed_track_id,
        evidence_step=history[evidence_index][1].step_index,
        divergence_step=None,
        divergence_kind=None,
    )


def run_expert_episode(
    client,
    session,
    scenario_id,
    audited_start,
    recorder=None,
    capture_timeout_ms=5000,
):
    run_started = time.perf_counter()
    generated = audited_start.generated
    blueprint = generated.blueprint
    episode_spec = AgentEpisodeSpec.from_blueprint(blueprint)
    executor = ResearchActionExecutor(
        client,
        session,
        generated.rgbd_pair,
        blueprint,
    )
    geometry = LocalGeometryFacade(
        client,
        session,
        blueprint,
    )
    grounder = VisibleTrackGrounder(
        blueprint,
        blueprint.observation_spec,
    )
    teacher = CueGroundedExpert(episode_spec, geometry)
    history = []
    previous_scenario = None
    valid_dynamic_cue = False
    observed_steps = 0
    visibility_seconds = 0.0
    scenario_snapshot_seconds = 0.0
    grounding_seconds = 0.0
    teacher_seconds = 0.0
    recording_seconds = 0.0
    action_pose_seconds = 0.0
    action_advance_seconds = 0.0
    action_capture_seconds = 0.0
    action_total_seconds = 0.0
    cue_sensitivity_seconds = 0.0

    if recorder is not None:
        recorder.write_metadata(
            agent={
                "episode_spec": episode_spec,
                "observation_spec": blueprint.observation_spec,
            },
            teacher={
                "teacher": "cue-grounded-stage2e-v2-close-source",
                "belief_cell_m": 4.0,
                "belief_radius_m": 120.0,
                "source_stop_policy": {
                    "maximum_horizontal_range_m": (
                        SOURCE_STOP_MAX_HORIZONTAL_RANGE_METERS
                    ),
                    "minimum_projected_span_pixels": (
                        SOURCE_STOP_MIN_PROJECTED_SPAN_PIXELS
                    ),
                    "consecutive_grounded_observations": 2,
                },
            },
            evaluation_truth={
                "start_blueprint": blueprint,
                "goal_budget_audit": audited_start.goal_budget_audit,
            },
        )
    setup_seconds = time.perf_counter() - run_started

    def timing_snapshot():
        return ExpertEpisodeTiming(
            observed_steps=observed_steps,
            setup_seconds=setup_seconds,
            visibility_seconds=visibility_seconds,
            scenario_snapshot_seconds=scenario_snapshot_seconds,
            grounding_seconds=grounding_seconds,
            teacher_seconds=teacher_seconds,
            recording_seconds=recording_seconds,
            action_pose_seconds=action_pose_seconds,
            action_advance_seconds=action_advance_seconds,
            action_capture_seconds=action_capture_seconds,
            action_total_seconds=action_total_seconds,
            cue_sensitivity_seconds=cue_sensitivity_seconds,
            geometry_requested_segments=geometry.requested_segments,
            geometry_queried_segments=geometry.queried_segments,
            geometry_cache_hits=geometry.cache_hits,
            geometry_batch_queries=geometry.batch_queries,
            geometry_query_seconds=geometry.query_seconds,
            total_seconds=time.perf_counter() - run_started,
        )

    def accumulate_action_timing(step_result):
        nonlocal action_pose_seconds
        nonlocal action_advance_seconds
        nonlocal action_capture_seconds
        nonlocal action_total_seconds
        action_pose_seconds += step_result.timing.pose_seconds
        action_advance_seconds += step_result.timing.advance_seconds
        action_capture_seconds += step_result.timing.capture_seconds
        action_total_seconds += step_result.timing.total_seconds

    while executor.action_count < episode_spec.horizon_steps:
        observed_steps += 1
        pair = executor.current_pair
        observation = make_agent_observation(
            pair,
            executor.odometry,
        )
        phase_started = time.perf_counter()
        pose = client.get_pose()
        visibility = client.query_visibility(
            scenario_id,
            session.session_id,
            pose[:3],
            timeout=30.0,
        )
        visibility_seconds += time.perf_counter() - phase_started
        if (
            visibility.step_index != pair.clock.step_index
            or visibility.game_timer_ms != pair.clock.game_timer_ms
        ):
            raise ExpertGenerationError(
                "Visibility and RGB-D do not belong to one frozen instant"
            )
        phase_started = time.perf_counter()
        scenario = client.get_scenario_state(scenario_id)
        scenario_snapshot_seconds += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        grounded = grounder.ground(pair, visibility)
        grounding_seconds += time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        decision = teacher.decide(observation, grounded)
        teacher_seconds += time.perf_counter() - phase_started
        history.append((observation, grounded, decision))
        valid_dynamic_cue = valid_dynamic_cue or _valid_evidence(
            decision.awareness,
            grounder,
            previous_scenario,
            scenario,
        )

        truth_record = {
            "step_index": grounded.step_index,
            "event_active": scenario.event_active,
            "event_position": scenario.event_position,
            "entities": scenario.entities,
            "valid_dynamic_cue_so_far": valid_dynamic_cue,
        }
        if recorder is not None:
            phase_started = time.perf_counter()
            recorder.record_step(
                grounded.step_index,
                pair,
                observation,
                grounded,
                decision,
                truth_record,
            )
            recording_seconds += time.perf_counter() - phase_started

        if isinstance(decision.action, StopAction):
            step_result = executor.execute(
                decision.action,
                capture_timeout_ms,
            )
            accumulate_action_timing(step_result)
            if recorder is not None and hasattr(
                recorder,
                "mark_last_action_executed",
            ):
                recorder.mark_last_action_executed()
            estimate_world = blueprint.local_to_world(
                decision.action.event_estimate_local
            )
            error_m = math.dist(
                estimate_world,
                scenario.event_position,
            )
            phase_started = time.perf_counter()
            sensitivity = _cue_sensitivity(
                history,
                episode_spec,
                geometry,
            )
            cue_sensitivity_seconds += (
                time.perf_counter() - phase_started
            )
            success = (
                scenario.event_active
                and error_m <= 5.0
                and valid_dynamic_cue
                and sensitivity.passed
            )
            return ExpertEpisodeResult(
                success=success,
                actions=executor.action_count,
                planner_calls=sum(
                    item[2].awareness.planner_replanned
                    for item in history
                ),
                localization_error_m=error_m,
                valid_dynamic_cue_observed=valid_dynamic_cue,
                cue_sensitivity=sensitivity,
                message=(
                    "PASS"
                    if success
                    else "Terminal checks, dynamic-cue validity, or "
                    "cue sensitivity failed"
                ),
                timing=timing_snapshot(),
            )

        if executor.action_count >= episode_spec.horizon_steps - 1:
            return ExpertEpisodeResult(
                success=False,
                actions=executor.action_count,
                planner_calls=sum(
                    item[2].awareness.planner_replanned
                    for item in history
                ),
                localization_error_m=None,
                valid_dynamic_cue_observed=valid_dynamic_cue,
                cue_sensitivity=CueSensitivityResult(
                    False,
                    None,
                    None,
                    None,
                    None,
                ),
                message=(
                    "TASK_HORIZON_EXHAUSTED_WITHOUT_STOP: teacher "
                    f"proposed {type(decision.action).__name__} when "
                    "the final action was reserved for STOP"
                ),
                timing=timing_snapshot(),
            )

        step_result = executor.execute(
            decision.action,
            capture_timeout_ms,
        )
        accumulate_action_timing(step_result)
        if recorder is not None and hasattr(
            recorder,
            "mark_last_action_executed",
        ):
            recorder.mark_last_action_executed()
        previous_scenario = scenario

    return ExpertEpisodeResult(
        success=False,
        actions=executor.action_count,
        planner_calls=sum(
            item[2].awareness.planner_replanned
            for item in history
        ),
        localization_error_m=None,
        valid_dynamic_cue_observed=valid_dynamic_cue,
        cue_sensitivity=CueSensitivityResult(
            False,
            None,
            None,
            None,
            None,
        ),
        message="TASK_HORIZON_EXHAUSTED without STOP",
        timing=timing_snapshot(),
    )
