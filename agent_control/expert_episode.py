"""Online Stage 2E expert rollout and strict acceptance checks."""

import math
from dataclasses import dataclass, replace

import numpy as np

from .dronesim_client import ScenarioEntityRole, ScenarioTaskState
from .expert_teacher import (
    AgentEpisodeSpec,
    CueGroundedExpert,
    ExpertGenerationError,
    LocalGeometryFacade,
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
    certified_start,
    recorder=None,
    capture_timeout_ms=5000,
):
    generated = certified_start.generated
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

    if recorder is not None:
        recorder.write_metadata(
            agent={
                "episode_spec": episode_spec,
                "observation_spec": blueprint.observation_spec,
            },
            teacher={
                "teacher": "cue-grounded-stage2e-v1",
                "belief_cell_m": 4.0,
                "belief_radius_m": 120.0,
            },
            evaluation_truth={
                "start_blueprint": blueprint,
                "static_path_certificate": certified_start.certificate,
            },
        )

    while executor.action_count < episode_spec.horizon_steps:
        pair = executor.current_pair
        observation = make_agent_observation(
            pair,
            executor.odometry,
        )
        pose = client.get_pose()
        visibility = client.query_visibility(
            scenario_id,
            session.session_id,
            pose[:3],
            timeout=30.0,
        )
        if (
            visibility.step_index != pair.clock.step_index
            or visibility.game_timer_ms != pair.clock.game_timer_ms
        ):
            raise ExpertGenerationError(
                "Visibility and RGB-D do not belong to one frozen instant"
            )
        scenario = client.get_scenario_state(scenario_id)
        grounded = grounder.ground(pair, visibility)
        decision = teacher.decide(observation, grounded)
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
            recorder.record_step(
                grounded.step_index,
                pair,
                observation,
                grounded,
                decision,
                truth_record,
            )

        if isinstance(decision.action, StopAction):
            executor.execute(decision.action, capture_timeout_ms)
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
            sensitivity = _cue_sensitivity(
                history,
                episode_spec,
                geometry,
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
            )

        executor.execute(decision.action, capture_timeout_ms)
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
    )
