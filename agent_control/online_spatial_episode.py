"""Online GTA rollout driven by a Spatial RNN belief posterior."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .expert_episode import _valid_evidence
from .expert_teacher import (
    AgentEpisodeSpec,
    ExpertGenerationError,
    LocalGeometryFacade,
    VisibleTrackGrounder,
)
from .research_actions import (
    AscendAction,
    DescendAction,
    ForwardAction,
    HoldAction,
    ResearchActionExecutor,
    StopAction,
    TurnLeftAction,
    TurnRightAction,
)
from .spatial_belief_agent import OnlineSpatialBeliefAgent
from .task_starts import make_agent_observation


STRICT_ACTION_TYPES = (
    ForwardAction,
    AscendAction,
    DescendAction,
    TurnLeftAction,
    TurnRightAction,
    HoldAction,
    StopAction,
)


@dataclass(frozen=True)
class OnlineSpatialEpisodeTiming:
    observed_steps: int
    visibility_seconds: float
    scenario_seconds: float
    grounding_seconds: float
    model_seconds: float
    planner_seconds: float
    recording_seconds: float
    action_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class OnlineSpatialEpisodeResult:
    success: bool
    status: str
    actions: int
    localization_error_m: float | None
    valid_dynamic_cue_observed: bool
    belief_updates: int
    first_update_step: int | None
    first_source_step: int | None
    inference_event_nll: float | None
    last_source_blind_event_nll: float | None
    last_source_blind_map_error_m: float | None
    planner_calls: int
    message: str
    timing: OnlineSpatialEpisodeTiming


def _source_blind_truth_metrics(agent, snapshot, event_position_local):
    forward_coordinates = (
        agent.runtime.model.grid_forward[:, 0].detach().cpu().numpy()
    )
    right_coordinates = (
        agent.runtime.model.grid_right[0].detach().cpu().numpy()
    )
    row = int(np.argmin(np.abs(forward_coordinates - event_position_local[0])))
    column = int(np.argmin(np.abs(right_coordinates - event_position_local[1])))
    valid = bool(agent.runtime.model.valid_mask[row, column].item())
    if not valid:
        raise ExpertGenerationError(
            "ONLINE_EVENT_OUTSIDE_BELIEF_GRID: evaluation event cell is invalid"
        )
    event_log_probability = float(snapshot.log_belief[row, column])
    if not math.isfinite(event_log_probability):
        raise ExpertGenerationError(
            "ONLINE_EVENT_NLL_INVALID: event log probability is non-finite"
        )
    return (
        -event_log_probability,
        math.dist(snapshot.map_local_xy, event_position_local[:2]),
    )


def run_online_spatial_belief_episode(
    client,
    session,
    scenario_id,
    audited_start,
    checkpoint_path,
    mode="control",
    device="auto",
    recorder=None,
    capture_timeout_ms=5000,
):
    started = time.perf_counter()
    blueprint = audited_start.generated.blueprint
    episode_spec = AgentEpisodeSpec.from_blueprint(blueprint)
    executor = ResearchActionExecutor(
        client, session, audited_start.generated.rgbd_pair, blueprint
    )
    geometry = LocalGeometryFacade(client, session, blueprint)
    grounder = VisibleTrackGrounder(blueprint, blueprint.observation_spec)
    agent = OnlineSpatialBeliefAgent(
        episode_spec,
        blueprint.observation_spec,
        geometry,
        checkpoint_path,
        mode=mode,
        device=device,
    )
    previous_scenario = None
    valid_dynamic_cue = False
    belief_updates = 0
    first_update_step = None
    first_source_step = None
    inference_nll_values = []
    last_source_blind_nll = None
    last_source_blind_map_error = None
    planner_calls = 0
    observed_steps = 0
    visibility_seconds = 0.0
    scenario_seconds = 0.0
    grounding_seconds = 0.0
    model_seconds = 0.0
    planner_seconds = 0.0
    recording_seconds = 0.0
    action_seconds = 0.0

    if recorder is not None:
        recorder.write_metadata(
            agent={
                "episode_spec": episode_spec,
                "observation_spec": blueprint.observation_spec,
                "mode": mode,
            },
            teacher={
                "teacher": "online-spatial-rnn-belief",
                "checkpoint": str(agent.runtime.checkpoint_path),
                "checkpoint_epoch": int(agent.checkpoint["epoch"]),
                "checkpoint_model": agent.checkpoint["model"],
                "source_boundary": agent.checkpoint["source_boundary"],
            },
            evaluation_truth={
                "start_blueprint": blueprint,
                "goal_budget_audit": audited_start.goal_budget_audit,
            },
        )

    def timing():
        return OnlineSpatialEpisodeTiming(
            observed_steps=observed_steps,
            visibility_seconds=visibility_seconds,
            scenario_seconds=scenario_seconds,
            grounding_seconds=grounding_seconds,
            model_seconds=model_seconds,
            planner_seconds=planner_seconds,
            recording_seconds=recording_seconds,
            action_seconds=action_seconds,
            total_seconds=time.perf_counter() - started,
        )

    def result(success, status, actions, localization_error, message):
        return OnlineSpatialEpisodeResult(
            success=success,
            status=status,
            actions=actions,
            localization_error_m=localization_error,
            valid_dynamic_cue_observed=valid_dynamic_cue,
            belief_updates=belief_updates,
            first_update_step=first_update_step,
            first_source_step=first_source_step,
            inference_event_nll=(
                None
                if not inference_nll_values
                else float(np.mean(inference_nll_values))
            ),
            last_source_blind_event_nll=last_source_blind_nll,
            last_source_blind_map_error_m=last_source_blind_map_error,
            planner_calls=planner_calls,
            message=message,
            timing=timing(),
        )

    while executor.action_count < episode_spec.horizon_steps:
        observed_steps += 1
        pair = executor.current_pair
        observation = make_agent_observation(pair, executor.odometry)

        phase = time.perf_counter()
        pose = client.get_pose()
        visibility = client.query_visibility(
            scenario_id, session.session_id, pose[:3], timeout=30.0
        )
        visibility_step_seconds = time.perf_counter() - phase
        visibility_seconds += visibility_step_seconds
        if (
            visibility.step_index != pair.clock.step_index
            or visibility.game_timer_ms != pair.clock.game_timer_ms
        ):
            raise ExpertGenerationError(
                "Visibility and RGB-D do not belong to one frozen instant"
            )

        phase = time.perf_counter()
        scenario = client.get_scenario_state(scenario_id)
        scenario_step_seconds = time.perf_counter() - phase
        scenario_seconds += scenario_step_seconds
        phase = time.perf_counter()
        grounded = grounder.ground(pair, visibility)
        grounding_step_seconds = time.perf_counter() - phase
        grounding_seconds += grounding_step_seconds
        try:
            decision = agent.decide(observation, grounded)
        except ExpertGenerationError as error:
            message = str(error)
            status = message.partition(":")[0].strip()
            if not status or any(character.isspace() for character in status):
                status = "AGENT_DECISION_FAILED"
            return result(
                False,
                status,
                executor.action_count,
                None,
                message,
            )
        model_seconds += decision.model_seconds
        planner_seconds += decision.planner_seconds
        if not isinstance(decision.action, STRICT_ACTION_TYPES):
            raise ExpertGenerationError(
                "INVALID_ONLINE_POLICY_ACTION: learned-belief controller returned "
                f"{type(decision.action).__name__}"
            )

        snapshot = decision.belief_snapshot
        if snapshot.belief_updated:
            belief_updates += 1
            if first_update_step is None:
                first_update_step = grounded.step_index
        if snapshot.source_visible_now and first_source_step is None:
            first_source_step = grounded.step_index
        if snapshot.inference_started and not snapshot.source_seen:
            event_local = blueprint.world_to_local(scenario.event_position)
            last_source_blind_nll, last_source_blind_map_error = (
                _source_blind_truth_metrics(agent, snapshot, event_local)
            )
            inference_nll_values.append(last_source_blind_nll)
        planner_calls += int(decision.awareness.navigation.planner_replanned)
        valid_dynamic_cue = valid_dynamic_cue or _valid_evidence(
            decision.awareness.navigation,
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
            phase = time.perf_counter()
            recorder.record_step(
                grounded.step_index,
                pair,
                observation,
                grounded,
                decision,
                truth_record,
                step_timing={
                    "visibility_seconds": visibility_step_seconds,
                    "scenario_seconds": scenario_step_seconds,
                    "grounding_seconds": grounding_step_seconds,
                    "model_seconds": decision.model_seconds,
                    "planner_seconds": decision.planner_seconds,
                },
            )
            recording_seconds += time.perf_counter() - phase

        if isinstance(decision.action, StopAction):
            step_result = executor.execute(decision.action, capture_timeout_ms)
            action_seconds += step_result.timing.total_seconds
            if recorder is not None:
                recorder.mark_last_action_executed(step_result.timing)
            estimate_world = blueprint.local_to_world(
                decision.action.event_estimate_local
            )
            error_m = math.dist(estimate_world, scenario.event_position)
            success = (
                scenario.event_active
                and snapshot.source_seen
                and valid_dynamic_cue
                and belief_updates > 0
                and error_m <= 5.0
            )
            return result(
                success,
                "PASS" if success else "TERMINAL_CHECK_FAILED",
                executor.action_count,
                error_m,
                "PASS" if success else "STOP did not satisfy all terminal checks",
            )

        if executor.action_count >= episode_spec.horizon_steps - 1:
            return result(
                False,
                "HORIZON_EXHAUSTED",
                executor.action_count,
                None,
                "Final action was reserved for STOP",
            )
        step_result = executor.execute(decision.action, capture_timeout_ms)
        action_seconds += step_result.timing.total_seconds
        if recorder is not None:
            recorder.mark_last_action_executed(step_result.timing)
        previous_scenario = scenario

    return result(
        False,
        "HORIZON_EXHAUSTED",
        executor.action_count,
        None,
        "Task horizon exhausted without STOP",
    )
