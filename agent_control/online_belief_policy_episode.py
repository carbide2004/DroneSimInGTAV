"""GTA lockstep rollout for the planner-free Stage 3C policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

import numpy as np

from .belief_policy_agent import OnlineExplicitBeliefPolicyAgent, strict_action_name
from .expert_teacher import (
    AgentEpisodeSpec,
    CueGroundedExpert,
    ExpertGenerationError,
    LocalGeometryFacade,
    VisibleTrackGrounder,
)
from .research_actions import ResearchActionExecutor, StopAction
from .task_starts import make_agent_observation
from learning.online_belief_policy import OnlinePolicyError


@dataclass(frozen=True)
class OnlineBeliefPolicyTiming:
    observed_steps: int
    visibility_seconds: float
    scenario_seconds: float
    grounding_seconds: float
    model_seconds: float
    expert_seconds: float
    recording_seconds: float
    action_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class OnlineBeliefPolicyEpisodeResult:
    success: bool
    status: str
    mode: str
    actions: int
    localization_error_m: float | None
    valid_dynamic_cue_observed: bool
    belief_updates: int
    first_update_step: int | None
    first_source_step: int | None
    inference_event_nll: float | None
    last_source_blind_event_nll: float | None
    last_source_blind_map_error_m: float | None
    policy_expert_agreement: float | None
    expert_labels: int
    no_expert_labels: int
    policy_action_counts: tuple[int, ...]
    executed_action_counts: tuple[int, ...]
    mean_policy_action_entropy: float | None
    inference_credible_coverage: tuple[float, float, float] | None
    last_source_blind_credible_coverage: tuple[bool, bool, bool] | None
    message: str
    timing: OnlineBeliefPolicyTiming


def _status_from_error(error):
    message = str(error)
    if "COLLISION_BLOCKED" in message:
        return "ACTION_COLLISION_BLOCKED"
    prefix = message.partition(":")[0].strip()
    if prefix and not any(character.isspace() for character in prefix):
        return prefix
    return "ONLINE_POLICY_ERROR"


def _truth_metrics(agent, snapshot, event_local):
    updater = agent.runtime.model.belief_updater
    forward = updater.grid_forward[:, 0].detach().cpu().numpy()
    right = updater.grid_right[0].detach().cpu().numpy()
    row = int(np.argmin(np.abs(forward - event_local[0])))
    column = int(np.argmin(np.abs(right - event_local[1])))
    if not bool(updater.valid_mask[row, column].item()):
        raise ExpertGenerationError("ONLINE_EVENT_OUTSIDE_BELIEF_GRID")
    log_probability = float(snapshot.log_belief[row, column])
    if not math.isfinite(log_probability):
        raise ExpertGenerationError("ONLINE_EVENT_NLL_INVALID")
    belief = np.asarray(snapshot.belief, dtype=np.float64)
    valid = updater.valid_mask.detach().cpu().numpy()
    flat_valid_indices = np.flatnonzero(valid.reshape(-1))
    order = np.argsort(belief.reshape(-1)[flat_valid_indices])[::-1]
    ordered_indices = flat_valid_indices[order]
    cumulative = np.cumsum(belief.reshape(-1)[ordered_indices])
    event_flat_index = np.ravel_multi_index((row, column), belief.shape)
    coverage = []
    for threshold in (0.5, 0.8, 0.9):
        count = int(np.searchsorted(cumulative, threshold, side="left") + 1)
        coverage.append(bool(np.any(ordered_indices[:count] == event_flat_index)))
    return (
        -log_probability,
        math.dist(snapshot.map_local_xy, event_local[:2]),
        (row, column),
        tuple(coverage),
    )


def run_online_belief_policy_episode(
    client,
    session,
    scenario_id,
    audited_start,
    checkpoint_path,
    mode="control",
    device="auto",
    recorder=None,
    dagger_recorder=None,
    dagger_beta=0.0,
    dagger_seed=0,
    capture_timeout_ms=5000,
):
    mode = str(mode).lower()
    if mode not in ("shadow", "control", "dagger"):
        raise ValueError("mode must be shadow, control, or dagger")
    dagger_beta = float(dagger_beta)
    if mode == "dagger" and not 0.0 <= dagger_beta <= 1.0:
        raise ValueError("dagger_beta must be in [0, 1]")
    started = time.perf_counter()
    blueprint = audited_start.generated.blueprint
    episode_spec = AgentEpisodeSpec.from_blueprint(blueprint)
    executor = ResearchActionExecutor(
        client, session, audited_start.generated.rgbd_pair, blueprint
    )
    grounder = VisibleTrackGrounder(blueprint, blueprint.observation_spec)
    agent = OnlineExplicitBeliefPolicyAgent(
        episode_spec,
        blueprint.observation_spec,
        checkpoint_path,
        mode=mode,
        device=device,
    )
    expert = None
    if mode in ("shadow", "dagger"):
        expert = CueGroundedExpert(
            episode_spec,
            LocalGeometryFacade(client, session, blueprint),
        )
    rng = random.Random(int(dagger_seed))
    valid_dynamic_cue = False
    belief_updates = 0
    first_update_step = None
    first_source_step = None
    inference_nll = []
    last_nll = None
    last_map_error = None
    expert_labels = 0
    no_expert_labels = 0
    agreements = 0
    observed_steps = 0
    visibility_seconds = 0.0
    scenario_seconds = 0.0
    grounding_seconds = 0.0
    policy_action_counts = [0] * 7
    executed_action_counts = [0] * 7
    policy_action_entropies = []
    inference_coverage = []
    last_coverage = None
    model_seconds = 0.0
    expert_seconds = 0.0
    recording_seconds = 0.0
    action_seconds = 0.0

    if recorder is not None:
        recorder.write_metadata(
            agent={
                "episode_spec": episode_spec,
                "observation_spec": blueprint.observation_spec,
                "mode": mode,
                "dagger_beta": dagger_beta if mode == "dagger" else None,
            },
            teacher={
                "teacher": "stage3c-explicit-belief-action-policy",
                "checkpoint": str(agent.runtime.checkpoint_path),
                "checkpoint_epoch": int(agent.checkpoint.get("epoch", -1)),
                "dagger_iteration": int(agent.checkpoint.get("dagger_iteration", 0)),
                "belief_bottleneck": agent.checkpoint.get("belief_bottleneck"),
                "action_names": agent.checkpoint.get("action_names"),
            },
            evaluation_truth={
                "start_blueprint": blueprint,
                "goal_budget_audit": audited_start.goal_budget_audit,
            },
        )

    def timing():
        return OnlineBeliefPolicyTiming(
            observed_steps=observed_steps,
            visibility_seconds=visibility_seconds,
            scenario_seconds=scenario_seconds,
            grounding_seconds=grounding_seconds,
            model_seconds=model_seconds,
            expert_seconds=expert_seconds,
            recording_seconds=recording_seconds,
            action_seconds=action_seconds,
            total_seconds=time.perf_counter() - started,
        )

    def result(success, status, localization_error, message):
        return OnlineBeliefPolicyEpisodeResult(
            success=bool(success),
            status=str(status),
            mode=mode,
            actions=executor.action_count,
            localization_error_m=localization_error,
            valid_dynamic_cue_observed=valid_dynamic_cue,
            belief_updates=belief_updates,
            first_update_step=first_update_step,
            first_source_step=first_source_step,
            inference_event_nll=(None if not inference_nll else float(np.mean(inference_nll))),
            last_source_blind_event_nll=last_nll,
            last_source_blind_map_error_m=last_map_error,
            policy_expert_agreement=(
                None if expert_labels == 0 else agreements / float(expert_labels)
            ),
            expert_labels=expert_labels,
            no_expert_labels=no_expert_labels,
            policy_action_counts=tuple(policy_action_counts),
            executed_action_counts=tuple(executed_action_counts),
            mean_policy_action_entropy=(
                None
                if not policy_action_entropies
                else float(np.mean(policy_action_entropies))
            ),
            inference_credible_coverage=(
                None
                if not inference_coverage
                else tuple(
                    float(value)
                    for value in np.mean(
                        np.asarray(inference_coverage, dtype=np.float64), axis=0
                    )
                )
            ),
            last_source_blind_credible_coverage=last_coverage,
            message=str(message),
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
            learner = agent.predict(
                pair, observation, grounded, executor.action_count
            )
        except OnlinePolicyError as error:
            return result(False, _status_from_error(error), None, error)
        model_seconds += learner.model_seconds
        snapshot = learner.policy_snapshot
        policy_action_counts[snapshot.selected_action_index] += 1
        probabilities = np.asarray(
            snapshot.legal_action_probabilities, dtype=np.float64
        )
        positive = probabilities > 0.0
        policy_action_entropies.append(
            float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        )
        if snapshot.belief_updated:
            belief_updates += 1
            if first_update_step is None:
                first_update_step = observed_steps
        if snapshot.source_visible_now and first_source_step is None:
            first_source_step = observed_steps
        valid_dynamic_cue = valid_dynamic_cue or bool(snapshot.evidence_track_ids)
        event_local = blueprint.world_to_local(scenario.event_position)
        current_nll, current_map_error, event_cell, current_coverage = _truth_metrics(
            agent, snapshot, event_local
        )
        if snapshot.inference_started and not snapshot.source_seen:
            last_nll = current_nll
            last_map_error = current_map_error
            last_coverage = current_coverage
            inference_nll.append(last_nll)
            inference_coverage.append(current_coverage)

        expert_decision = None
        expert_error = None
        expert_step_seconds = 0.0
        if expert is not None:
            phase = time.perf_counter()
            try:
                expert_decision = expert.decide(observation, grounded)
            except ExpertGenerationError as error:
                expert_error = str(error)
                no_expert_labels += 1
            expert_step_seconds = time.perf_counter() - phase
            expert_seconds += expert_step_seconds
        if expert_decision is not None:
            expert_labels += 1
            agreements += int(
                strict_action_name(expert_decision.action)
                == snapshot.selected_action_name
            )

        truth_record = {
            "step_index": grounded.step_index,
            "event_active": scenario.event_active,
            "event_position": scenario.event_position,
            "entities": scenario.entities,
            "valid_dynamic_cue_so_far": valid_dynamic_cue,
        }
        observation_timing = {
            "visibility_seconds": visibility_step_seconds,
            "scenario_seconds": scenario_step_seconds,
            "grounding_seconds": grounding_step_seconds,
            "model_seconds": learner.model_seconds,
            "expert_seconds": expert_step_seconds,
        }

        if mode == "control":
            actual_action = learner.action
            executed_by = "LEARNER"
        elif mode == "shadow":
            if expert_decision is None:
                decision = agent.bind_no_execution(
                    learner, expert_error=expert_error
                )
                if recorder is not None:
                    phase = time.perf_counter()
                    recorder.record_step(
                        grounded.step_index,
                        pair,
                        observation,
                        grounded,
                        decision,
                        truth_record,
                        step_timing=observation_timing,
                    )
                    recorder.mark_last_action_not_executed(
                        "NO_EXPERT_LABEL"
                    )
                    recording_seconds += time.perf_counter() - phase
                return result(False, "NO_EXPERT_LABEL", None, expert_error)
            actual_action = expert_decision.action
            executed_by = "EXPERT"
        else:
            execute_expert = expert_decision is not None and rng.random() < dagger_beta
            actual_action = expert_decision.action if execute_expert else learner.action
            executed_by = "EXPERT" if execute_expert else "LEARNER"
        decision = agent.bind_execution(
            learner,
            actual_action,
            executed_by,
            None if expert_decision is None else expert_decision.action,
            expert_error,
        )

        if dagger_recorder is not None:
            dagger_recorder.record_step(
                step_index=observed_steps,
                training_row=agent.runtime.last_training_row,
                learner_action=snapshot.selected_action_name,
                expert_action=(
                    None if expert_decision is None
                    else strict_action_name(expert_decision.action)
                ),
                executed_by=executed_by,
                event_local=event_local,
                event_cell=event_cell,
            )

        if recorder is not None:
            phase = time.perf_counter()
            recorder.record_step(
                grounded.step_index,
                pair,
                observation,
                grounded,
                decision,
                truth_record,
                step_timing=observation_timing,
            )
            recording_seconds += time.perf_counter() - phase

        if not isinstance(actual_action, StopAction) and (
            executor.action_count >= episode_spec.horizon_steps - 1
        ):
            if recorder is not None:
                recorder.mark_last_action_not_executed(
                    "HORIZON_EXHAUSTED"
                )
            return result(
                False,
                "HORIZON_EXHAUSTED",
                None,
                "Final action slot cannot execute a non-STOP action",
            )
        try:
            step_result = executor.execute(actual_action, capture_timeout_ms)
        except Exception as error:
            if recorder is not None:
                recorder.mark_last_action_not_executed(
                    f"{type(error).__name__}: {error}"
                )
            return result(False, _status_from_error(error), None, error)
        action_names = (
            "FORWARD", "ASCEND", "DESCEND", "TURNLEFT", "TURNRIGHT", "HOLD", "STOP"
        )
        executed_action_counts[
            action_names.index(strict_action_name(actual_action))
        ] += 1
        action_seconds += step_result.timing.total_seconds
        agent.commit_executed_action(actual_action)
        if recorder is not None:
            recorder.mark_last_action_executed(step_result.timing)

        if isinstance(actual_action, StopAction):
            estimate_world = blueprint.local_to_world(
                actual_action.event_estimate_local
            )
            error_m = math.dist(estimate_world, scenario.event_position)
            success = (
                scenario.event_active
                and snapshot.source_visible_now
                and valid_dynamic_cue
                and belief_updates > 0
                and error_m <= 5.0
            )
            return result(
                success,
                "PASS" if success else "TERMINAL_CHECK_FAILED",
                error_m,
                "PASS" if success else "STOP did not satisfy terminal checks",
            )

    return result(
        False,
        "HORIZON_EXHAUSTED",
        None,
        "Task horizon exhausted without STOP",
    )
