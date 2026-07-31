import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent_control.dronesim_client import (  # noqa: E402
    DroneSimClient,
    LockstepSession,
    OBLIQUE_PITCH_DEGREES,
    ScenarioEntityRole,
    ScenarioTaskState,
    VisibilityTargetRole,
)
from agent_control.feasibility import (  # noqa: E402
    FeasibilityStatus,
    REQUIRED_ACTION_MARGIN,
    SpatiotemporalFeasibilityAuditor,
)
from agent_control.research_actions import (  # noqa: E402
    AscendAction,
    DescendAction,
    ForwardAction,
    HoldAction,
    InvalidTaskAction,
    ResearchActionExecutor,
    StopAction,
    TurnLeftAction,
    TurnRightAction,
)
from agent_control.task_starts import (  # noqa: E402
    ObservationSpec,
    StartVisibilityStratum,
    TASK_FORWARD_STEP_METERS,
    TASK_HORIZON_STEPS,
    TASK_MIN_CUE_HORIZONTAL_DISPLACEMENT_METERS,
    TASK_VERTICAL_STEP_METERS,
    TASK_YAW_STEP_DEGREES,
    assess_visibility,
    generate_task_start,
    pair_view_matrices,
)
from validation.trajectory_recording import (  # noqa: E402
    TrajectoryRecorder,
    write_json,
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find and replay a bounded Stage 2D cue-to-goal witness. "
            "No payload is written unless --record-dir is provided."
        )
    )
    parser.add_argument(
        "--anchor",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=TASK_HORIZON_STEPS,
        help=(
            "Agent action horizon; the formal Stage 2D default is "
            f"{TASK_HORIZON_STEPS}"
        ),
    )
    parser.add_argument("--firetrucks", type=int, default=1)
    parser.add_argument("--pedestrians", type=int, default=32)
    parser.add_argument("--prepare-timeout", type=float, default=20.0)
    parser.add_argument(
        "--search-timeout",
        type=float,
        default=120.0,
        help=(
            "Maximum fixed-action search wall time per visibility "
            "stratum in seconds"
        ),
    )
    parser.add_argument("--max-start-candidates", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.5")
    parser.add_argument("--port", type=int, default=23456)
    parser.add_argument("--show-witness", action="store_true")
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument(
        "--record-dir",
        type=Path,
        help=(
            "Write compressed replay RGB and visualization metadata "
            "for both visibility strata to a new directory"
        ),
    )
    parser.add_argument(
        "--record-jpeg-quality",
        type=int,
        default=85,
    )
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in args.anchor):
        parser.error("--anchor values must be finite")
    if not 0 <= args.seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--seed must fit uint64")
    if not 0 <= args.start_seed <= 0xFFFFFFFFFFFFFFFF:
        parser.error("--start-seed must fit uint64")
    if not 8 <= args.horizon_steps <= 256:
        parser.error("--horizon-steps must be in [8, 256]")
    if not 0 <= args.firetrucks <= 4:
        parser.error("--firetrucks must be in [0, 4]")
    if not 0 <= args.pedestrians <= 32:
        parser.error("--pedestrians must be in [0, 32]")
    if args.firetrucks + args.pedestrians == 0:
        parser.error("At least one response actor is required")
    if not 1 <= args.record_jpeg_quality <= 95:
        parser.error("--record-jpeg-quality must be in [1, 95]")
    if (
        not math.isfinite(args.search_timeout)
        or args.search_timeout <= 0.0
    ):
        parser.error("--search-timeout must be finite and positive")
    return args


def _prepare_running(
    client,
    args,
    blueprint_id,
    stratum,
    start_seed,
):
    scenario_id = None
    session = None
    try:
        scenario_id = client.prepare_fire_scenario(
            args.anchor,
            seed=args.seed,
            firetruck_count=args.firetrucks,
            pedestrian_count=args.pedestrians,
            blueprint_id=blueprint_id,
        )
        ready = client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
        )
        session = LockstepSession(client)
        session.__enter__()
        client.start_scenario(scenario_id)
        session.advance()
        calibration = session.capture_rgbd_pair()
        spec = ObservationSpec.from_pair(calibration)
        scenario = client.get_scenario_state(scenario_id)
        generated = generate_task_start(
            client,
            session,
            scenario,
            spec,
            stratum,
            start_seed,
            max_candidates=args.max_start_candidates,
            horizon_steps=args.horizon_steps,
        )
        return scenario_id, ready.blueprint_id, session, generated
    except Exception:
        if scenario_id is not None:
            client.reset_scenario(scenario_id)
        if session is not None:
            session.close()
        raise


def _witness_digest(report):
    digest = hashlib.blake2b(digest_size=16)
    digest.update(report.search_digest.encode("ascii"))
    digest.update(int(report.status).to_bytes(4, "little"))
    if report.witness is not None:
        for action in report.witness.actions:
            digest.update(repr(action).encode("utf-8"))
    return digest.hexdigest()


def _validate_witness_structure(report, horizon_steps):
    witness = report.witness
    if witness is None:
        raise RuntimeError(
            f"{report.visibility_stratum.name} has no joint witness: "
            f"{report.message}; "
            f"cue_path_found={report.cue_path_found}, "
            f"goal_view_path_found={report.goal_view_path_found}, "
            f"minimum_ordered_actions={report.minimum_ordered_actions}"
        )
    if report.status != FeasibilityStatus.JOINT_WITNESS_WITH_MARGIN:
        raise RuntimeError(
            f"{report.visibility_stratum.name} witness lacks the required "
            f"{REQUIRED_ACTION_MARGIN}-action margin: {report.message}"
        )
    if witness.stop_actions != 1:
        raise RuntimeError("Witness must contain exactly one STOP")
    if not isinstance(witness.actions[-1], StopAction):
        raise RuntimeError("Witness does not terminate with STOP")
    if (
        witness.forward_actions
        + witness.ascend_actions
        + witness.descend_actions
        + witness.turn_left_actions
        + witness.turn_right_actions
        + witness.hold_actions
        + witness.stop_actions
        != witness.total_actions
    ):
        raise RuntimeError(
            "Witness action-class counts do not sum to total_actions"
        )
    if (
        witness.total_actions > horizon_steps
        or witness.remaining_actions < REQUIRED_ACTION_MARGIN
    ):
        raise RuntimeError("Witness violates horizon or margin")
    allowed = (
        ForwardAction,
        AscendAction,
        DescendAction,
        TurnLeftAction,
        TurnRightAction,
        HoldAction,
        StopAction,
    )
    for action in witness.actions:
        if not isinstance(action, allowed):
            raise RuntimeError(
                f"Witness contains unknown action {action!r}"
            )


def _show_witness(auditor, report):
    import matplotlib.pyplot as plt

    witness = report.witness
    figure, axis = plt.subplots(figsize=(9, 8))
    searched = np.asarray(auditor.search_positions)
    if searched.size:
        axis.scatter(
            searched[:, 0],
            searched[:, 1],
            s=3,
            alpha=0.2,
            color="gray",
            label="searched fixed-action states",
        )
    start = np.asarray(
        auditor.generated_start.blueprint.absolute_pose[:3]
    )
    yaw = float(
        auditor.generated_start.blueprint.absolute_pose[5]
    )
    trajectory = [start.copy()]
    position = start.copy()
    for action in witness.actions:
        if isinstance(action, ForwardAction):
            yaw_radians = math.radians(yaw)
            forward = np.asarray(
                (-math.sin(yaw_radians), math.cos(yaw_radians), 0.0)
            )
            position = position + forward * TASK_FORWARD_STEP_METERS
            trajectory.append(position.copy())
        elif isinstance(action, AscendAction):
            position[2] += TASK_VERTICAL_STEP_METERS
            trajectory.append(position.copy())
        elif isinstance(action, DescendAction):
            position[2] -= TASK_VERTICAL_STEP_METERS
            trajectory.append(position.copy())
        elif isinstance(action, TurnLeftAction):
            yaw += TASK_YAW_STEP_DEGREES
        elif isinstance(action, TurnRightAction):
            yaw -= TASK_YAW_STEP_DEGREES
        elif isinstance(action, HoldAction):
            trajectory.append(position.copy())
    trajectory = np.asarray(trajectory)
    cue = np.asarray(witness.cue.first_pose[:3])
    goal = np.asarray(witness.goal.pose[:3])
    axis.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        linestyle="-",
        color="black",
        linewidth=1.2,
        alpha=0.75,
        label="executed translation path",
    )
    axis.scatter(
        start[0],
        start[1],
        facecolors="none",
        edgecolors="blue",
        linewidths=2.5,
        s=150,
        zorder=6,
        label="start",
    )
    axis.scatter(
        cue[0],
        cue[1],
        color="orange",
        s=70,
        zorder=5,
        label="cue",
    )
    axis.scatter(
        goal[0],
        goal[1],
        color="red",
        s=70,
        zorder=5,
        label="goal",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(
        f"{report.visibility_stratum.name} "
        f"actions={witness.total_actions} "
        f"slack={witness.remaining_actions}"
    )
    axis.set_xlabel("GTA world X")
    axis.set_ylabel("GTA world Y")
    axis.legend()
    plt.show()


def _observable_roles(client, session, scenario_id, pair):
    pose = client.get_pose()
    visibility = client.query_visibility(
        scenario_id,
        session.session_id,
        pose[:3],
        timeout=30.0,
    )
    assessment = assess_visibility(
        visibility,
        pair_view_matrices(pair),
        ObservationSpec.from_pair(pair),
    )
    roles = {}
    for target in assessment.targets:
        if (
            target.oblique.task_observable
            or target.nadir.task_observable
        ):
            roles[target.stable_id] = target.role
    return roles


def _record_current_observation(
    recorder,
    client,
    session,
    scenario_id,
    executor,
    action,
):
    pair = executor.current_pair
    pose = client.get_pose()
    scenario = client.get_scenario_state(scenario_id)
    visibility = client.query_visibility(
        scenario_id,
        session.session_id,
        pose[:3],
        timeout=30.0,
    )
    if (
        visibility.lockstep_session_id != pair.clock.session_id
        or visibility.step_index != pair.clock.step_index
        or visibility.game_timer_ms != pair.clock.game_timer_ms
    ):
        raise RuntimeError(
            "Trajectory annotation does not belong to the RGB instant"
        )
    assessment = assess_visibility(
        visibility,
        pair_view_matrices(pair),
        ObservationSpec.from_pair(pair),
    )
    recorder.record(
        executor.action_count,
        action,
        pair,
        pose,
        executor.odometry,
        scenario,
        assessment,
    )


def _direction_cosine(previous, current, role, event_position):
    displacement = (
        np.asarray(current.position[:2], dtype=np.float64)
        - np.asarray(previous.position[:2], dtype=np.float64)
    )
    length = float(np.linalg.norm(displacement))
    if length <= 1.0e-9:
        return -1.0, 0.0
    event = np.asarray(event_position[:2], dtype=np.float64)
    origin = np.asarray(previous.position[:2], dtype=np.float64)
    if role == ScenarioEntityRole.FIRE_TRUCK:
        expected = event - origin
    else:
        expected = origin - event
    expected_length = float(np.linalg.norm(expected))
    if expected_length <= 1.0e-9:
        return -1.0, length
    return (
        float(
            np.dot(displacement, expected)
            / (length * expected_length)
        ),
        length,
    )


def _validate_replay_cue(
    first_state,
    first_roles,
    second_state,
    second_roles,
    expected_role,
):
    target_role = (
        VisibilityTargetRole.FIRE_TRUCK
        if expected_role == ScenarioEntityRole.FIRE_TRUCK
        else VisibilityTargetRole.FLEEING_PEDESTRIAN
    )
    first_entities = {
        entity.stable_id: entity
        for entity in first_state.entities
        if entity.exists and entity.role == expected_role
    }
    second_entities = {
        entity.stable_id: entity
        for entity in second_state.entities
        if entity.exists and entity.role == expected_role
    }
    first_visible = {
        key
        for key, role in first_roles.items()
        if role == target_role
    }
    second_visible = {
        key
        for key, role in second_roles.items()
        if role == target_role
    }
    common_ids = sorted(
        set(first_entities)
        & set(second_entities)
        & first_visible
        & second_visible
    )
    diagnostics = []
    for stable_id in common_ids:
        first = first_entities[stable_id]
        second = second_entities[stable_id]
        cosine, displacement = _direction_cosine(
            first,
            second,
            expected_role,
            second_state.event_position,
        )
        if (
            first.task_state == ScenarioTaskState.ACTIVE
            and second.task_state == ScenarioTaskState.ACTIVE
            and displacement
            >= TASK_MIN_CUE_HORIZONTAL_DISPLACEMENT_METERS
            and cosine >= 0.5
        ):
            return stable_id, displacement, cosine
        diagnostics.append(
            f"id={stable_id}:"
            f"{first.task_state.name}->{second.task_state.name},"
            f"disp={displacement:.3f}m,"
            f"cos={cosine:.3f}"
        )
    raise RuntimeError(
        "Replay did not observe one same-role responder as a valid "
        "consecutive dynamic cue; "
        f"first_visible={sorted(first_visible)}, "
        f"second_visible={sorted(second_visible)}, "
        f"common_diagnostics={diagnostics}"
    )


def _validate_replay_cue_pose(actual_pose, expected_pose, label):
    position_error = math.dist(
        actual_pose[:3],
        expected_pose[:3],
    )
    yaw_error = abs(
        (
            float(actual_pose[5])
            - float(expected_pose[3])
            + 180.0
        )
        % 360.0
        - 180.0
    )
    if position_error > 1.0e-3 or yaw_error > 1.0e-3:
        raise RuntimeError(
            f"Replay {label} camera missed witness pose: "
            f"position_error={position_error:.6f}m, "
            f"yaw_error={yaw_error:.6f}deg"
        )


def _replay(
    client,
    args,
    blueprint_id,
    generated,
    report,
    recording_root=None,
):
    replay_started = time.perf_counter()
    witness = report.witness
    scenario_id = None
    session = None
    recorder = None
    replay_error = None
    cue_reproduced = None
    try:
        scenario_id = client.prepare_fire_scenario(
            args.anchor,
            seed=args.seed,
            firetruck_count=args.firetrucks,
            pedestrian_count=args.pedestrians,
            blueprint_id=blueprint_id,
        )
        client.wait_scenario_ready(
            scenario_id,
            timeout=args.prepare_timeout,
        )
        session = LockstepSession(client)
        session.__enter__()
        client.start_scenario(scenario_id)
        session.advance()
        client.set_camera_pose(
            *generated.blueprint.absolute_pose[:3],
            generated.blueprint.absolute_pose[5],
            collision_check=False,
        )
        client.set_camera_pitch(OBLIQUE_PITCH_DEGREES)
        initial_pair = session.capture_rgbd_pair()
        executor = ResearchActionExecutor(
            client,
            session,
            initial_pair,
            generated.blueprint,
            collision_check=True,
        )
        if recording_root is not None:
            recorder = TrajectoryRecorder(
                Path(recording_root)
                / report.visibility_stratum.name,
                report,
                generated,
                jpeg_quality=args.record_jpeg_quality,
            )
        invalid_pose = client.get_pose()
        invalid_clock = session.refresh()
        try:
            executor.execute(object())
        except InvalidTaskAction:
            pass
        else:
            raise RuntimeError(
                "Research executor accepted an action outside the "
                "seven fixed task actions"
            )
        if (
            executor.action_count != 0
            or any(
                abs(left - right) > 1.0e-3
                for left, right in zip(
                    client.get_pose(),
                    invalid_pose,
                )
            )
            or session.refresh().step_index != invalid_clock.step_index
        ):
            raise RuntimeError(
                "Rejected invalid action changed pose, clock, or "
                "action count"
            )
        cue_observations = {}
        if witness.cue.first_step == 0:
            cue_observations[0] = (
                client.get_scenario_state(scenario_id),
                _observable_roles(
                    client,
                    session,
                    scenario_id,
                    initial_pair,
                ),
                client.get_pose(),
            )

        for action in witness.actions:
            if recorder is not None:
                _record_current_observation(
                    recorder,
                    client,
                    session,
                    scenario_id,
                    executor,
                    action,
                )
            if isinstance(action, StopAction):
                pair = executor.current_pair
                state = client.get_scenario_state(scenario_id)
                roles = _observable_roles(
                    client,
                    session,
                    scenario_id,
                    pair,
                )
                source = next(
                    entity
                    for entity in state.entities
                    if entity.role
                    == ScenarioEntityRole.FIRE_SOURCE_VEHICLE
                )
                if (
                    roles.get(source.stable_id)
                    != VisibilityTargetRole.FIRE_SOURCE_VEHICLE
                ):
                    raise RuntimeError(
                        "Replay STOP observation is not task-observable"
                    )
                estimated_world = (
                    generated.blueprint.local_to_world(
                        action.event_estimate_local
                    )
                )
                error = math.dist(
                    estimated_world,
                    state.event_position,
                )
                if error > 5.0:
                    raise RuntimeError(
                        f"Replay STOP estimate error is {error:.3f} m"
                    )
                if not state.event_active:
                    raise RuntimeError(
                        "Replay fire is inactive at STOP"
                    )
                executor.execute(action)
                continue

            result = executor.execute(action)
            task_step = result.action_index
            if task_step in (
                witness.cue.first_step,
                witness.cue.second_step,
            ):
                cue_observations[task_step] = (
                    client.get_scenario_state(scenario_id),
                    _observable_roles(
                        client,
                        session,
                        scenario_id,
                        executor.current_pair,
                    ),
                    client.get_pose(),
                )

        first = cue_observations.get(witness.cue.first_step)
        second = cue_observations.get(witness.cue.second_step)
        if first is None or second is None:
            raise RuntimeError(
                "Replay did not capture both declared cue observations"
            )
        _validate_replay_cue_pose(
            first[2],
            witness.cue.first_pose,
            "first cue",
        )
        _validate_replay_cue_pose(
            second[2],
            witness.cue.second_pose,
            "second cue",
        )
        cue_reproduced = False
        cue_diagnostic = ""
        try:
            matched_id, displacement, cosine = _validate_replay_cue(
                first[0],
                first[1],
                second[0],
                second[1],
                witness.cue.role,
            )
            cue_reproduced = True
            cue_diagnostic = (
                f"matched_entity={matched_id} "
                f"cue_displacement={displacement:.3f}m "
                f"cue_cosine={cosine:.3f}"
            )
        except RuntimeError as error:
            cue_diagnostic = str(error)
        if not executor.stopped:
            raise RuntimeError("Replay action sequence did not STOP")
        if executor.action_count != witness.total_actions:
            raise RuntimeError(
                "Replay action count does not match witness"
            )
        print(
            f"{report.visibility_stratum.name} structural replay PASS "
            f"cue_reproduced={str(cue_reproduced).lower()} "
            f"actions={executor.action_count} "
            f"time={time.perf_counter() - replay_started:.1f}s "
            f"cue_diagnostic={cue_diagnostic}"
        )
        if recorder is not None:
            recorder.finish(
                "PASS",
                cue_reproduced=cue_reproduced,
            )
            print(
                f"{report.visibility_stratum.name} recording "
                f"path={recorder.root} "
                f"frames={len(recorder.frames)} "
                f"size={recorder.size_bytes() / (1024 * 1024):.1f}MiB"
            )
        client.reset_scenario(scenario_id)
        scenario_id = None
        session.close()
        session = None
    except Exception as error:
        replay_error = error
        raise
    finally:
        if recorder is not None and replay_error is not None:
            recorder.finish(
                "FAILED",
                cue_reproduced=cue_reproduced,
                error=replay_error,
            )
        if scenario_id is not None:
            client.reset_scenario(scenario_id)
        if session is not None:
            session.close()


def _search_once(
    client,
    args,
    blueprint_id,
    stratum,
    start_seed,
):
    search_started = time.perf_counter()
    scenario_id = None
    session = None
    try:
        print(
            f"{stratum.name} setup start "
            f"search_timeout={args.search_timeout:.1f}s"
        )
        setup_started = time.perf_counter()
        (
            scenario_id,
            resolved_blueprint,
            session,
            generated,
        ) = _prepare_running(
            client,
            args,
            blueprint_id,
            stratum,
            start_seed,
        )
        setup_seconds = time.perf_counter() - setup_started
        print(
            f"{stratum.name} setup PASS "
            f"time={setup_seconds:.1f}s; fixed-action search start"
        )
        auditor = SpatiotemporalFeasibilityAuditor(
            client,
            session,
            scenario_id,
            generated,
            search_timeout_seconds=args.search_timeout,
            progress_callback=print,
        )
        report = auditor.run()
        digest = _witness_digest(report)
        witness = report.witness
        timing = dict(report.phase_seconds)
        timing_text = (
            f"setup={setup_seconds:.1f}s "
            f"action_search={timing['action_search']:.1f}s "
            f"goal={timing['goal_visibility']:.1f}s "
            f"cue_search={timing['temporal_cue_search']:.1f}s "
            f"total={time.perf_counter() - search_started:.1f}s"
        )
        if (
            report.status
            != FeasibilityStatus.JOINT_WITNESS_WITH_MARGIN
        ):
            cue_diagnostics = ",".join(
                f"{name}={value}"
                for name, value in report.cue_diagnostics
            )
            candidate_text = "none"
            if witness is not None:
                candidate_text = (
                    f"F{witness.forward_actions}/"
                    f"U{witness.ascend_actions}/"
                    f"D{witness.descend_actions}/"
                    f"L{witness.turn_left_actions}/"
                    f"R{witness.turn_right_actions}/"
                    f"H{witness.hold_actions}/"
                    f"S{witness.stop_actions} "
                    f"total={witness.total_actions} "
                    f"slack={witness.remaining_actions}"
                )
            print(
                f"{stratum.name} search FAIL "
                f"start={report.start_id} "
                f"status={report.status.name} "
                f"search={report.searched_states}states/"
                f"{report.checked_motion_edges}edges "
                f"steps={report.queried_steps} "
                f"cue_path={report.cue_path_found} "
                f"goal_path={report.goal_view_path_found} "
                f"joint_path={report.cue_then_goal_path_found} "
                f"minimum_ordered_actions="
                f"{report.minimum_ordered_actions} "
                f"candidate_actions[{candidate_text}] "
                f"cue_diagnostics[{cue_diagnostics}] "
                f"time[{timing_text}] "
                f"message={report.message}"
            )
            if args.show_witness and witness is not None:
                _show_witness(auditor, report)
        else:
            _validate_witness_structure(
                report,
                generated.blueprint.action_spec.horizon_steps,
            )
            print(
                f"{stratum.name} search PASS "
                f"start={report.start_id} "
                f"search={report.searched_states}states/"
                f"{report.checked_motion_edges}edges "
                f"steps={report.queried_steps} "
                f"actions=F{witness.forward_actions}/"
                f"U{witness.ascend_actions}/"
                f"D{witness.descend_actions}/"
                f"L{witness.turn_left_actions}/"
                f"R{witness.turn_right_actions}/"
                f"H{witness.hold_actions}/"
                f"S{witness.stop_actions} "
                f"total={witness.total_actions} "
                f"slack={witness.remaining_actions} "
                f"cue={witness.cue.role.name}@"
                f"{witness.cue.first_step}->"
                f"{witness.cue.second_step} "
                f"transition="
                f"{type(witness.cue.transition_action).__name__} "
                f"time[{timing_text}] "
                f"digest={digest}"
            )
            if args.show_witness:
                _show_witness(auditor, report)
        client.reset_scenario(scenario_id)
        scenario_id = None
        session.close()
        session = None
        return resolved_blueprint, generated, report, digest
    finally:
        if scenario_id is not None:
            client.reset_scenario(scenario_id)
        if session is not None:
            session.close()


def _register_recorded_stratum(
    recording_manifest,
    recording_root,
    stratum,
):
    if recording_manifest is None:
        return
    relative = Path(stratum.name) / "trajectory.json"
    if not (recording_root / relative).is_file():
        return
    if not any(
        item["name"] == stratum.name
        for item in recording_manifest["strata"]
    ):
        recording_manifest["strata"].append(
            {
                "name": stratum.name,
                "path": relative.as_posix(),
            }
        )
    write_json(
        recording_root / "run.json",
        recording_manifest,
    )


def main():
    validation_started = time.perf_counter()
    args = _parse_args()
    recording_root = None
    recording_manifest = None
    validation_error = None
    if args.record_dir is not None:
        TrajectoryRecorder.require_dependencies()
        recording_root = args.record_dir.resolve()
        if recording_root.exists():
            raise FileExistsError(
                f"--record-dir already exists: {recording_root}"
            )
        recording_root.mkdir(parents=True)
        recording_manifest = {
            "schema_version": 1,
            "status": "RUNNING",
            "error": None,
            "config": {
                "anchor": list(args.anchor),
                "seed": args.seed,
                "start_seed": args.start_seed,
                "horizon_steps": args.horizon_steps,
                "firetrucks": args.firetrucks,
                "pedestrians": args.pedestrians,
                "jpeg_quality": args.record_jpeg_quality,
            },
            "strata": [],
        }
        write_json(
            recording_root / "run.json",
            recording_manifest,
        )
    client = DroneSimClient(args.host, args.port)
    client.require_camera_active()
    original_pose = client.get_pose()
    client.set_time(12, 0, 0)
    client.set_weather("EXTRASUNNY")
    client.teleport_player(*args.anchor)
    blueprint_id = 0
    reports = []
    failures = []
    try:
        for index, stratum in enumerate(
            (
                StartVisibilityStratum.CUE_VISIBLE,
                StartVisibilityStratum.CUE_HIDDEN,
            )
        ):
            start_seed = args.start_seed + index
            (
                blueprint_id,
                generated,
                report,
                digest,
            ) = _search_once(
                client,
                args,
                blueprint_id,
                stratum,
                start_seed,
            )
            if (
                report.status
                != FeasibilityStatus.JOINT_WITNESS_WITH_MARGIN
            ):
                failures.append(
                    f"{stratum.name} search: {report.message}"
                )
                continue
            if args.verify_determinism:
                (
                    repeated_blueprint,
                    _repeated_generated,
                    repeated_report,
                    repeated_digest,
                ) = _search_once(
                    client,
                    args,
                    blueprint_id,
                    stratum,
                    start_seed,
                )
                if (
                    repeated_blueprint != blueprint_id
                    or repeated_report.start_id != report.start_id
                    or repeated_report.search_digest
                    != report.search_digest
                ):
                    raise RuntimeError(
                        f"{stratum.name} task start or fixed action "
                        "lattice is not deterministic"
                    )
                if repeated_digest != digest:
                    print(
                        f"{stratum.name} deterministic infrastructure "
                        "PASS; witness changed with GTA AI trajectory "
                        f"{digest}->{repeated_digest}"
                    )
            try:
                _replay(
                    client,
                    args,
                    blueprint_id,
                    generated,
                    report,
                    recording_root=recording_root,
                )
            except Exception as error:
                _register_recorded_stratum(
                    recording_manifest,
                    recording_root,
                    stratum,
                )
                print(f"{stratum.name} replay FAIL {error}")
                failures.append(
                    f"{stratum.name} replay: {error}"
                )
                continue
            reports.append(report)
            _register_recorded_stratum(
                recording_manifest,
                recording_root,
                stratum,
            )
        if failures:
            raise RuntimeError(
                "Stage 2D validation failed:\n- "
                + "\n- ".join(failures)
            )
        print(
            "PASS strata=2 "
            f"blueprint={blueprint_id} "
            f"actions={','.join(str(r.witness.total_actions) for r in reports)}"
        )
        print(
            "No RGB-D, image, point-cloud, or trajectory payload "
            "was written to disk."
            if recording_root is None
            else (
                "Optional compressed trajectory recording was "
                f"written to {recording_root}; no Depth payload "
                "was written."
            )
        )
    except BaseException as error:
        validation_error = error
        raise
    finally:
        try:
            client.set_camera_pose(
                *original_pose[:3],
                original_pose[5],
                collision_check=False,
            )
            client.set_camera_pitch(original_pose[3])
        finally:
            client.restore_player()
            if recording_manifest is not None:
                recording_manifest["status"] = (
                    "PASS"
                    if validation_error is None
                    else "FAILED"
                )
                recording_manifest["error"] = (
                    None
                    if validation_error is None
                    else str(validation_error)
                )
                write_json(
                    recording_root / "run.json",
                    recording_manifest,
                )
            print(
                f"validation wall time="
                f"{time.perf_counter() - validation_started:.1f}s"
            )


if __name__ == "__main__":
    main()
