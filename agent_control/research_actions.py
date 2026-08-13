"""Strict task actions for the hidden-event localization environment.

The low-level camera protocol accepts an absolute position and yaw together.
This module deliberately exposes only seven fixed research actions:
forward, ascend, descend, turn left, turn right, hold, and stop.
"""

import math
import time
from dataclasses import dataclass

from .dronesim_client import (
    LockstepRgbdPair,
    LockstepSession,
)
from .task_starts import (
    TASK_FORWARD_STEP_METERS,
    TASK_VERTICAL_STEP_METERS,
    TASK_YAW_STEP_DEGREES,
    TaskRelativePoseController,
    TaskStartBlueprint,
    make_agent_observation,
)


_POSE_POSITION_TOLERANCE_METERS = 1.0e-3
_POSE_ANGLE_TOLERANCE_DEGREES = 1.0e-3


class InvalidTaskAction(ValueError):
    status_name = "INVALID_TASK_ACTION"

    def __init__(self, message):
        super().__init__(f"{self.status_name}: {message}")


def _finite_values(*values):
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise InvalidTaskAction("Action contains a non-finite value")
    return converted


def _angle_error_degrees(left, right):
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class ForwardAction:
    pass


@dataclass(frozen=True)
class AscendAction:
    pass


@dataclass(frozen=True)
class DescendAction:
    pass


@dataclass(frozen=True)
class TurnLeftAction:
    pass


@dataclass(frozen=True)
class TurnRightAction:
    pass


@dataclass(frozen=True)
class HoldAction:
    pass


@dataclass(frozen=True)
class StopAction:
    event_estimate_local: tuple

    def __post_init__(self):
        try:
            values = tuple(self.event_estimate_local)
        except TypeError as error:
            raise InvalidTaskAction(
                "STOP estimate must contain three values"
            ) from error
        if len(values) != 3:
            raise InvalidTaskAction(
                "STOP estimate must contain three values"
            )
        values = _finite_values(*values)
        object.__setattr__(self, "event_estimate_local", values)


TaskAction = (
    ForwardAction
    | AscendAction
    | DescendAction
    | TurnLeftAction
    | TurnRightAction
    | HoldAction
    | StopAction
)


@dataclass(frozen=True)
class TaskStepResult:
    action_index: int
    action: TaskAction
    odometry: object
    clock: object
    agent_observation: object | None
    stopped: bool
    timing: object


@dataclass(frozen=True)
class ActionExecutionTiming:
    pose_seconds: float
    advance_seconds: float
    capture_seconds: float
    total_seconds: float


class ResearchActionExecutor:
    """Execute fixed research actions in one lockstep session."""

    def __init__(
        self,
        client,
        lockstep,
        initial_pair,
        blueprint,
        collision_check=True,
    ):
        if not isinstance(lockstep, LockstepSession):
            raise TypeError("lockstep must be an active LockstepSession")
        if not isinstance(initial_pair, LockstepRgbdPair):
            raise TypeError("initial_pair must be a LockstepRgbdPair")
        if initial_pair.clock.session_id != lockstep.session_id:
            raise ValueError(
                "initial_pair does not belong to the lockstep session"
            )
        if not isinstance(blueprint, TaskStartBlueprint):
            raise TypeError("blueprint must be a TaskStartBlueprint")
        self.client = client
        self.lockstep = lockstep
        if not collision_check:
            raise ValueError(
                "Research actions require collision_check=True"
            )
        self.controller = TaskRelativePoseController(client, blueprint)
        self.controller.synchronize()
        self._horizon_steps = int(
            blueprint.action_spec.horizon_steps
        )
        self._pair = initial_pair
        self._action_count = 0
        self._stopped = False
        self._failed = False

    @property
    def action_count(self):
        return self._action_count

    @property
    def odometry(self):
        return self.controller.odometry

    @property
    def current_pair(self):
        """Return the raw pair for evaluation code, never agent input."""
        return self._pair

    @property
    def stopped(self):
        return self._stopped

    def _require_active(self):
        if self._failed:
            raise RuntimeError(
                "ResearchActionExecutor is invalid after a partial "
                "execution failure"
            )
        if self._stopped:
            raise RuntimeError("The task episode has already stopped")
        if self._action_count >= self._horizon_steps:
            raise RuntimeError(
                f"The {self._horizon_steps}-action task horizon "
                "is exhausted"
            )

    def _advance_and_capture(
        self,
        action,
        timeout_ms,
        execution_started,
        pose_seconds,
    ):
        try:
            phase_started = time.perf_counter()
            clock = self.lockstep.advance()
            advance_seconds = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            pair = self.lockstep.capture_rgbd_pair(timeout_ms)
            capture_seconds = time.perf_counter() - phase_started
        except Exception:
            self._failed = True
            raise
        if pair.clock.step_index != clock.step_index:
            self._failed = True
            raise RuntimeError(
                "RGB-D observation does not belong to the completed action"
            )
        self._action_count += 1
        self._pair = pair
        return TaskStepResult(
            action_index=self._action_count,
            action=action,
            odometry=self.controller.odometry,
            clock=pair.clock,
            agent_observation=make_agent_observation(
                pair,
                self.controller.odometry,
            ),
            stopped=False,
            timing=ActionExecutionTiming(
                pose_seconds=pose_seconds,
                advance_seconds=advance_seconds,
                capture_seconds=capture_seconds,
                total_seconds=time.perf_counter() - execution_started,
            ),
        )

    def execute(self, action, timeout_ms=5000):
        execution_started = time.perf_counter()
        self._require_active()
        if not isinstance(
            action,
            (
                ForwardAction,
                AscendAction,
                DescendAction,
                TurnLeftAction,
                TurnRightAction,
                HoldAction,
                StopAction,
            ),
        ):
            raise InvalidTaskAction(
                "Expected one fixed research action: ForwardAction, "
                "AscendAction, DescendAction, TurnLeftAction, "
                "TurnRightAction, HoldAction, or StopAction"
            )

        before_pose = self.client.get_pose()
        self.controller.synchronize()
        before_odometry = self.controller.odometry
        if isinstance(action, StopAction):
            self._action_count += 1
            self._stopped = True
            return TaskStepResult(
                action_index=self._action_count,
                action=action,
                odometry=self.controller.odometry,
                clock=self.lockstep.snapshot,
                agent_observation=None,
                stopped=True,
                timing=ActionExecutionTiming(
                    pose_seconds=time.perf_counter() - execution_started,
                    advance_seconds=0.0,
                    capture_seconds=0.0,
                    total_seconds=time.perf_counter() - execution_started,
                ),
            )
        if self._action_count >= self._horizon_steps - 1:
            raise InvalidTaskAction(
                "A non-terminal action would leave no action for STOP"
            )

        try:
            if isinstance(action, ForwardAction):
                after_odometry = self.controller.step_relative(
                    TASK_FORWARD_STEP_METERS,
                    0.0,
                    0.0,
                    0.0,
                )
                actual = self.client.get_pose()
                if _angle_error_degrees(
                    actual[5],
                    before_pose[5],
                ) > _POSE_ANGLE_TOLERANCE_DEGREES:
                    self._failed = True
                    raise RuntimeError(
                        "FORWARD changed camera yaw"
                    )
                yaw_radians = math.radians(
                    before_odometry.yaw_from_start_degrees
                )
                expected_delta = (
                    math.cos(yaw_radians)
                    * TASK_FORWARD_STEP_METERS,
                    -math.sin(yaw_radians)
                    * TASK_FORWARD_STEP_METERS,
                    0.0,
                )
                actual_delta = tuple(
                    float(after) - float(before)
                    for after, before in zip(
                        after_odometry.position_local,
                        before_odometry.position_local,
                    )
                )
                if math.dist(actual_delta, expected_delta) > (
                    _POSE_POSITION_TOLERANCE_METERS
                ):
                    self._failed = True
                    raise RuntimeError(
                        "FORWARD actual start-local displacement does "
                        "not match GTA yaw convention; "
                        f"actual={actual_delta}, expected={expected_delta}"
                    )
            elif isinstance(action, AscendAction):
                self.controller.step_relative(
                    0.0,
                    0.0,
                    TASK_VERTICAL_STEP_METERS,
                    0.0,
                )
                actual = self.client.get_pose()
                if _angle_error_degrees(
                    actual[5],
                    before_pose[5],
                ) > _POSE_ANGLE_TOLERANCE_DEGREES:
                    self._failed = True
                    raise RuntimeError(
                        "ASCEND changed camera yaw"
                    )
            elif isinstance(action, DescendAction):
                self.controller.step_relative(
                    0.0,
                    0.0,
                    -TASK_VERTICAL_STEP_METERS,
                    0.0,
                )
                actual = self.client.get_pose()
                if _angle_error_degrees(
                    actual[5],
                    before_pose[5],
                ) > _POSE_ANGLE_TOLERANCE_DEGREES:
                    self._failed = True
                    raise RuntimeError(
                        "DESCEND changed camera yaw"
                    )
            elif isinstance(action, (TurnLeftAction, TurnRightAction)):
                yaw_delta = (
                    TASK_YAW_STEP_DEGREES
                    if isinstance(action, TurnLeftAction)
                    else -TASK_YAW_STEP_DEGREES
                )
                self.controller.step_relative(
                    0.0,
                    0.0,
                    0.0,
                    yaw_delta,
                )
                actual = self.client.get_pose()
                position_error = math.dist(
                    actual[:3],
                    before_pose[:3],
                )
                if position_error > _POSE_POSITION_TOLERANCE_METERS:
                    self._failed = True
                    raise RuntimeError(
                        "TURN changed camera position"
                    )
            else:
                actual = self.client.get_pose()
                if (
                    math.dist(actual[:3], before_pose[:3])
                    > _POSE_POSITION_TOLERANCE_METERS
                    or _angle_error_degrees(
                        actual[5],
                        before_pose[5],
                    )
                    > _POSE_ANGLE_TOLERANCE_DEGREES
                ):
                    self._failed = True
                    raise RuntimeError(
                        "HOLD began from an unstable camera pose"
                    )
                self.controller.synchronize()
        except Exception:
            actual = self.client.get_pose()
            if (
                math.dist(actual[:3], before_pose[:3])
                > _POSE_POSITION_TOLERANCE_METERS
                or _angle_error_degrees(
                    actual[5],
                    before_pose[5],
                )
                > _POSE_ANGLE_TOLERANCE_DEGREES
            ):
                self._failed = True
            else:
                self.controller.synchronize()
            raise

        pose_seconds = time.perf_counter() - execution_started
        result = self._advance_and_capture(
            action,
            timeout_ms,
            execution_started,
            pose_seconds,
        )
        if isinstance(action, HoldAction):
            after_pose = self.client.get_pose()
            if (
                math.dist(after_pose[:3], before_pose[:3])
                > _POSE_POSITION_TOLERANCE_METERS
                or _angle_error_degrees(
                    after_pose[5],
                    before_pose[5],
                )
                > _POSE_ANGLE_TOLERANCE_DEGREES
            ):
                self._failed = True
                raise RuntimeError(
                    "HOLD changed camera pose while simulation advanced"
                )
            self.controller.synchronize()
            result = TaskStepResult(
                action_index=result.action_index,
                action=result.action,
                odometry=self.controller.odometry,
                clock=result.clock,
                agent_observation=result.agent_observation,
                stopped=False,
                timing=ActionExecutionTiming(
                    pose_seconds=result.timing.pose_seconds,
                    advance_seconds=result.timing.advance_seconds,
                    capture_seconds=result.timing.capture_seconds,
                    total_seconds=time.perf_counter() - execution_started,
                ),
            )
        return result
