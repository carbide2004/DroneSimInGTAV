"""Strict task actions for the hidden-event localization environment.

The low-level camera protocol accepts an absolute position and yaw together.
This module deliberately exposes the narrower research action semantics:
translation, yaw rotation, hold, and stop are mutually exclusive.
"""

import math
from dataclasses import dataclass

from .dronesim_client import (
    LockstepRgbdPair,
    LockstepSession,
)
from .task_starts import (
    TASK_HORIZON_STEPS,
    TASK_MAX_TRANSLATION_METERS,
    TASK_MAX_YAW_DEGREES,
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
class TranslateAction:
    dx_body: float
    dy_body: float
    dz_world: float

    def __post_init__(self):
        values = _finite_values(
            self.dx_body,
            self.dy_body,
            self.dz_world,
        )
        distance = math.sqrt(sum(value * value for value in values))
        if distance <= 0.0:
            raise InvalidTaskAction(
                "Zero translation must be represented by HoldAction"
            )
        if distance > TASK_MAX_TRANSLATION_METERS + 1.0e-6:
            raise InvalidTaskAction(
                "Translation norm exceeds "
                f"{TASK_MAX_TRANSLATION_METERS:.1f} m"
            )


@dataclass(frozen=True)
class RotateAction:
    dyaw: float

    def __post_init__(self):
        (dyaw,) = _finite_values(self.dyaw)
        if abs(dyaw) <= 0.0:
            raise InvalidTaskAction(
                "Zero rotation must be represented by HoldAction"
            )
        if abs(dyaw) > TASK_MAX_YAW_DEGREES + 1.0e-6:
            raise InvalidTaskAction(
                "Yaw rotation exceeds "
                f"{TASK_MAX_YAW_DEGREES:.1f} degrees"
            )


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
    TranslateAction
    | RotateAction
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


class ResearchActionExecutor:
    """Execute mutually exclusive research actions in one lockstep session."""

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
        if self._action_count >= TASK_HORIZON_STEPS:
            raise RuntimeError(
                f"The {TASK_HORIZON_STEPS}-action task horizon "
                "is exhausted"
            )

    def _advance_and_capture(self, action, timeout_ms):
        try:
            clock = self.lockstep.advance()
            pair = self.lockstep.capture_rgbd_pair(timeout_ms)
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
        )

    def execute(self, action, timeout_ms=5000):
        self._require_active()
        if not isinstance(
            action,
            (
                TranslateAction,
                RotateAction,
                HoldAction,
                StopAction,
            ),
        ):
            raise InvalidTaskAction(
                "Expected TranslateAction, RotateAction, HoldAction, "
                "or StopAction"
            )

        before_pose = self.client.get_pose()
        self.controller.synchronize()
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
            )
        if self._action_count >= TASK_HORIZON_STEPS - 1:
            raise InvalidTaskAction(
                "A non-terminal action would leave no action for STOP"
            )

        try:
            if isinstance(action, TranslateAction):
                self.controller.step_relative(
                    action.dx_body,
                    action.dy_body,
                    action.dz_world,
                    0.0,
                )
                actual = self.client.get_pose()
                if _angle_error_degrees(
                    actual[5],
                    before_pose[5],
                ) > _POSE_ANGLE_TOLERANCE_DEGREES:
                    self._failed = True
                    raise RuntimeError(
                        "TRANSLATE changed camera yaw"
                    )
            elif isinstance(action, RotateAction):
                self.controller.step_relative(
                    0.0,
                    0.0,
                    0.0,
                    action.dyaw,
                )
                actual = self.client.get_pose()
                position_error = math.dist(
                    actual[:3],
                    before_pose[:3],
                )
                if position_error > _POSE_POSITION_TOLERANCE_METERS:
                    self._failed = True
                    raise RuntimeError(
                        "ROTATE changed camera position"
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

        result = self._advance_and_capture(
            action,
            timeout_ms,
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
            )
        return result

    def execute_components(
        self,
        dx_body,
        dy_body,
        dz_world,
        dyaw,
        timeout_ms=5000,
    ):
        dx_body, dy_body, dz_world, dyaw = _finite_values(
            dx_body,
            dy_body,
            dz_world,
            dyaw,
        )
        translating = math.sqrt(
            dx_body * dx_body
            + dy_body * dy_body
            + dz_world * dz_world
        ) > 0.0
        rotating = abs(dyaw) > 0.0
        if translating and rotating:
            raise InvalidTaskAction(
                "Translation and yaw rotation cannot share one action"
            )
        if translating:
            action = TranslateAction(
                dx_body,
                dy_body,
                dz_world,
            )
        elif rotating:
            action = RotateAction(dyaw)
        else:
            action = HoldAction()
        return self.execute(action, timeout_ms=timeout_ms)
