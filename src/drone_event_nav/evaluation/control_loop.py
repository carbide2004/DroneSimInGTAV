from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Tuple

from drone_event_nav.actions import dispatch_action, parse_action
from drone_event_nav.config import MovementConfig


Pose = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ControlLoopConfig:
    max_steps: int
    max_consecutive_failures: int = 3
    movement: MovementConfig = MovementConfig()

    def __post_init__(self) -> None:
        if int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive")
        if int(self.max_consecutive_failures) <= 0:
            raise ValueError("max_consecutive_failures must be positive")


@dataclass(frozen=True)
class ControlLoopResult:
    status: str
    steps: int
    final_pose: Optional[Pose]
    last_known_pose: Optional[Pose] = None
    failure_count: int = 0
    stopped_by_policy: bool = False
    error: Optional[str] = None
    cleanup_error: Optional[str] = None


def _execute_control_loop(
    client: Any,
    model: Any,
    config: ControlLoopConfig,
    make_observation: Callable[[Pose, Any], Any],
) -> ControlLoopResult:
    steps = 0
    pose_failures = 0
    capture_failures = 0
    total_failures = 0
    last_known_pose = None

    try:
        while steps < int(config.max_steps):
            pose = client.get_pose()
            if pose is None:
                pose_failures += 1
                total_failures += 1
                if pose_failures >= int(config.max_consecutive_failures):
                    return ControlLoopResult(
                        status="pose_failure_budget_exhausted",
                        steps=steps,
                        final_pose=None,
                        last_known_pose=last_known_pose,
                        failure_count=total_failures,
                    )
                continue

            pose_failures = 0
            last_known_pose = pose
            capture = client.capture()
            if capture is None:
                capture_failures += 1
                total_failures += 1
                if capture_failures >= int(config.max_consecutive_failures):
                    return ControlLoopResult(
                        status="capture_failure_budget_exhausted",
                        steps=steps,
                        final_pose=pose,
                        last_known_pose=pose,
                        failure_count=total_failures,
                    )
                continue

            capture_failures = 0
            observation = make_observation(pose, capture)
            action = parse_action(model.generate_action(observation))
            if action is None:
                return ControlLoopResult(
                    status="invalid_policy_output",
                    steps=steps,
                    final_pose=pose,
                    last_known_pose=pose,
                    failure_count=total_failures,
                )
            if action == "AUTO_STOP_REACHED":
                return ControlLoopResult(
                    status="stopped_by_policy",
                    steps=steps,
                    final_pose=pose,
                    last_known_pose=pose,
                    failure_count=total_failures,
                    stopped_by_policy=True,
                )

            dispatch_action(client, action, **config.movement.to_dispatch_kwargs())
            steps += 1
            post_action_pose = client.get_pose()
            if post_action_pose is None:
                total_failures += 1
                return ControlLoopResult(
                    status="final_pose_unavailable",
                    steps=steps,
                    final_pose=None,
                    last_known_pose=last_known_pose,
                    failure_count=total_failures,
                )
            last_known_pose = post_action_pose

        return ControlLoopResult(
            status="max_steps_reached",
            steps=steps,
            final_pose=last_known_pose,
            last_known_pose=last_known_pose,
            failure_count=total_failures,
        )
    except Exception as exc:
        return ControlLoopResult(
            status="exception",
            steps=steps,
            final_pose=None,
            last_known_pose=last_known_pose,
            failure_count=total_failures,
            error=str(exc),
        )


def run_control_loop(
    client: Any,
    model: Any,
    config: ControlLoopConfig,
    make_observation: Callable[[Pose, Any], Any],
    cleanup: Optional[Callable[[], None]] = None,
) -> ControlLoopResult:
    result = _execute_control_loop(client, model, config, make_observation)
    if cleanup is not None:
        try:
            cleanup()
        except Exception as exc:
            result = replace(result, cleanup_error=str(exc))
    return result
