from dataclasses import asdict, dataclass
import math
from typing import Any, Dict


@dataclass(frozen=True)
class MovementConfig:
    forward_step: float = 5.0
    up_step: float = 5.0
    down_step: float = 5.0
    yaw_step: float = 15.0

    def __post_init__(self) -> None:
        for name, raw_value in asdict(self).items():
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, value)

    @classmethod
    def from_namespace(cls, args: Any) -> "MovementConfig":
        down_step = args.down_step
        up_step = getattr(args, "up_step", None)
        return cls(
            forward_step=args.forward_step,
            up_step=down_step if up_step is None else up_step,
            down_step=down_step,
            yaw_step=args.yaw_step,
        )

    def to_dispatch_kwargs(self) -> Dict[str, float]:
        return asdict(self)
