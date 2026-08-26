"""Strict streaming runtime for the trained Spatial RNN belief updater."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from learning.belief_features import (
    EncodedGroundedTrackStep,
    GroundedTrackFeatureConfig,
    StreamingGroundedTrackEncoder,
    TRACK_FEATURE_NAMES,
)
from learning.spatial_belief_runtime import load_spatial_checkpoint


@dataclass(frozen=True)
class OnlineBeliefSnapshot:
    step_index: int
    belief: np.ndarray
    log_belief: np.ndarray
    belief_updated: bool
    evidence_track_ids: tuple[int, ...]
    inference_started: bool
    source_visible_now: bool
    source_seen: bool
    entropy: float
    map_cell: tuple[int, int]
    map_local_xy: tuple[float, float]
    credible_cell_counts: tuple[int, int, int]
    credible_region_masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    credible_areas_m2: tuple[float, float, float]
    evidence_map: np.ndarray
    update_gate: np.ndarray


def resolve_torch_device(name: str) -> torch.device:
    normalized = str(name).lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(normalized)


class OnlineSpatialBeliefRuntime:
    """Own exactly one recurrent log-belief and one feature stream."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        feature_config: GroundedTrackFeatureConfig,
        device: str | torch.device = "auto",
    ):
        self.device = (
            resolve_torch_device(device)
            if isinstance(device, str)
            else torch.device(device)
        )
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.checkpoint, self.model = load_spatial_checkpoint(
            self.checkpoint_path, self.device
        )
        config = self.model.config
        if abs(config.radius_m - feature_config.radius_m) > 1.0e-6:
            raise RuntimeError("Checkpoint/online belief radius mismatch")
        self.feature_config = feature_config
        self.encoder = StreamingGroundedTrackEncoder(feature_config)
        self._inference_started = False
        self._log_belief = None
        self._last_snapshot = None
        self.reset()

    @property
    def last_snapshot(self) -> OnlineBeliefSnapshot | None:
        return self._last_snapshot

    @property
    def log_belief_tensor(self) -> torch.Tensor:
        return self._log_belief

    def reset(self) -> None:
        self.encoder.reset()
        self._inference_started = False
        self._log_belief = self.model.initial_log_belief(
            1, torch.float32
        ).to(self.device)
        self._last_snapshot = None

    def _step_tensors(self, encoded: EncodedGroundedTrackStep):
        track_count = max(1, len(encoded.track_classes))
        features = np.zeros(
            (track_count, len(TRACK_FEATURE_NAMES)), dtype=np.float32
        )
        classes = np.zeros(track_count, dtype=np.int64)
        mask = np.zeros(track_count, dtype=np.bool_)
        count = len(encoded.track_classes)
        if count:
            features[:count] = encoded.track_features
            classes[:count] = encoded.track_classes
            mask[:count] = encoded.track_mask
        return (
            torch.from_numpy(features).unsqueeze(0).to(self.device),
            torch.from_numpy(classes).unsqueeze(0).to(self.device),
            torch.from_numpy(mask).unsqueeze(0).to(self.device),
        )

    def _summarize(
        self,
        encoded: EncodedGroundedTrackStep,
        output: dict[str, torch.Tensor],
        belief_updated: bool,
    ) -> OnlineBeliefSnapshot:
        belief = output["belief"][0].detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        log_belief = output["log_belief"][0].detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        valid = self.model.valid_mask.detach().cpu().numpy()
        if (
            not np.isfinite(belief[valid]).all()
            or np.any(belief < 0.0)
            or not np.isclose(float(belief.sum()), 1.0, atol=2.0e-5)
            or np.any(belief[~valid] != 0.0)
        ):
            raise RuntimeError("ONLINE_BELIEF_INVALID: posterior is not normalized")
        valid_values = belief[valid]
        entropy = float(
            -np.sum(valid_values[valid_values > 0.0] * np.log(valid_values[valid_values > 0.0]))
        )
        masked = np.where(valid, belief, -1.0)
        row, column = np.unravel_index(int(np.argmax(masked)), belief.shape)
        map_local = (
            float(self.model.grid_forward[row, column].item()),
            float(self.model.grid_right[row, column].item()),
        )
        ordered = np.sort(valid_values)[::-1]
        cumulative = np.cumsum(ordered)
        flat_order = np.argsort(masked.ravel())[::-1]
        counts = tuple(
            int(np.searchsorted(cumulative, threshold, side="left") + 1)
            for threshold in (0.5, 0.8, 0.9)
        )
        region_masks = []
        for count in counts:
            region = np.zeros(belief.size, dtype=np.bool_)
            region[flat_order[:count]] = True
            region_masks.append(region.reshape(belief.shape))
        region_masks = tuple(region_masks)
        cell_area = self.model.config.cell_m**2
        areas = tuple(float(count * cell_area) for count in counts)
        evidence_map = output["evidence_map"][0].detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        gate = output["update_gate"][0].detach().cpu().numpy().astype(
            np.float32, copy=True
        )
        return OnlineBeliefSnapshot(
            step_index=encoded.step_index,
            belief=belief,
            log_belief=log_belief,
            belief_updated=bool(belief_updated),
            evidence_track_ids=encoded.motion_evidence_track_ids,
            inference_started=self._inference_started,
            source_visible_now=encoded.source_visible_now,
            source_seen=encoded.source_seen,
            entropy=entropy,
            map_cell=(int(row), int(column)),
            map_local_xy=map_local,
            credible_cell_counts=counts,
            credible_region_masks=region_masks,
            credible_areas_m2=areas,
            evidence_map=evidence_map,
            update_gate=gate,
        )

    @torch.no_grad()
    def update(self, grounded_frame) -> OnlineBeliefSnapshot:
        encoded = self.encoder.encode(grounded_frame)
        if encoded.has_motion_evidence:
            self._inference_started = True
        update_allowed = self._inference_started and not encoded.source_seen
        belief_updated = update_allowed and encoded.has_motion_evidence
        features, classes, mask = self._step_tensors(encoded)
        output = self.model.forward_step(
            self._log_belief,
            features,
            classes,
            mask,
            torch.tensor([update_allowed], dtype=torch.bool, device=self.device),
        )
        self._log_belief = output["log_belief"]
        snapshot = self._summarize(encoded, output, belief_updated)
        self._last_snapshot = snapshot
        return snapshot
