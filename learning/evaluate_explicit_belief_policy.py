"""Evaluate the Stage 3C explicit-belief action policy offline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from learning.belief_dataset import discover_belief_episodes  # noqa: E402
from learning.belief_objective import belief_metrics  # noqa: E402
from learning.evaluate_spatial_belief import selected_records  # noqa: E402
from learning.policy_dataset import (  # noqa: E402
    ACTION_NAMES,
    StructuredBeliefPolicyDataset,
    collate_policy_episodes,
)
from learning.policy_objective import explicit_policy_objective  # noqa: E402
from learning.policy_runtime import load_policy_checkpoint, resolve_device  # noqa: E402
from learning.spatial_belief_runtime import METRIC_NAMES, last_true_mask  # noqa: E402


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="validation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--maximum-episodes", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=("normal", "uniform", "teacher", "no-depth", "rotate"),
        default=("normal",),
        help="Policy-input interventions; belief metrics always describe the predicted Spatial RNN belief",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.maximum_episodes is not None and args.maximum_episodes <= 0:
        parser.error("--maximum-episodes must be positive")
    return args


def _to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _forward(model, batch, ablation="normal"):
    geometry = (
        torch.zeros_like(batch["local_geometry"])
        if ablation == "no-depth"
        else batch["local_geometry"]
    )
    output = model.forward_sequence(
        batch["track_features"], batch["track_classes"], batch["track_mask"],
        batch["sequence_mask"], batch["belief_update_mask"],
        batch["pose_features"], batch["action_history"], geometry,
        batch["source_features"], batch["source_mask"], batch["source_position_local"],
    )
    if ablation in ("normal", "no-depth"):
        return output
    batch_size, steps = batch["sequence_mask"].shape
    valid = model.belief_updater.valid_mask[None, None]
    if ablation == "uniform":
        log_belief = model.belief_updater.initial_log_belief(
            batch_size, output["belief"].dtype
        )[:, None].expand(batch_size, steps, -1, -1)
        policy_belief = log_belief.exp()
    elif ablation == "teacher":
        policy_belief = torch.where(
            valid, batch["teacher_belief"].clamp_min(0.0),
            torch.zeros_like(batch["teacher_belief"]),
        )
        totals = policy_belief.sum(dim=(-2, -1), keepdim=True)
        if bool((totals <= 0.0).any()):
            raise RuntimeError("Teacher-belief policy ablation contains an empty map")
        policy_belief = policy_belief / totals
        log_belief = torch.where(
            valid,
            policy_belief.log(),
            torch.full_like(policy_belief, -torch.inf),
        )
    elif ablation == "rotate":
        policy_belief = torch.rot90(output["belief"], 1, dims=(-2, -1))
        policy_belief = torch.where(valid, policy_belief, torch.zeros_like(policy_belief))
        policy_belief = policy_belief / policy_belief.sum(
            dim=(-2, -1), keepdim=True
        )
        log_belief = torch.where(
            valid,
            policy_belief.clamp_min(1.0e-30).log(),
            torch.full_like(policy_belief, -torch.inf),
        )
    else:
        raise ValueError(f"Unknown Stage3C ablation: {ablation}")
    policy_rows = []
    for step in range(steps):
        policy_rows.append(model.policy_from_belief(
            policy_belief[:, step],
            log_belief[:, step],
            batch["pose_features"][:, step],
            batch["action_history"][:, step],
            geometry[:, step],
            batch["source_features"][:, step],
            batch["source_mask"][:, step],
            batch["source_position_local"][:, step],
        ))
    for name in (
        "action_logits", "action_probabilities", "remaining_value",
        "event_estimate_local", "agent_centric_belief",
    ):
        output[name] = torch.stack([row[name] for row in policy_rows], dim=1)
    return output


def _confusion(logits, target, mask):
    selected = logits.argmax(dim=-1)
    flat = target[mask] * len(ACTION_NAMES) + selected[mask]
    matrix = torch.zeros(
        len(ACTION_NAMES) * len(ACTION_NAMES), dtype=torch.long, device=logits.device
    )
    if flat.numel():
        matrix.scatter_add_(0, flat, torch.ones_like(flat))
    return matrix.reshape(len(ACTION_NAMES), len(ACTION_NAMES))


def _classification(matrix):
    matrix = matrix.to(torch.float64)
    true = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    diagonal = matrix.diag()
    recall = diagonal / true.clamp_min(1)
    precision = diagonal / predicted.clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1.0e-12)
    total = matrix.sum().clamp_min(1)
    return {
        "accuracy": float(diagonal.sum() / total),
        "macro_f1": float(f1.mean()),
        "per_class_recall": {
            name: float(recall[index]) for index, name in enumerate(ACTION_NAMES)
        },
        "confusion": matrix.to(torch.long).tolist(),
    }


def _format_classification(label, result):
    recalls = ",".join(
        f"{name}={result['per_class_recall'][name]:.3f}" for name in ACTION_NAMES
    )
    return (
        f"{label} accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
        f"recall[{recalls}]"
    )


@torch.no_grad()
def _evaluate(model, loader, device, class_weights, ablation="normal"):
    model.eval()
    totals = defaultdict(float)
    episodes = 0
    action_steps = 0
    matrices = {
        "all": torch.zeros(7, 7, dtype=torch.long),
        "pre_source": torch.zeros(7, 7, dtype=torch.long),
        "source_visible": torch.zeros(7, 7, dtype=torch.long),
    }
    coordinate_errors = []
    anchor_matrices = defaultdict(
        lambda: torch.zeros(len(ACTION_NAMES), len(ACTION_NAMES), dtype=torch.long)
    )
    anchor_coordinate_errors = defaultdict(list)
    belief_rows = {
        "inference": defaultdict(float),
        "last_source_blind": defaultdict(float),
    }
    for raw in loader:
        batch = _to_device(raw, device)
        output = _forward(model, batch, ablation)
        objective = explicit_policy_objective(output, batch, class_weights)
        count = int(batch["sequence_mask"].shape[0])
        episodes += count
        for name in ("loss", "action_loss", "belief_loss", "coordinate_loss", "value_loss"):
            totals[name] += float(objective[name]) * count
        labeled = batch["sequence_mask"] & batch.get(
            "action_label_mask", batch["sequence_mask"]
        )
        masks = {
            "all": labeled,
            "pre_source": labeled & ~batch["source_visible_mask"],
            "source_visible": labeled & batch["source_visible_mask"],
        }
        for label, mask in masks.items():
            matrices[label] += _confusion(
                output["action_logits"], batch["action_target"], mask
            ).cpu()
        all_mask = masks["all"]
        raw_nll = F.cross_entropy(
            output["action_logits"].transpose(1, 2),
            batch["action_target"],
            reduction="none",
        )
        totals["raw_action_nll_sum"] += float(raw_nll[all_mask].sum())
        totals["value_absolute_sum"] += float(
            torch.abs(output["remaining_value"] - batch["value_target"])[all_mask].sum()
        )
        action_steps += int(all_mask.sum())
        source_mask = batch["source_mask"] & batch["sequence_mask"]
        if bool(source_mask.any()):
            target = batch["event_xyz"][:, None, :].expand_as(output["event_estimate_local"])
            coordinate_errors.extend(
                torch.linalg.vector_norm(output["event_estimate_local"] - target, dim=-1)[source_mask]
                .cpu().tolist()
            )
        for row_index, anchor_name in enumerate(batch["anchor_name"]):
            row_mask = all_mask[row_index : row_index + 1]
            anchor_matrices[anchor_name] += _confusion(
                output["action_logits"][row_index : row_index + 1],
                batch["action_target"][row_index : row_index + 1],
                row_mask,
            ).cpu()
            row_source_mask = source_mask[row_index]
            if bool(row_source_mask.any()):
                target = batch["event_xyz"][row_index][None].expand_as(
                    output["event_estimate_local"][row_index]
                )
                anchor_coordinate_errors[anchor_name].extend(
                    torch.linalg.vector_norm(
                        output["event_estimate_local"][row_index] - target, dim=-1
                    )[row_source_mask].cpu().tolist()
                )
        for label, mask in (
            ("inference", batch["inference_mask"]),
            ("last_source_blind", last_true_mask(batch["inference_mask"])),
        ):
            metrics = belief_metrics(
                output,
                batch["teacher_belief"],
                batch["event_cell"],
                batch["event_xy"],
                mask,
                model.belief_updater.grid_forward,
                model.belief_updater.grid_right,
            )
            weight = metrics["steps"]
            belief_rows[label]["steps"] += weight
            for name in METRIC_NAMES:
                belief_rows[label][name] += metrics[name] * weight
    if episodes == 0 or action_steps == 0:
        raise RuntimeError("Evaluation split is empty")
    stop = ACTION_NAMES.index("STOP")
    matrix = matrices["all"].to(torch.float64)
    stop_tp = float(matrix[stop, stop])
    stop_precision = stop_tp / max(1.0, float(matrix[:, stop].sum()))
    stop_recall = stop_tp / max(1.0, float(matrix[stop].sum()))
    belief = {}
    for label, row in belief_rows.items():
        steps = row["steps"]
        belief[label] = {name: row[name] / steps for name in METRIC_NAMES}
    return {
        "episodes": episodes,
        "steps": action_steps,
        "losses": {
            name: totals[name] / episodes
            for name in ("loss", "action_loss", "belief_loss", "coordinate_loss", "value_loss")
        },
        "action_nll": totals["raw_action_nll_sum"] / action_steps,
        "value_mae": totals["value_absolute_sum"] / action_steps,
        "coordinate_error_m": float(np.mean(coordinate_errors)) if coordinate_errors else math.nan,
        "stop_precision": stop_precision,
        "stop_recall": stop_recall,
        "classification": {
            name: _classification(value) for name, value in matrices.items()
        },
        "belief": belief,
        "per_anchor": {
            anchor_name: {
                "classification": _classification(anchor_matrix),
                "coordinate_error_m": (
                    float(np.mean(anchor_coordinate_errors[anchor_name]))
                    if anchor_coordinate_errors[anchor_name] else math.nan
                ),
            }
            for anchor_name, anchor_matrix in sorted(anchor_matrices.items())
        },
    }


def _belief_line(label, metrics):
    return (
        f"belief {label} nll={metrics['event_nll']:.4f} map_error={metrics['map_error_m']:.2f}m "
        f"rank={metrics['event_rank']:.1f} top10={metrics['top_10_recall']:.3f} "
        f"top50={metrics['top_50_recall']:.3f} top100={metrics['top_100_recall']:.3f} "
        f"entropy={metrics['entropy']:.3f} coverage80={metrics['coverage_80']:.3f} "
        f"area80={metrics['credible_area_80_m2']:.1f}m2 teacher_kl={metrics['teacher_kl']:.4f}"
    )


def main():
    args = _arguments()
    device = resolve_device(args.device)
    checkpoint, model, geometry = load_policy_checkpoint(args.checkpoint, device)
    records = selected_records(
        discover_belief_episodes(args.dataset_root), checkpoint, args.split
    )
    if args.maximum_episodes is not None:
        records = records[: args.maximum_episodes]
    dataset = StructuredBeliefPolicyDataset(
        records, augment_d4=False, geometry_config=geometry
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_policy_episodes,
        num_workers=0,
    )
    counts = checkpoint.get("action_class_counts", {})
    weights = torch.tensor(
        [float(checkpoint["action_class_weights"][name]) for name in ACTION_NAMES],
        dtype=torch.float32,
        device=device,
    )
    print(
        f"PASS Stage3C evaluation split={args.split} device={device} "
        f"epoch={checkpoint.get('epoch')} "
        f"anchors={sorted({record.anchor_name for record in records})} counts={counts}"
    )
    results = {}
    for ablation in dict.fromkeys(args.ablations):
        result = _evaluate(model, loader, device, weights, ablation)
        results[ablation] = result
        losses = result["losses"]
        all_metrics = result["classification"]["all"]
        print(
            f"COMPARISON ablation={ablation} episodes={result['episodes']} steps={result['steps']} "
            f"loss={losses['loss']:.4f} action_nll={result['action_nll']:.4f} "
            f"accuracy={all_metrics['accuracy']:.4f} macro_f1={all_metrics['macro_f1']:.4f} "
            f"stop_precision={result['stop_precision']:.3f} stop_recall={result['stop_recall']:.3f} "
            f"coordinate_error={result['coordinate_error_m']:.3f}m value_mae={result['value_mae']:.4f}"
        )
        for label in ("pre_source", "source_visible"):
            print(_format_classification(f"  {label}", result["classification"][label]))
        if ablation == "normal":
            print("  confusion rows=true columns=predicted order=" + ",".join(ACTION_NAMES))
            for action_name, row in zip(
                ACTION_NAMES, all_metrics["confusion"], strict=True
            ):
                print(f"    {action_name}: {row}")
        for anchor_name, anchor_result in result["per_anchor"].items():
            classification = anchor_result["classification"]
            print(
                f"  anchor={anchor_name} accuracy={classification['accuracy']:.4f} "
                f"macro_f1={classification['macro_f1']:.4f} "
                f"coordinate_error={anchor_result['coordinate_error_m']:.3f}m"
            )
    reference = results[next(iter(results))]
    majority_index = max(
        range(len(ACTION_NAMES)), key=lambda index: int(counts.get(ACTION_NAMES[index], 0))
    )
    confusion = np.asarray(reference["classification"]["all"]["confusion"])
    majority_accuracy = float(confusion[majority_index].sum() / max(1, confusion.sum()))
    print(
        f"BASELINE majority={ACTION_NAMES[majority_index]} accuracy={majority_accuracy:.4f}"
    )
    normal = results.get("normal", reference)
    print(_belief_line("inference", normal["belief"]["inference"]))
    print(_belief_line("last_source_blind", normal["belief"]["last_source_blind"]))
    print("teacher_KL is diagnostic only. No RGB-D or prediction payload was written to disk.")


if __name__ == "__main__":
    main()
