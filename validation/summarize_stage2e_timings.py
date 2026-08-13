"""Summarize Stage 2E batch timing JSONL without writing output files."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ATTEMPT_PHASES = (
    "prepare",
    "lockstep_setup",
    "start_audit",
    "rollout",
    "write_finalize",
    "cleanup",
    "total",
)

START_PHASES = (
    "task_start_generation_seconds",
    "rgbd_grounding_seconds",
    "static_goal_budget_audit_seconds",
    "total_seconds",
)

ROLLOUT_PHASES = (
    "setup_seconds",
    "visibility_seconds",
    "scenario_snapshot_seconds",
    "grounding_seconds",
    "teacher_seconds",
    "recording_seconds",
    "action_pose_seconds",
    "action_advance_seconds",
    "action_capture_seconds",
    "action_total_seconds",
    "cue_sensitivity_seconds",
    "geometry_query_seconds",
    "total_seconds",
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize timings.jsonl emitted by the Stage 2E batch "
            "generator. This command is read-only."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="A timings.jsonl file or its containing output directory",
    )
    return parser.parse_args()


def _load(path):
    path = path.resolve()
    if path.is_dir():
        path = path / "timings.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Timing file does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object"
                )
            records.append(value)
    if not records:
        raise ValueError(f"Timing file is empty: {path}")
    return path, records


def _finite_values(records, section, fields):
    values = defaultdict(list)
    for record in records:
        payload = record.get(section)
        if not isinstance(payload, dict):
            continue
        for field in fields:
            value = payload.get(field)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values[field].append(float(value))
    return values


def _print_table(title, values, denominator_field):
    denominator = sum(values.get(denominator_field, ()))
    print(title)
    for field, samples in values.items():
        if not samples:
            continue
        array = np.asarray(samples, dtype=np.float64)
        share = (
            100.0 * float(np.sum(array)) / denominator
            if denominator > 0.0 and field != denominator_field
            else None
        )
        suffix = "" if share is None else f" share={share:.1f}%"
        print(
            f"  {field:<38} n={len(array):<4d} "
            f"mean={np.mean(array):8.3f}s "
            f"p50={np.percentile(array, 50):8.3f}s "
            f"p95={np.percentile(array, 95):8.3f}s"
            f"{suffix}"
        )


def main():
    args = _parse_args()
    path, records = _load(args.path)
    outcomes = Counter(str(record.get("outcome", "UNKNOWN")) for record in records)
    print(f"file={path} attempts={len(records)} outcomes={dict(outcomes)}")

    for outcome in sorted(outcomes):
        selected = [
            record
            for record in records
            if str(record.get("outcome", "UNKNOWN")) == outcome
        ]
        print(f"\n[{outcome}] attempts={len(selected)}")
        _print_table(
            "attempt timing",
            _finite_values(selected, "timing", ATTEMPT_PHASES),
            "total",
        )
        _print_table(
            "audited-start timing",
            _finite_values(selected, "audited_start", START_PHASES),
            "total_seconds",
        )
        _print_table(
            "rollout timing",
            _finite_values(selected, "rollout", ROLLOUT_PHASES),
            "total_seconds",
        )


if __name__ == "__main__":
    main()
