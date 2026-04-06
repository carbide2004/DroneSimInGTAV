import argparse
import json
from pathlib import Path

import numpy as np


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Visualize stage2 offline evaluation results")
    parser.add_argument(
        "--input_json",
        default=str(Path(__file__).resolve().parent.parent / "agent_control" / "checkpoints" / "stage2" / "offline_eval.json"),
        help="Path to offline_eval.json",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent.parent / "agent_control" / "checkpoints" / "stage2" / "viz"),
        help="Output directory for figures",
    )
    parser.add_argument("--show", action="store_true", help="Show interactive windows")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("需要 matplotlib 才能可视化，请先安装 matplotlib。") from e

    data = _read_json(Path(args.input_json))
    labels = data.get("labels", [])
    cm = np.array(data.get("confusion_matrix", []), dtype=np.float32)
    per_class = data.get("per_class", {})
    summary = data.get("summary", {})

    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    if cm.size > 0:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, np.maximum(row_sum, 1.0))

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title("Stage2 Confusion Matrix (Row-Normalized)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                v = cm_norm[i, j]
                text_color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=text_color, fontsize=8)

        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix_norm.png", dpi=180)
        if args.show:
            plt.show()
        plt.close(fig)

    if per_class:
        names = list(labels) if labels else list(per_class.keys())
        f1 = [float(per_class.get(n, {}).get("f1", 0.0)) for n in names]
        precision = [float(per_class.get(n, {}).get("precision", 0.0)) for n in names]
        recall = [float(per_class.get(n, {}).get("recall", 0.0)) for n in names]

        x = np.arange(len(names))
        w = 0.26
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - w, precision, width=w, label="Precision")
        ax.bar(x, recall, width=w, label="Recall")
        ax.bar(x + w, f1, width=w, label="F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Stage2 Per-Class Metrics")
        ax.set_xlabel("Action")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "per_class_metrics.png", dpi=180)
        if args.show:
            plt.show()
        plt.close(fig)

    report_path = output_dir / "summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Stage2 Offline Evaluation Summary\n")
        f.write("=" * 40 + "\n")
        for key in [
            "split",
            "total_steps",
            "valid_predictions",
            "invalid_predictions",
            "accuracy_valid_predictions",
            "avg_action_ce",
        ]:
            if key in summary:
                f.write(f"{key}: {summary[key]}\n")
    print(f"Saved figures to: {output_dir}")
    print(f"Saved summary to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
