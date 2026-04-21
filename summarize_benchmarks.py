"""
Build a consolidated benchmark summary from configs, logs, checkpoints, and comparison outputs.

Outputs:
- experiments/benchmark_summary/<timestamp>/benchmark_summary.json
- experiments/benchmark_summary/<timestamp>/benchmark_summary.csv
- experiments/benchmark_summary/<timestamp>/benchmark_matrix.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


BEST_ACC_PATTERN = re.compile(r"Best Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%")
EPOCH_PATTERN = re.compile(r"Epoch\s*\[(\d+)/(\d+)\]")
GPU_MEM_PATTERN = re.compile(r"GPU Mem Used:\s*([0-9]+(?:\.[0-9]+)?)\s*MiB")
GPU_UTIL_PATTERN = re.compile(r"GPU Util:\s*([0-9]+(?:\.[0-9]+)?)%")


@dataclass
class RunRecord:
    run_name: str
    config_path: str
    algorithm: str
    labels_per_class: int | None
    labels_total: int | None
    log_path: str | None
    checkpoint_best_path: str | None
    best_test_acc: float | None
    epochs_completed: int | None
    expected_epochs: int | None
    peak_gpu_mem_used_mb: float | None
    peak_gpu_util_pct: float | None
    comparison_dir: str | None
    comparison_file: str | None
    status: str
    notes: str


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def latest_log_for_run(logs_root: Path, run_name: str) -> Path | None:
    run_dir = logs_root / run_name
    if not run_dir.exists():
        return None
    log_files = sorted(run_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return log_files[0] if log_files else None


def parse_log_metrics(log_path: Path) -> tuple[float | None, int | None, float | None, float | None]:
    best_acc = None
    max_epoch = None
    peak_gpu_mem = None
    peak_gpu_util = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            best_match = BEST_ACC_PATTERN.search(line)
            if best_match:
                value = float(best_match.group(1))
                if best_acc is None or value > best_acc:
                    best_acc = value

            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                epoch_value = int(epoch_match.group(1))
                if max_epoch is None or epoch_value > max_epoch:
                    max_epoch = epoch_value

            mem_match = GPU_MEM_PATTERN.search(line)
            if mem_match:
                mem_value = float(mem_match.group(1))
                if peak_gpu_mem is None or mem_value > peak_gpu_mem:
                    peak_gpu_mem = mem_value

            util_match = GPU_UTIL_PATTERN.search(line)
            if util_match:
                util_value = float(util_match.group(1))
                if peak_gpu_util is None or util_value > peak_gpu_util:
                    peak_gpu_util = util_value

    return best_acc, max_epoch, peak_gpu_mem, peak_gpu_util


def find_comparison_hits(comparison_root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """
    Map (supervised_config, ssl_config) -> (comparison_dir, comparison_file).
    """
    hits: dict[tuple[str, str], tuple[str, str]] = {}
    for file_path in comparison_root.glob("*/comparison_results.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        sup_cfg = payload.get("supervised", {}).get("config")
        ssl_cfg = payload.get("ssl", {}).get("config")
        if not sup_cfg or not ssl_cfg:
            continue

        hits[(str(sup_cfg), str(ssl_cfg))] = (str(file_path.parent.as_posix()), str(file_path.as_posix()))
    return hits


def detect_status(
    best_acc: float | None,
    best_ckpt_exists: bool,
    epochs_completed: int | None,
    expected_epochs: int | None,
) -> str:
    """Classify run status.

    Notes:
    - "complete" means the run appears to have reached the configured epoch budget.
    - "partial" includes early-stopped runs that produced checkpoints/metrics but did not
      reach the configured epoch budget.
    """
    if best_acc is None and not best_ckpt_exists:
        return "missing"

    if best_acc is not None and best_ckpt_exists:
        if (
            isinstance(epochs_completed, int)
            and isinstance(expected_epochs, int)
            and epochs_completed < expected_epochs
        ):
            return "partial"
        return "complete"

    return "partial"


def build_record(
    config_path: Path,
    project_root: Path,
    logs_root: Path,
    checkpoints_root: Path,
    comparison_hits: dict[tuple[str, str], tuple[str, str]],
) -> RunRecord:
    cfg = load_yaml(config_path)
    run_name = str(cfg.get("run_name"))

    ssl_section = cfg.get("ssl", {}) or {}
    is_ssl = bool(ssl_section.get("enabled", False))
    if is_ssl:
        algorithm = str(ssl_section.get("algorithm", "ssl")).lower()
    else:
        algorithm = "supervised"

    labels_per_class = ssl_section.get("labels_per_class")
    num_classes = cfg.get("dataset", {}).get("num_classes")
    if isinstance(labels_per_class, int) and isinstance(num_classes, int):
        labels_total = labels_per_class * num_classes
    else:
        labels_total = None

    expected_epochs = cfg.get("training", {}).get("epochs")
    expected_epochs = int(expected_epochs) if isinstance(expected_epochs, int) else None

    latest_log = latest_log_for_run(logs_root, run_name)
    if latest_log is not None:
        best_acc, epochs_completed, peak_gpu_mem_used_mb, peak_gpu_util_pct = parse_log_metrics(latest_log)
        log_path_str = str(latest_log.as_posix())
    else:
        best_acc, epochs_completed, peak_gpu_mem_used_mb, peak_gpu_util_pct = None, None, None, None
        log_path_str = None

    best_ckpt = checkpoints_root / run_name / f"{run_name}_best.pt"
    best_ckpt_exists = best_ckpt.exists()
    best_ckpt_str = str(best_ckpt.as_posix()) if best_ckpt_exists else None

    status = detect_status(best_acc, best_ckpt_exists, epochs_completed, expected_epochs)
    notes = ""

    if algorithm == "mixmatch" and labels_total == 250 and best_acc is not None and best_acc < 40.0:
        notes = "underperforming run; rerun or tune recommended"

    # Comparison availability only applies to supervised-vs-SSL pair outputs.
    comparison_dir = None
    comparison_file = None
    if algorithm != "supervised":
        label_suffix = f"{labels_total}labels" if labels_total is not None else ""
        supervised_cfg_name = f"supervised_limited_cifar10_{label_suffix}.yaml" if label_suffix else ""
        supervised_cfg_rel = (
            f"configs/benchmarks/{supervised_cfg_name}" if supervised_cfg_name else ""
        )
        ssl_cfg_rel = str(config_path.relative_to(project_root).as_posix())

        if supervised_cfg_rel and (supervised_cfg_rel, ssl_cfg_rel) in comparison_hits:
            comparison_dir, comparison_file = comparison_hits[(supervised_cfg_rel, ssl_cfg_rel)]

    return RunRecord(
        run_name=run_name,
        config_path=str(config_path.relative_to(project_root).as_posix()),
        algorithm=algorithm,
        labels_per_class=labels_per_class if isinstance(labels_per_class, int) else None,
        labels_total=labels_total,
        log_path=log_path_str,
        checkpoint_best_path=best_ckpt_str,
        best_test_acc=best_acc,
        epochs_completed=epochs_completed,
        expected_epochs=expected_epochs,
        peak_gpu_mem_used_mb=peak_gpu_mem_used_mb,
        peak_gpu_util_pct=peak_gpu_util_pct,
        comparison_dir=comparison_dir,
        comparison_file=comparison_file,
        status=status,
        notes=notes,
    )


def write_csv(records: list[RunRecord], path: Path) -> None:
    fieldnames = [
        "run_name",
        "config_path",
        "algorithm",
        "labels_per_class",
        "labels_total",
        "best_test_acc",
        "epochs_completed",
        "expected_epochs",
        "peak_gpu_mem_used_mb",
        "peak_gpu_util_pct",
        "status",
        "notes",
        "log_path",
        "checkpoint_best_path",
        "comparison_dir",
        "comparison_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "run_name": r.run_name,
                    "config_path": r.config_path,
                    "algorithm": r.algorithm,
                    "labels_per_class": r.labels_per_class,
                    "labels_total": r.labels_total,
                    "best_test_acc": r.best_test_acc,
                    "epochs_completed": r.epochs_completed,
                    "expected_epochs": r.expected_epochs,
                    "peak_gpu_mem_used_mb": r.peak_gpu_mem_used_mb,
                    "peak_gpu_util_pct": r.peak_gpu_util_pct,
                    "status": r.status,
                    "notes": r.notes,
                    "log_path": r.log_path,
                    "checkpoint_best_path": r.checkpoint_best_path,
                    "comparison_dir": r.comparison_dir,
                    "comparison_file": r.comparison_file,
                }
            )


def write_markdown(records: list[RunRecord], path: Path) -> None:
    labels = [250, 1000, 4000]
    algorithms = ["supervised", "pseudolabel", "fixmatch", "mixmatch", "flexmatch"]

    index: dict[tuple[str, int], RunRecord] = {}
    for r in records:
        if r.labels_total in labels and r.algorithm in algorithms:
            index[(r.algorithm, r.labels_total)] = r

    lines: list[str] = []
    lines.append("# Benchmark Matrix")
    lines.append("")
    lines.append("| Algorithm | 250 labels | 1000 labels | 4000 labels |")
    lines.append("|---|---:|---:|---:|")

    for algo in algorithms:
        row = [algo]
        for label_total in labels:
            rec = index.get((algo, label_total))
            if rec is None:
                row.append("missing")
                continue
            if rec.best_test_acc is None:
                row.append(f"{rec.status}")
            else:
                cell = f"{rec.best_test_acc:.2f}% ({rec.status})"
                if rec.notes:
                    cell += " *"
                row.append(cell)
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    lines.append("")
    lines.append("* Notes marker indicates a quality warning in run notes.")
    lines.append("")
    lines.append("## Detailed Notes")
    lines.append("")
    for r in records:
        if r.notes:
            lines.append(
                f"- {r.run_name}: {r.notes} (best_acc={r.best_test_acc}, status={r.status})"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark runs into a consolidated report")
    parser.add_argument("--project-root", type=str, default=".", help="Project root path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/benchmark_summary",
        help="Output directory root",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    benchmarks_dir = project_root / "configs" / "benchmarks"
    logs_root = project_root / "logs"
    checkpoints_root = project_root / "checkpoints"
    comparison_root = project_root / "experiments" / "comparison"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (project_root / args.output_dir / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_hits = find_comparison_hits(comparison_root)

    records: list[RunRecord] = []
    config_files = sorted(benchmarks_dir.glob("*.yaml"))
    for cfg_path in config_files:
        records.append(
            build_record(
                config_path=cfg_path,
                project_root=project_root,
                logs_root=logs_root,
                checkpoints_root=checkpoints_root,
                comparison_hits=comparison_hits,
            )
        )

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(project_root.as_posix()),
        "records": [r.__dict__ for r in records],
    }

    json_path = out_dir / "benchmark_summary.json"
    csv_path = out_dir / "benchmark_summary.csv"
    md_path = out_dir / "benchmark_matrix.md"

    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_csv(records, csv_path)
    write_markdown(records, md_path)

    print(f"Saved JSON: {json_path.as_posix()}")
    print(f"Saved CSV:  {csv_path.as_posix()}")
    print(f"Saved MD:   {md_path.as_posix()}")


if __name__ == "__main__":
    main()
