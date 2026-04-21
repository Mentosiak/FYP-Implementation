from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments" / "benchmark_summary" / "20260405_010605" / "benchmark_summary.csv"
COMPARISON_DIR = ROOT / "experiments" / "comparison"
FIG_DIR = ROOT / "poster" / "figures"
TABLE_PATH = ROOT / "poster" / "poster_results_table.csv"
CH6_DRAFT_PATH = ROOT / "Chapter6_Testing_Evaluation_Draft.tex"

ALGO_ORDER = ["supervised", "pseudolabel", "fixmatch", "flexmatch", "mixmatch"]
TARGET_BUDGETS = [250, 1000, 4000]
CH6_METHOD_TO_ALGO = {
    "Supervised": "supervised",
    "Pseudo-Label": "pseudolabel",
    "FixMatch": "fixmatch",
    "FlexMatch": "flexmatch",
    "MixMatch": "mixmatch",
}
ALGO_LABELS = {
    "supervised": "Supervised",
    "pseudolabel": "Pseudo-Label",
    "fixmatch": "FixMatch",
    "flexmatch": "FlexMatch",
    "mixmatch": "MixMatch",
}
COLORS = {
    "supervised": "#595959",
    "pseudolabel": "#00897B",
    "fixmatch": "#1E88E5",
    "flexmatch": "#FB8C00",
    "mixmatch": "#8E24AA",
}


def parse_float(value: str) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_best_accuracy_by_budget() -> tuple[list[int], dict[str, dict[int, float]]]:
    best: dict[str, dict[int, float]] = defaultdict(dict)
    budgets: set[int] = set()

    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip().lower()
            algo = (row.get("algorithm") or "").strip().lower()
            labels_total = row.get("labels_total")
            acc = parse_float(row.get("best_test_acc") or "")

            if status != "complete" or algo not in ALGO_ORDER:
                continue
            if acc is None:
                continue

            budget = int(labels_total)
            budgets.add(budget)

            # Keep strongest completed run per method and label budget.
            prev = best[algo].get(budget)
            if prev is None or acc > prev:
                best[algo][budget] = acc

    # Merge comparison JSON results so newer reruns are included even if the
    # benchmark summary snapshot has not been regenerated.
    if COMPARISON_DIR.exists():
        for result_file in COMPARISON_DIR.glob("*/comparison_results.json"):
            with result_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            sup = payload.get("supervised") or {}
            sup_acc = parse_float(str(sup.get("best_test_acc", "")))
            sup_cfg = str(sup.get("config", ""))
            if sup_acc is not None and "supervised" in sup_cfg:
                for budget in (250, 500, 1000, 4000):
                    if f"_{budget}labels" in sup_cfg or f"_{budget}_" in sup_cfg:
                        budgets.add(budget)
                        prev = best["supervised"].get(budget)
                        if prev is None or sup_acc > prev:
                            best["supervised"][budget] = sup_acc
                        break

            ssl = payload.get("ssl") or {}
            algo = str(ssl.get("algorithm", "")).strip().lower()
            acc = parse_float(str(ssl.get("best_test_acc", "")))
            labels_per_class = ssl.get("labels_per_class")
            if algo in ALGO_ORDER and algo != "supervised" and acc is not None:
                budget = None
                if isinstance(labels_per_class, (int, float)):
                    budget = int(labels_per_class) * 10
                if budget is None:
                    cfg = str(ssl.get("config", ""))
                    for guess in (250, 500, 1000, 4000):
                        if f"_{guess}labels" in cfg:
                            budget = guess
                            break

                if budget is not None:
                    budgets.add(budget)
                    prev = best[algo].get(budget)
                    if prev is None or acc > prev:
                        best[algo][budget] = acc

    final_budgets = [b for b in TARGET_BUDGETS if b in budgets]
    return final_budgets, best


def write_results_table(budgets: list[int], best: dict[str, dict[int, float]]) -> None:
    with TABLE_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "labels_total", "best_test_acc"])
        for algo in ALGO_ORDER:
            for b in budgets:
                score = best.get(algo, {}).get(b)
                writer.writerow([algo, b, "" if score is None else f"{score:.2f}"])


def load_published_anchors_from_ch6() -> dict[str, dict[int, float]]:
    if not CH6_DRAFT_PATH.exists():
        return {}

    published: dict[str, dict[int, float]] = defaultdict(dict)
    row_re = re.compile(
        r"^\s*(?P<method>[^&]+?)\s*&\s*(?P<budget>\d+)\s*&\s*(?P<project>[0-9.]+)\s*&\s*(?P<published>[0-9.]+)\s*&"
    )

    for line in CH6_DRAFT_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        m = row_re.match(line)
        if not m:
            continue

        method = m.group("method").strip()
        algo = CH6_METHOD_TO_ALGO.get(method)
        if not algo:
            continue

        budget = int(m.group("budget"))
        if budget not in TARGET_BUDGETS:
            continue

        val = parse_float(m.group("published"))
        if val is None:
            continue

        published[algo][budget] = float(val)

    return published


def plot_project_vs_published_anchors(budgets: list[int], best: dict[str, dict[int, float]]) -> Path | None:
    published = load_published_anchors_from_ch6()
    if not published:
        return None

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        nrows=len(budgets),
        ncols=1,
        figsize=(12.0, 3.8 * len(budgets)),
        dpi=180,
        sharex=True,
    )
    if len(budgets) == 1:
        axes = [axes]

    x = list(range(len(ALGO_ORDER)))
    width = 0.36

    for ax, budget in zip(axes, budgets):
        project_vals = [best.get(algo, {}).get(budget, float("nan")) for algo in ALGO_ORDER]
        published_vals = [published.get(algo, {}).get(budget, float("nan")) for algo in ALGO_ORDER]

        bars_project = ax.bar(
            [xi - width / 2 for xi in x],
            project_vals,
            width=width,
            color=[COLORS[a] for a in ALGO_ORDER],
            edgecolor="white",
            linewidth=0.8,
            label="Project",
        )
        bars_pub = ax.bar(
            [xi + width / 2 for xi in x],
            published_vals,
            width=width,
            color="white",
            edgecolor=[COLORS[a] for a in ALGO_ORDER],
            linewidth=1.2,
            hatch="///",
            label="Published anchor (Chapter 6 draft)",
        )

        for bar in bars_project:
            v = bar.get_height()
            if math.isnan(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.35,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for bar in bars_pub:
            v = bar.get_height()
            if math.isnan(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.35,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(f"Project vs Published Anchors ({budget} labeled)", fontsize=14, weight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(0, 100)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([ALGO_LABELS[a] for a in ALGO_ORDER], fontsize=11)
    axes[0].legend(ncol=2, frameon=True, fontsize=9)

    fig.tight_layout()
    out_path = FIG_DIR / "project_vs_published_anchors.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_accuracy_grouped_bar(budgets: list[int], best: dict[str, dict[int, float]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=180)

    x = list(range(len(budgets)))
    width = 0.16
    center_offset = (len(ALGO_ORDER) - 1) / 2

    for i, algo in enumerate(ALGO_ORDER):
        offsets = [xi + (i - center_offset) * width for xi in x]
        vals = []
        for b in budgets:
            v = best.get(algo, {}).get(b)
            vals.append(float("nan") if v is None else v)

        bars = ax.bar(
            offsets,
            vals,
            width=width,
            label=ALGO_LABELS[algo],
            color=COLORS[algo],
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, v in zip(bars, vals):
            if math.isnan(v):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.35,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title("CIFAR-10 Accuracy by Label Budget", fontsize=16, weight="bold")
    ax.set_xlabel("Labeled samples", fontsize=12)
    ax.set_ylabel("Best test accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_by_budget.png", bbox_inches="tight")
    plt.close(fig)


def plot_improvement_over_supervised(budgets: list[int], best: dict[str, dict[int, float]]) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=180)
    ssl_algos = [a for a in ALGO_ORDER if a != "supervised"]
    x = list(range(len(budgets)))
    width = 0.18
    center_offset = (len(ssl_algos) - 1) / 2

    for i, algo in enumerate(ssl_algos):
        offsets = [xi + (i - center_offset) * width for xi in x]
        gains = []
        for b in budgets:
            sup = best.get("supervised", {}).get(b)
            val = best.get(algo, {}).get(b)
            if sup is None or val is None:
                gains.append(float("nan"))
            else:
                gains.append(val - sup)

        bars = ax.bar(
            offsets,
            gains,
            width=width,
            label=ALGO_LABELS[algo],
            color=COLORS[algo],
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, g in zip(bars, gains):
            if math.isnan(g):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"+{g:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_title("SSL Gain Over Supervised Baseline", fontsize=16, weight="bold")
    ax.set_xlabel("Labeled samples", fontsize=12)
    ax.set_ylabel("Accuracy gain (percentage points)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=11)
    ax.legend(ncol=2, frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "improvement_over_supervised.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {CSV_PATH}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    budgets, best = load_best_accuracy_by_budget()
    if not budgets:
        raise RuntimeError("No complete benchmark rows found for plotting.")

    write_results_table(budgets, best)
    plot_accuracy_grouped_bar(budgets, best)
    plot_improvement_over_supervised(budgets, best)
    anchors_plot = plot_project_vs_published_anchors(budgets, best)

    print("Generated files:")
    print(f"- {FIG_DIR / 'accuracy_by_budget.png'}")
    print(f"- {FIG_DIR / 'improvement_over_supervised.png'}")
    if anchors_plot is not None:
        print(f"- {anchors_plot}")
    print(f"- {TABLE_PATH}")


if __name__ == "__main__":
    main()
