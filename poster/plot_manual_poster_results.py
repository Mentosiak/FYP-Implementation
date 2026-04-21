from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "poster" / "figures"

ALGO_ORDER = ["supervised", "pseudolabel", "fixmatch", "flexmatch", "mixmatch"]
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

# Results provided by user (Apr 16, 2026)
BUDGETS = [250, 1000, 4000]
BEST = {
    "supervised": {250: 38.5, 1000: 52.0, 4000: 75.5},
    "pseudolabel": {250: 66.0, 1000: 89.0, 4000: 90.0},
    "fixmatch": {250: 83.0, 1000: 84.0, 4000: 90.0},
    "flexmatch": {250: 81.0, 1000: 83.2, 4000: 89.9},
    "mixmatch": {250: 78.0, 1000: 80.0, 4000: 91.5},
}


def plot_accuracy_grouped_bar(budgets: list[int], best: dict[str, dict[int, float]]) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15.5, 9.2), dpi=300)

    x = list(range(len(budgets)))
    width = 0.16
    center_offset = (len(ALGO_ORDER) - 1) / 2

    for i, algo in enumerate(ALGO_ORDER):
        offsets = [xi + (i - center_offset) * width for xi in x]
        vals = [best.get(algo, {}).get(b, float("nan")) for b in budgets]

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
                bar.get_height() + 0.5,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="#1f1f1f",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.25},
            )

    ax.set_title("CIFAR-10 Accuracy by Label Budget", fontsize=30, weight="bold", pad=16)
    ax.set_xlabel("Labeled samples", fontsize=22, labelpad=10)
    ax.set_ylabel("Best test accuracy (%)", fontsize=22, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.35)
    ax.set_axisbelow(True)
    legend = ax.legend(
        ncol=5,
        frameon=True,
        fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        borderaxespad=0.6,
    )
    legend.get_frame().set_alpha(0.95)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "accuracy_by_budget_rerun_poster.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    out = plot_accuracy_grouped_bar(BUDGETS, BEST)
    print(out)


if __name__ == "__main__":
    main()
