"""Plot average_at_n_pct vs checkpoint step from eval JSON files.

This follows the visual style of plot_opd_avg_at_n.py, but reads the eval
summaries directly from eval/eval_results and plots only average_at_n_pct.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "eval" / "eval_results"
OUT_DIR = RESULTS_DIR / "plots"

METRIC = "average_at_n_pct"

DATASET_TITLE = {
    "aime24": "AIME 2024",
    "aime25": "AIME 2025",
    "hmmt25": "HMMT 2025",
}

DIVERGENCE_ORDER = [
    "reverse_kl",
    "improved_reverse_kl",
    "forward_kl",
    "improved_forward_kl",
    "jsd",
    "improved_jsd",
]

PRETTY = {
    "reverse_kl": r"Reverse KL",
    "improved_reverse_kl": r"Reverse KL$^{*}$",
    "forward_kl": r"Forward KL",
    "improved_forward_kl": r"Forward KL$^{*}$",
    "jsd": r"JSD",
    "improved_jsd": r"JSD$^{*}$",
}

PALETTE = {
    "reverse_kl": "#1b9e77",
    "improved_reverse_kl": "#1b9e77",
    "forward_kl": "#d95f02",
    "improved_forward_kl": "#d95f02",
    "jsd": "#7570b3",
    "improved_jsd": "#7570b3",
}

IS_IMPROVED = {k: k.startswith("improved_") for k in DIVERGENCE_ORDER}


def _set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "savefig.bbox": "tight",
        }
    )


def _load_json(path: Path) -> tuple[dict, bool]:
    """Load one eval JSON file, recovering from leading junk if needed."""
    text = path.read_text()
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        first_object = text.find("{")
        if first_object <= 0:
            raise
        return json.loads(text[first_object:]), True


def _extract_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(f"could not find checkpoint step in {path}")
    return int(match.group(1))


def _extract_divergence(path: Path) -> str:
    haystack = f"{path.parent.name}_{path.name}"
    for divergence in sorted(DIVERGENCE_ORDER, key=len, reverse=True):
        if divergence in haystack:
            return divergence
    raise ValueError(f"could not infer divergence type from {path}")


def _collect_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    recovered = []
    skipped = []

    for path in sorted(results_dir.rglob("*.json")):
        try:
            data, did_recover = _load_json(path)
            dataset = data["dataset"]
            value = float(data[METRIC])
            rows.append(
                {
                    "dataset": dataset,
                    "divergence": _extract_divergence(path),
                    "step": _extract_step(path),
                    METRIC: value,
                    "path": str(path),
                }
            )
            if did_recover:
                recovered.append(path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            skipped.append((path, exc))

    for path in recovered:
        print(f"warning: recovered JSON after leading junk: {path}")
    for path, exc in skipped:
        print(f"warning: skipped {path}: {exc}")

    if not rows:
        raise RuntimeError(f"found no usable eval JSON files under {results_dir}")

    df = pd.DataFrame(rows)
    duplicate_keys = ["dataset", "divergence", "step"]
    duplicates = df[df.duplicated(duplicate_keys, keep=False)]
    if not duplicates.empty:
        print("warning: duplicate points found; averaging duplicate metric values")
        df = (
            df.groupby(duplicate_keys, as_index=False)[METRIC]
            .mean()
            .sort_values(duplicate_keys)
        )

    return df


def _plot_curve(ax, steps, values, divergence: str, label: str | None = None) -> None:
    improved = IS_IMPROVED[divergence]
    ax.plot(
        steps,
        values,
        color=PALETTE[divergence],
        linestyle="-" if improved else (0, (5, 2)),
        linewidth=3.0 if improved else 2.2,
        alpha=0.8 if improved else 1.0,
        marker="o" if improved else "s",
        markersize=7 if improved else 6,
        markerfacecolor=PALETTE[divergence] if improved else "white",
        markeredgecolor=PALETTE[divergence],
        markeredgewidth=1.4,
        label=label,
        zorder=3 if improved else 2,
    )


def _add_hazard_title(fig, title_text: str, ax_for_extent):
    """Centered title flanked by thick bars spanning the axes width."""
    from matplotlib.lines import Line2D

    title_y = 0.955
    bbox = ax_for_extent.get_position()
    x_left = bbox.x0
    x_right = bbox.x1
    title_center_x = (x_left + x_right) / 2

    txt = fig.text(
        title_center_x,
        title_y,
        title_text,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#111111",
        zorder=10,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_bbox_fig = txt.get_window_extent(renderer=renderer).transformed(
        fig.transFigure.inverted()
    )

    pad = 0.020
    spans = (
        (x_left, max(x_left, text_bbox_fig.x0 - pad)),
        (min(x_right, text_bbox_fig.x1 + pad), x_right),
    )

    bars = []
    for x0, x1 in spans:
        if x1 - x0 <= 0:
            continue
        bar = Line2D(
            [x0, x1],
            [title_y, title_y],
            transform=fig.transFigure,
            color="#111111",
            linewidth=2.6,
            solid_capstyle="butt",
            clip_on=False,
            zorder=9,
        )
        fig.add_artist(bar)
        bars.append(bar)

    return [txt, *bars]


def _nice_dataset_title(dataset: str) -> str:
    return DATASET_TITLE.get(dataset, dataset.replace("_", " ").upper())


def _plot_dataset(df: pd.DataFrame, dataset: str, out_dir: Path) -> None:
    sub = df[df["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.subplots_adjust(top=0.88, bottom=0.24)

    for divergence in DIVERGENCE_ORDER:
        d = sub[sub["divergence"] == divergence].sort_values("step")
        if d.empty:
            continue
        _plot_curve(
            ax,
            list(d["step"]),
            list(d[METRIC]),
            divergence,
            label=PRETTY[divergence],
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(r"average@$N$ accuracy (%)")

    steps = sorted(sub["step"].unique())
    ax.set_xticks(steps)
    if len(steps) > 6:
        ax.tick_params(axis="x", labelrotation=35)

    ymin = max(0.0, sub[METRIC].min() - 2.5)
    ymax = min(100.0, sub[METRIC].max() + 2.5)
    if ymax - ymin < 8:
        center = (ymax + ymin) / 2
        ymin = max(0.0, center - 4)
        ymax = min(100.0, center + 4)
    ax.set_ylim(ymin, ymax)

    fig.canvas.draw()
    title_artists = _add_hazard_title(fig, _nice_dataset_title(dataset), ax)

    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
        handletextpad=0.55,
        labelspacing=0.45,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"eval_{dataset}_{METRIC}.png"
    pdf = out_dir / f"eval_{dataset}_{METRIC}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", bbox_extra_artists=[leg, *title_artists])
    fig.savefig(pdf, bbox_inches="tight", bbox_extra_artists=[leg, *title_artists])
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    _set_paper_style()
    df = _collect_results(args.results_dir)
    for dataset in sorted(df["dataset"].unique()):
        _plot_dataset(df, dataset, args.out_dir)


if __name__ == "__main__":
    main()
