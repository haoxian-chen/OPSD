"""Plot avg@n vs checkpoint step for each eval family (one figure per family).

Conference-style design with a broken y-axis so the collapsed (forward_kl / jsd)
runs do not compress the interesting region around the baseline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

CSV_PATH = Path("/Users/chenhaoxian/Distillation-v1/results/opd_eval_with_stage0_baselines.csv")
OUT_DIR = Path("/Users/chenhaoxian/Distillation-v1/results")

FAMILIES = ["aime24", "aime25", "hmmt25", "average"]
FAMILY_TITLE = {
    "aime24": "AIME 2024",
    "aime25": "AIME 2025",
    "hmmt25": "HMMT 2025",
    "average": "AVERAGE",
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
    "reverse_kl":          r"Reverse KL",
    "improved_reverse_kl": r"Reverse KL$^{*}$",
    "forward_kl":          r"Forward KL",
    "improved_forward_kl": r"Forward KL$^{*}$",
    "jsd":                 r"JSD",
    "improved_jsd":        r"JSD$^{*}$",
}

# Per-family colour: KL family / fwd-KL family / JSD family.
# Each pair shares a hue; "improved" gets the solid / filled treatment.
PALETTE = {
    "reverse_kl":          "#1b9e77",
    "improved_reverse_kl": "#1b9e77",
    "forward_kl":          "#d95f02",
    "improved_forward_kl": "#d95f02",
    "jsd":                 "#7570b3",
    "improved_jsd":        "#7570b3",
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
            "hatch.linewidth": 1.4,
        }
    )


def _ceil_to(x: float, step: float) -> float:
    import math
    return math.ceil(x / step) * step


def _floor_to(x: float, step: float) -> float:
    import math
    return math.floor(x / step) * step


def _plot_curve(ax, steps, avgs, divergence: str, label: str | None = None) -> None:
    improved = IS_IMPROVED[divergence]
    ax.plot(
        steps,
        avgs,
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


def _draw_baseline_anchor(ax, baseline: float, label: str | None) -> None:
    ax.scatter(
        [0],
        [baseline],
        marker="D",
        s=55,
        facecolor="white",
        edgecolor="black",
        linewidth=1.3,
        zorder=5,
        label=label,
    )


def _add_hazard_title(fig, title_text: str, ax_for_extent):
    """Centered title flanked by thick solid bars spanning the axes width.

    The bar endpoints are computed from the rendered text's bounding box so
    the gap between bars and text stays constant regardless of title width.
    """
    from matplotlib.lines import Line2D

    title_y = 0.955

    # Use the plotting-area extents so the bars line up with the y-axis spine
    # on the left and the right edge of the plot on the right.
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

    # Render once so we can measure the actual text bbox.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_bbox_fig = txt.get_window_extent(renderer=renderer).transformed(
        fig.transFigure.inverted()
    )
    text_left = text_bbox_fig.x0
    text_right = text_bbox_fig.x1

    pad = 0.020  # figure-coord gap between text and bars
    span_left = (x_left, max(x_left, text_left - pad))
    span_right = (min(x_right, text_right + pad), x_right)

    bars = []
    for x0, x1 in (span_left, span_right):
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


def _add_break_marks(ax_top, ax_bot) -> None:
    """Diagonal slashes spanning the gap between the two y-axes."""
    d = 0.012
    kwargs = dict(transform=ax_top.transAxes, color="black", clip_on=False, linewidth=0.9)
    ax_top.plot((-d, +d), (-d * 6, +d * 6), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d * 6, +d * 6), **kwargs)
    kwargs = dict(transform=ax_bot.transAxes, color="black", clip_on=False, linewidth=0.9)
    ax_bot.plot((-d, +d), (1 - d * 6 * 4 / 1.2, 1 + d * 6 * 4 / 1.2), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d * 6 * 4 / 1.2, 1 + d * 6 * 4 / 1.2), **kwargs)


def _compute_average_family(df: pd.DataFrame) -> pd.DataFrame:
    """Macro-average across the three benchmark families, per (divergence, step)."""
    src = df[df["family"].isin(["aime24", "aime25", "hmmt25"])]
    grouped = (
        src.groupby(["run_type", "divergence", "step"], as_index=False)["avg"]
        .mean()
    )
    grouped["family"] = "average"
    grouped["samples"] = 0  # placeholder; not used for plotting
    return grouped


def main() -> None:
    _set_paper_style()
    df = pd.read_csv(CSV_PATH)
    df = pd.concat([df, _compute_average_family(df)], ignore_index=True)

    for family in FAMILIES:
        sub = df[df["family"] == family]
        baseline_rows = sub[sub["run_type"] == "baseline"]
        baseline = float(baseline_rows["avg"].iloc[0])
        baseline_n = int(baseline_rows["samples"].iloc[0])
        ckpt = sub[sub["run_type"] == "checkpoint_eval"]

        # Per-family zoomed top range; bottom is fixed 0–0.20 for collapsed curves.
        # Hug the data with a 2.5-pp grid so the upper limit isn't far above ymax.
        good = ckpt[ckpt["avg"] >= 0.20]
        ymax = max(good["avg"].max(), baseline)
        ymin = min(good["avg"].min(), baseline)
        y_top_lo = _floor_to(ymin - 0.015, 0.025)
        y_top_hi = _ceil_to(ymax + 0.010, 0.025)
        y_bot_hi = 0.20

        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(6.4, 4.6),
            gridspec_kw={"height_ratios": [4.0, 1.2], "hspace": 0.08},
        )
        fig.subplots_adjust(top=0.90)

        # Draw every curve on both axes; limits will hide the unused regions.
        for divergence in DIVERGENCE_ORDER:
            d = ckpt[ckpt["divergence"] == divergence].sort_values("step")
            if d.empty:
                continue
            steps = [0] + list(d["step"])
            avgs = [baseline] + list(d["avg"])
            label = PRETTY[divergence]
            _plot_curve(ax_top, steps, avgs, divergence, label=label)
            _plot_curve(ax_bot, steps, avgs, divergence, label=None)

        baseline_label = "Stage-0 baseline"
        _ = baseline_n  # kept for future use; not shown in legend
        _draw_baseline_anchor(ax_top, baseline, label=baseline_label)
        _draw_baseline_anchor(ax_bot, baseline, label=None)

        # Light reference line at the baseline value.
        ax_top.axhline(baseline, color="gray", linestyle=":", linewidth=0.8, zorder=1, alpha=0.6)

        ax_top.set_ylim(y_top_lo, y_top_hi)
        ax_bot.set_ylim(0, y_bot_hi)

        # Show ticks as fraction * 100 (e.g. 0.325 -> "32.5"); the "%" unit
        # lives in the y-axis label. Let AutoLocator pick the spacing.
        pct_num = FuncFormatter(lambda v, _pos: f"{v * 100:g}")
        for ax in (ax_top, ax_bot):
            ax.yaxis.set_major_formatter(pct_num)

        # Hide the spines between the two axes and add break marks.
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(labelbottom=False, bottom=False)
        ax_bot.xaxis.tick_bottom()
        _add_break_marks(ax_top, ax_bot)

        ax_bot.set_xticks([0, 20, 60, 100])
        ax_bot.set_xlabel("Steps", labelpad=4)
        ylabel = fig.supylabel("avg@$N$ accuracy (%)", fontsize=14)

        # Banner title with thick bars flanking the dataset name; pulled from
        # the top axes bbox so the bars align with the y-axis spine and right
        # edge of the plotting area.
        fig.canvas.draw()  # ensure get_position reflects subplots_adjust
        title_artists = _add_hazard_title(fig, FAMILY_TITLE[family].upper(), ax_top)

        # Build a 4-column, 2-row legend (4-3 layout):
        #   Row 1: Stage-0 baseline | Reverse KL  | Forward KL  | JSD
        #   Row 2: (blank)          | Reverse KL* | Forward KL* | JSD*
        # Matplotlib fills the legend column-by-column with ncol set, so list
        # entries in column-major order.
        from matplotlib.lines import Line2D

        handles, labels = ax_top.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        blank = Line2D([], [], linestyle="None", marker="None", label=" ")

        grid_order = [
            baseline_label,                # col1 row1
            None,                          # col1 row2 (blank)
            PRETTY["reverse_kl"],          # col2 row1
            PRETTY["improved_reverse_kl"], # col2 row2
            PRETTY["forward_kl"],          # col3 row1
            PRETTY["improved_forward_kl"], # col3 row2
            PRETTY["jsd"],                 # col4 row1
            PRETTY["improved_jsd"],        # col4 row2
        ]

        new_handles = []
        new_labels = []
        for key in grid_order:
            if key is None:
                new_handles.append(blank)
                new_labels.append(" ")
            else:
                new_handles.append(label_to_handle[key])
                new_labels.append(key)

        leg = fig.legend(
            new_handles,
            new_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.7),
            bbox_transform=ax_bot.transAxes,
            ncol=4,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.8,
            handletextpad=0.55,
            labelspacing=0.45,
        )

        png = OUT_DIR / f"opd_eval_{family}_avg_at_n.png"
        pdf = OUT_DIR / f"opd_eval_{family}_avg_at_n.pdf"
        extra = [leg, ylabel, *title_artists]
        fig.savefig(png, dpi=200, bbox_inches="tight", bbox_extra_artists=extra)
        fig.savefig(pdf, bbox_inches="tight", bbox_extra_artists=extra)
        plt.close(fig)
        print(f"wrote {png}\nwrote {pdf}")


if __name__ == "__main__":
    main()
