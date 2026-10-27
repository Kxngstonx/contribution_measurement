"""matplotlib figures for the demo (no pyplot; plain `Figure` objects that
embed in Tk through FigureCanvasTkAgg or save through Agg)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyBboxPatch

from . import config, texts

CLEAN = "#3b6fb6"
CONTAM = "#d9534f"
SERVER = "#6c757d"
FIT = "#999999"


def setup_fonts() -> str | None:
    """Register the Korean TrueType font with matplotlib; returns its family name."""

    path = config.korean_font_path()
    if path is None:
        rcParams["font.size"] = 12
        rcParams["axes.unicode_minus"] = False
        return None
    font_manager.fontManager.addfont(path)
    bold = config.korean_font_path(bold=True)
    if bold and bold != path:
        font_manager.fontManager.addfont(bold)
    name = FontProperties(fname=path).get_name()
    rcParams["font.family"] = name
    rcParams["font.size"] = 12
    rcParams["axes.unicode_minus"] = False
    return name


def _colors(contaminated) -> list:
    return [CONTAM if bool(c) else CLEAN for c in contaminated]


def _legend(ax):
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=CLEAN, label=texts.FIG_CLEAN),
                 Patch(color=CONTAM, label=texts.FIG_CONTAM)],
        loc="best", fontsize=11, frameon=False,
    )


def schematic_figure(figsize=(6.6, 4.2)) -> Figure:
    """Clients send updates to the server; the server returns a contribution
    score per client (one contaminated client drawn in red)."""

    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ys = [5.0, 3.7, 2.4, 1.1]
    flags = [False, False, True, False]
    for i, (y, bad) in enumerate(zip(ys, flags)):
        ax.add_patch(Circle((1.2, y), 0.45, color=CONTAM if bad else CLEAN))
        ax.text(1.2, y, str(i + 1), ha="center", va="center", color="white",
                fontsize=13, fontweight="bold")
        ax.annotate("", xy=(4.3, 3.05 + (y - 3.05) * 0.15), xytext=(1.75, y),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))
    ax.text(1.2, 0.3, texts.SCHEMATIC_CLIENT, ha="center", va="center", fontsize=12)
    ax.text(3.0, 5.6, texts.SCHEMATIC_UPDATE, ha="center", va="center", fontsize=11,
            color="#444444")

    ax.add_patch(FancyBboxPatch((4.4, 2.1), 2.2, 1.9, boxstyle="round,pad=0.05",
                                color=SERVER))
    ax.text(5.5, 3.05, texts.SCHEMATIC_SERVER, ha="center", va="center",
            color="white", fontsize=12)

    ax.annotate("", xy=(7.3, 3.05), xytext=(6.7, 3.05),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))
    heights = [1.6, 1.4, 0.3, 1.5]
    for i, (h, bad) in enumerate(zip(heights, flags)):
        x = 7.5 + i * 0.55
        ax.add_patch(FancyBboxPatch((x, 2.2), 0.4, h, boxstyle="square,pad=0",
                                    color=CONTAM if bad else CLEAN))
        ax.text(x + 0.2, 2.0, str(i + 1), ha="center", va="top", fontsize=10)
    ax.text(8.5, 4.3, texts.SCHEMATIC_SCORE, ha="center", va="center", fontsize=12)
    return fig


def bar_figure(results, method_title: str, figsize=(14.5, 4.2)) -> Figure:
    fig = Figure(figsize=figsize, dpi=100)
    axes = fig.subplots(1, len(results))
    if len(results) == 1:
        axes = [axes]
    for i, (ax, result) in enumerate(zip(axes, results)):
        n = result.n_players
        ax.bar(range(1, n + 1), result.scores, color=_colors(result.contaminated))
        ax.axhline(0.0, color="#333333", lw=0.6)
        ax.set_xticks(range(1, n + 1))
        ax.set_xlabel(texts.FIG_XLABEL_CLIENT)
        if i == 0:
            ax.set_ylabel(texts.FIG_YLABEL)
            _legend(ax)
        ax.set_title(f"{config.DATASET_TITLES[result.dataset]} {result.label}", fontsize=13)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(texts.FIG_BAR_TITLE.format(method=method_title), fontsize=15)
    fig.tight_layout()
    return fig


def scatter_figure(results, method_title: str, figsize=(14.5, 4.2)) -> Figure:
    fig = Figure(figsize=figsize, dpi=100)
    axes = fig.subplots(1, len(results))
    if len(results) == 1:
        axes = [axes]
    for i, (ax, result) in enumerate(zip(axes, results)):
        x = np.asarray(result.gt, dtype=np.float64)
        y = np.asarray(result.scores, dtype=np.float64)
        ax.scatter(x, y, c=_colors(result.contaminated), s=60, zorder=3)
        for j, (xj, yj) in enumerate(zip(x, y)):
            ax.annotate(str(j + 1), (xj, yj), textcoords="offset points",
                        xytext=(5, 4), fontsize=9)
        if len(x) >= 2 and x.std() > 0:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 20)
            ax.plot(grid, slope * grid + intercept, color=FIT, lw=0.9, ls="--", zorder=2)
        ax.set_xlabel(texts.FIG_XLABEL_GT)
        if i == 0:
            ax.set_ylabel(texts.FIG_YLABEL)
            _legend(ax)
        ax.set_title(
            f"{config.DATASET_TITLES[result.dataset]} {result.label}\n"
            + texts.FIG_PCC.format(pcc=result.pcc),
            fontsize=13,
        )
        ax.grid(alpha=0.3)
    fig.suptitle(texts.FIG_SCATTER_TITLE.format(method=method_title), fontsize=15)
    fig.tight_layout()
    return fig


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds >= 3600:
        return texts.TIME_HOURS.format(h=seconds / 3600)
    if seconds >= 60:
        return texts.TIME_MINUTES.format(m=seconds / 60)
    return texts.TIME_SECONDS.format(s=seconds)


def ratio_percent(gt_seconds: float, our_seconds: float) -> str:
    """`our_seconds` as a percentage of `gt_seconds`, e.g. "0.0059%" or "64.1%"."""

    pct = 100.0 * float(our_seconds) / float(gt_seconds)
    if pct >= 10:
        return f"{pct:.1f}%"
    if pct >= 0.1:
        return f"{pct:.2f}%"
    return f"{pct:.2g}%"


def time_figure(result, method_title: str, figsize=(14.5, 3.6)) -> Figure:
    """Log-scale bars: exact-Shapley wall clock vs the method's measurement time."""

    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    timing = result.timing or {}
    ours = timing.get("total_seconds")
    gt = result.gt_seconds
    if gt is None or ours is None:
        ax.axis("off")
        ax.text(0.5, 0.5, texts.FIG_TIME_NO_GT, ha="center", va="center", fontsize=13)
        fig.suptitle(texts.FIG_TIME_TITLE.format(method=method_title), fontsize=15)
        return fig

    label = method_title
    if timing.get("rounds"):
        seeds = timing.get("seeds") or 1
        label = texts.FIG_TIME_METHOD_ROUNDS.format(
            method=method_title, rounds=timing["rounds"],
            seeds=texts.FIG_TIME_SEEDS.format(seeds=seeds) if seeds > 1 else "")
    values = [float(gt), float(ours)]
    bars = ax.bar([texts.FIG_TIME_GT_LABEL, label], values, color=[SERVER, CLEAN], width=0.5)
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(values) * 40)
    ax.set_yticks([1, 60, 3600, 86400])
    ax.set_yticklabels(texts.TIME_TICKS)
    ax.set_ylabel(texts.FIG_TIME_YLABEL)
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.annotate(format_seconds(value), (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=12)
    ax.text(0.5, 0.9, texts.FIG_TIME_SUMMARY.format(
                gt=format_seconds(gt), ours=format_seconds(ours), ratio=ratio_percent(gt, ours)),
            transform=ax.transAxes, ha="center", va="top", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12)
    fig.suptitle(texts.FIG_TIME_TITLE.format(method=method_title), fontsize=15)
    fig.tight_layout()
    return fig


def save_figures(result, method: str, out_dir: Path) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = texts.METHOD_NAMES[method]
    paths = []
    for name, fig in (
        (f"{method}_bar.png", bar_figure([result], title)),
        (f"{method}_scatter.png", scatter_figure([result], title)),
        (f"{method}_time.png", time_figure(result, title)),
    ):
        path = out_dir / name
        fig.savefig(path, dpi=150)
        paths.append(path)
    schematic = out_dir / "schematic.png"
    schematic_figure().savefig(schematic, dpi=150)
    paths.append(schematic)
    return paths
