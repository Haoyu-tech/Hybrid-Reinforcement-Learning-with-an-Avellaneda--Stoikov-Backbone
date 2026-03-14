"""
figures.py  -  Publication-quality figure generation
=====================================================
Produces a 5-page A4-landscape PDF and individual per-figure PDFs.

Style targets:
  - IEEE / Elsevier / NeurIPS two-column layout (7.16" wide, 3.5" per panel)
  - Okabe-Ito colour palette (colour-blind safe)
  - Serif (Times-compatible) typeface, consistent sizing
  - No top/right spines; light dashed grid
  - 300 dpi raster + lossless PDF vector output
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from evaluate import (
    AgentMetrics,
    to_comprehensive_table_df,
    to_constraint_satisfaction_df,
    to_detailed_table_df,
    to_inventory_execution_table_df,
    to_numeric_df,
    to_per_metric_rank_df,
    to_pnl_distribution_df,
    to_returns_table_df,
    to_risk_adjusted_df,
    to_risk_table_df,
    to_scored_overview_table_df,
    to_statistical_significance_df,
    to_table_df,
)

# ---------------------------------------------------------------------------
# Global rcParams  -  journal-ready defaults
# ---------------------------------------------------------------------------
_FS_BASE   = 9      # body / tick labels
_FS_TITLE  = 10     # subplot titles
_FS_LABEL  = 9      # axis labels
_FS_TICK   = 8      # tick labels
_FS_LEGEND = 8      # legend text
_FS_CAP    = 7.5    # figure captions
_LW        = 1.6    # default line width
_DPI       = 300

plt.rcParams.update(
    {
        # ----- typeface -----
        "font.family":           "serif",
        "font.serif":            ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":      "stix",
        "font.size":             _FS_BASE,
        "axes.titlesize":        _FS_TITLE,
        "axes.labelsize":        _FS_LABEL,
        "xtick.labelsize":       _FS_TICK,
        "ytick.labelsize":       _FS_TICK,
        "legend.fontsize":       _FS_LEGEND,
        "legend.title_fontsize": _FS_LEGEND,
        # ----- lines -----
        "lines.linewidth":       _LW,
        "lines.antialiased":     True,
        # ----- axes -----
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.linewidth":        0.8,
        "axes.grid":             True,
        "axes.grid.which":       "major",
        "grid.linestyle":        "--",
        "grid.linewidth":        0.4,
        "grid.alpha":            0.45,
        "grid.color":            "#AAAAAA",
        # ----- patches -----
        "patch.linewidth":       0.5,
        # ----- figure / saving -----
        "figure.dpi":            _DPI,
        "savefig.dpi":           _DPI,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.05,
        # ----- legend -----
        "legend.frameon":        True,
        "legend.framealpha":     0.90,
        "legend.edgecolor":      "#CCCCCC",
        "legend.borderpad":      0.4,
        "legend.handlelength":   1.8,
    }
)

# ---------------------------------------------------------------------------
# Colour palette  -  Okabe & Ito (2002), colour-blind safe
# ---------------------------------------------------------------------------
_OI = [
    "#0072B2",   # 0  blue
    "#E69F00",   # 1  orange
    "#009E73",   # 2  green
    "#D55E00",   # 3  vermilion
    "#CC79A7",   # 4  reddish-purple
    "#56B4E9",   # 5  sky blue
    "#F0E442",   # 6  yellow  (avoid on white)
    "#000000",   # 7  black
]

# method key -> (colour, linestyle, marker, bar-hatch)
_STYLE: Dict[str, Tuple[str, str, str, str]] = {
    "static_as":      (_OI[0], "-",  "o",  ""),
    "constrained_as": (_OI[1], "--", "s",  "//"),
    "model_free_rl":  (_OI[2], "-.", "^",  ""),
    "hybrid_tuning":  (_OI[3], ":",  "D",  ".."),
    "hybrid_as_rl":   (_OI[4], "-",  "*",  "xx"),
}
_DEFAULT_STYLE = (_OI[5], "-", "o", "")

_LONG_NAME = {
    "static_as":      "Static AS",
    "constrained_as": "Constrained AS (Gu\u00e9ant)",
    "model_free_rl":  "Model-Free PPO",
    "hybrid_tuning":  "Hybrid-Tuning PPO",
    "hybrid_as_rl":   "Hybrid AS+RL (ours)",
}
_SHORT_NAME = {
    "static_as":      "Static AS",
    "constrained_as": "Constr. AS",
    "model_free_rl":  "Model-Free PPO",
    "hybrid_tuning":  "Hybrid-Tuning",
    "hybrid_as_rl":   "Hybrid AS+RL",
}

_HEATMAP_COLS = {
    "mean_pnl":      "Mean\nPnL",
    "sharpe":        "Sharpe",
    "sortino":       "Sortino",
    "max_drawdown":  "Max\nDD",
    "cvar_5":        "CVaR\n5%",
    "mean_abs_inv":  "Mean\n|Inv|",
    "inv_tail_mass": "Inv\nTail",
    "time_at_limit": "Time@\nLimit",
    "fill_rate":     "Fill\nRate",
}
_RISK_COLS = {"max_drawdown", "cvar_5", "mean_abs_inv", "inv_tail_mass", "time_at_limit"}

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _s(key: str) -> Tuple[str, str, str, str]:
    return _STYLE.get(key, _DEFAULT_STYLE)


def _lbl(key: str) -> str:
    return _LONG_NAME.get(key, key)


def _slbl(key: str) -> str:
    return _SHORT_NAME.get(key, _lbl(key))


def _methods(results: Dict[str, AgentMetrics]) -> List[str]:
    return list(results.keys())


def _arr(results: Dict[str, AgentMetrics], attr: str) -> np.ndarray:
    return np.asarray([float(getattr(results[k], attr, 0.0)) for k in _methods(results)])


def _mean_std_curve(
    episodes: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(steps, mean, std) aligned to shortest episode."""
    arrs = [np.asarray(ep, dtype=float).ravel() for ep in episodes if len(ep) > 0]
    if not arrs:
        return np.zeros(1), np.zeros(1), np.zeros(1)
    m = min(len(a) for a in arrs)
    stack = np.vstack([a[:m] for a in arrs])
    return np.arange(m), stack.mean(0), stack.std(0)


def _clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _axis_pad(ax, x, y, xp=0.10, yp=0.10):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size:
        dx = max(x.max() - x.min(), 1e-6)
        ax.set_xlim(x.min() - dx * xp, x.max() + dx * xp)
    if y.size:
        dy = max(y.max() - y.min(), 1e-6)
        ax.set_ylim(y.min() - dy * yp, y.max() + dy * yp)


def _add_caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.005, text, ha="center", va="bottom",
             fontsize=_FS_CAP, style="italic", color="#444444")


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _pres_df(results):
    df = to_table_df(results).copy()
    return df[[
        "Method", "Mean PnL", "Delta PnL(vs Static)",
        "Sharpe", "Delta Sharpe(vs Static)",
        "Max DD", "CVaR 5%", "Mean |Inv|", "Fill Rate",
    ]]


def _style_table(tbl, n_rows: int, n_cols: int) -> None:
    HDR  = "#1A3A5C"
    EVEN = "#EBF1FA"
    ODD  = "#FFFFFF"
    EDGE = "#C8D8EC"
    for c in range(n_cols):
        cell = tbl[(0, c)]
        cell.set_facecolor(HDR)
        cell.set_text_props(color="white", weight="bold", fontsize=8.5)
        cell.set_edgecolor(HDR)
    for r in range(1, n_rows + 1):
        bg = EVEN if r % 2 == 0 else ODD
        for c in range(n_cols):
            cell = tbl[(r, c)]
            cell.set_facecolor(bg)
            cell.set_edgecolor(EDGE)
            cell.set_text_props(fontsize=8)
            if c == 0:
                cell.set_text_props(weight="bold", fontsize=8)
    # highlight top row (best method after sorting)
    if n_rows >= 1:
        for c in range(n_cols):
            tbl[(1, c)].set_facecolor("#D4E8C2")


def _export_tables(results, out_dir):
    pairs = [
        (_pres_df(results),                        "table_presentation"),
        (to_scored_overview_table_df(results),      "table_scored_overview"),
        (to_detailed_table_df(results),             "table_detailed"),
        (to_comprehensive_table_df(results),        "table_comprehensive"),
        (to_returns_table_df(results),              "table_returns"),
        (to_risk_table_df(results),                 "table_risk"),
        (to_inventory_execution_table_df(results),  "table_inventory_execution"),
        # --- new tables ---
        (to_statistical_significance_df(results),   "table_statistical_significance"),
        (to_risk_adjusted_df(results),              "table_risk_adjusted"),
        (to_pnl_distribution_df(results),           "table_pnl_distribution"),
        (to_per_metric_rank_df(results),            "table_per_metric_rank"),
        (to_constraint_satisfaction_df(results),    "table_constraint_satisfaction"),
    ]
    for df, stem in pairs:
        df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False, encoding="utf-8-sig")
        try:
            df.to_latex(os.path.join(out_dir, f"{stem}.tex"),
                        index=False, float_format="%.4f", escape=True)
        except Exception:
            pass


# ===========================================================================
# Individual figure functions
# ===========================================================================

# --- Figure sizes (inches) consistent with journal column widths -------------
# single column  ~ 3.54"  |  double column ~ 7.16"
_W1 = 3.54   # single column width
_W2 = 7.16   # double column width
_H1 = 2.80   # compact height (half-page row)
_H2 = 3.60   # standard height
_H3 = 4.60   # taller panels


def _fig_pnl_trajectory(results, out: str) -> None:
    """Mean +/- 1 s.d. PnL trajectory."""
    fig, ax = plt.subplots(figsize=(_W2, _H2))
    _clean(ax)
    for key in _methods(results):
        c, ls, _, _ = _s(key)
        t, mu, sg = _mean_std_curve(results[key].pnl_episodes)
        ax.plot(t, mu, color=c, ls=ls, lw=_LW, label=_lbl(key))
        ax.fill_between(t, mu - sg, mu + sg, color=c, alpha=0.13)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title("Mean Episode PnL Trajectory (\u00b11 s.d.)", fontweight="bold")
    ax.legend(ncol=2, loc="upper left", fontsize=_FS_LEGEND)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _fig_inventory_trajectory(results, out: str) -> None:
    """Mean +/- 1 s.d. inventory trajectory."""
    fig, ax = plt.subplots(figsize=(_W2, _H2))
    _clean(ax)
    for key in _methods(results):
        c, ls, _, _ = _s(key)
        t, mu, sg = _mean_std_curve(results[key].inv_episodes)
        ax.plot(t, mu, color=c, ls=ls, lw=_LW, label=_lbl(key))
        ax.fill_between(t, mu - sg, mu + sg, color=c, alpha=0.13)
    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Inventory (shares)")
    ax.set_title("Mean Episode Inventory Trajectory (\u00b11 s.d.)", fontweight="bold")
    ax.legend(ncol=2, loc="upper right", fontsize=_FS_LEGEND)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _fig_bar_metrics(results, out: str) -> None:
    """2x2 horizontal bar panel: PnL / Sharpe / Risk / Inventory."""
    methods = _methods(results)
    colours = [_s(k)[0] for k in methods]
    hatches = [_s(k)[3] for k in methods]
    labels  = [_slbl(k) for k in methods]
    y = np.arange(len(methods))
    h = 0.55

    fig, axes = plt.subplots(2, 2, figsize=(_W2, _H3 + 0.8))
    fig.subplots_adjust(hspace=0.45, wspace=0.22, bottom=0.10,
                        left=0.14, right=0.97, top=0.93)
    fig.suptitle("Aggregate Performance Metrics", fontsize=_FS_TITLE + 1,
                 fontweight="bold")

    # ---- Mean PnL ----
    ax = axes[0, 0]; _clean(ax)
    vals = _arr(results, "mean_pnl")
    errs = _arr(results, "std_pnl")
    bars = ax.barh(y, vals, h, color=colours, edgecolor="white", linewidth=0.5,
                   xerr=errs, capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#555"})
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("PnL")
    ax.set_title("Mean Terminal PnL (\u00b11 s.d.)", fontweight="bold", fontsize=_FS_TITLE)
    ax.axvline(0, color="#888888", lw=0.7, ls=":")
    ax.invert_yaxis()

    # ---- Sharpe ----
    ax = axes[0, 1]; _clean(ax)
    vals = _arr(results, "sharpe")
    bars = ax.barh(y, vals, h, color=colours, edgecolor="white", linewidth=0.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Annualised Sharpe Ratio", fontweight="bold", fontsize=_FS_TITLE)
    ax.axvline(0, color="#888888", lw=0.7, ls=":")
    ax.invert_yaxis()

    # ---- Tail Risk ----
    ax = axes[1, 0]; _clean(ax)
    dd   = _arr(results, "max_drawdown")
    cvar = _arr(results, "cvar_5")
    w2   = 0.32
    ax.barh(y - w2 / 2, dd,   w2, color=_OI[0], label="Max Drawdown",
            edgecolor="white", linewidth=0.5)
    ax.barh(y + w2 / 2, cvar, w2, color=_OI[3], label="CVaR 5%",
            edgecolor="white", linewidth=0.5, hatch="..")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Magnitude")
    ax.set_title("Tail Risk Metrics", fontweight="bold", fontsize=_FS_TITLE)
    ax.legend(ncol=1, fontsize=_FS_LEGEND - 0.5, loc="lower right")
    ax.invert_yaxis()

    # ---- Mean |Inv| ----
    ax = axes[1, 1]; _clean(ax)
    vals = _arr(results, "mean_abs_inv")
    errs = _arr(results, "std_inv")
    bars = ax.barh(y, vals, h, color=colours, edgecolor="white", linewidth=0.5,
                   xerr=errs, capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#555"})
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("Mean |Inventory| (shares)")
    ax.set_title("Mean Absolute Inventory (\u00b11 s.d.)", fontweight="bold", fontsize=_FS_TITLE)
    ax.invert_yaxis()

    fig.savefig(out)
    plt.close(fig)


def _fig_risk_return(results, out: str) -> None:
    """Risk-return scatter; marker area proportional to Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(_W1 + 0.5, _W1 + 0.2))
    _clean(ax)

    methods = _methods(results)
    pnl    = _arr(results, "mean_pnl")
    risk   = _arr(results, "std_pnl")
    sharpe = _arr(results, "sharpe")
    sizes  = np.clip(60 + 28 * np.maximum(sharpe, 0), 40, 280)

    offsets = {
        "static_as":       (16, 14),
        "constrained_as":  (-68, 10),
        "model_free_rl":   (12, 8),
        "hybrid_tuning":   (-68, -16),
        "hybrid_as_rl":    (12, 10),
    }
    for i, key in enumerate(methods):
        c = _s(key)[0]; mk = _s(key)[2]
        dx, dy = offsets.get(key, (8, 4))
        ax.scatter(risk[i], pnl[i], s=sizes[i], color=c, marker=mk,
                   zorder=4, linewidths=0.8, edgecolors="white", label=_lbl(key))
        ax.annotate(
            _lbl(key), (risk[i], pnl[i]),
            fontsize=6.5, xytext=(dx, dy), textcoords="offset points",
            color=c,
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": c, "alpha": 0.65},
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
        )

    _axis_pad(ax, risk, pnl, xp=0.14, yp=0.12)
    ax.set_xlabel("Std PnL (Risk)")
    ax.set_ylabel("Mean Terminal PnL (Return)")
    ax.set_title("Risk\u2013Return Map\n(marker area \u221d Sharpe)", fontweight="bold")
    fig.tight_layout(pad=0.7)
    fig.savefig(out)
    plt.close(fig)


def _fig_violin(results, out: str) -> None:
    """Violin + IQR box overlay for terminal PnL."""
    methods = _methods(results)
    colours = [_s(k)[0] for k in methods]
    labels  = [_slbl(k) for k in methods]

    term_data = []
    for key in methods:
        eps = results[key].pnl_episodes
        td  = [float(np.asarray(ep).ravel()[-1]) for ep in eps if len(ep) > 0]
        term_data.append(td if td else [0.0])

    fig, ax = plt.subplots(figsize=(_W2, _H2))
    _clean(ax)

    parts = ax.violinplot(term_data, positions=range(len(methods)),
                          showmeans=False, showmedians=False,
                          showextrema=False, widths=0.70)
    for body, c in zip(parts["bodies"], colours):
        body.set_facecolor(c); body.set_edgecolor("white"); body.set_alpha(0.55)

    bp = ax.boxplot(
        term_data, positions=range(len(methods)), widths=0.20,
        patch_artist=True, showfliers=False,
        medianprops={"color": "#111111", "linewidth": 2.0},
        whiskerprops={"linewidth": 0.9, "linestyle": "--"},
        capprops={"linewidth": 0.9},
        boxprops={"linewidth": 0.5},
    )
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.80)

    ax.axhline(0, color="#888888", lw=0.8, ls=":")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Terminal PnL")
    ax.set_title("Terminal PnL Distribution by Method", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _fig_inv_hist(results, out: str) -> None:
    """Overlaid step-filled inventory density histograms."""
    fig, ax = plt.subplots(figsize=(_W2, _H2))
    _clean(ax)
    for key in _methods(results):
        c, ls, _, _ = _s(key)
        arrs = [np.asarray(ep).ravel() for ep in results[key].inv_episodes if len(ep) > 0]
        if not arrs:
            continue
        inv = np.concatenate(arrs)
        ax.hist(inv, bins=40, density=True, histtype="stepfilled",
                alpha=0.20, color=c, linewidth=0.4)
        ax.hist(inv, bins=40, density=True, histtype="step",
                alpha=0.90, color=c, linewidth=1.0, label=_lbl(key))
    ax.axvline(0, color="#444444", lw=0.8, ls=":")
    ax.set_xlabel("Inventory (shares)")
    ax.set_ylabel("Density")
    ax.set_title("Inventory Distribution Across Episodes", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, fontsize=_FS_LEGEND)
    fig.tight_layout(pad=0.5)
    fig.savefig(out)
    plt.close(fig)


def _fig_heatmap(results, out: str) -> None:
    """Normalised performance heatmap with cell annotations."""
    dfn     = to_numeric_df(results).copy()
    metrics = list(_HEATMAP_COLS.keys())

    mat   = dfn[metrics].to_numpy(float)
    mn    = mat.min(0, keepdims=True)
    mx    = mat.max(0, keepdims=True)
    denom = np.where((mx - mn) < 1e-12, 1.0, mx - mn)
    norm  = (mat - mn) / denom
    for j, name in enumerate(metrics):
        if name in _RISK_COLS:
            norm[:, j] = 1.0 - norm[:, j]

    row_labels = dfn["label"].tolist()
    col_labels = [_HEATMAP_COLS[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(_W2 + 0.4, max(3.2, 0.78 * len(row_labels) + 1.5)))
    _clean(ax)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    im = ax.imshow(norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            raw = float(mat[r, c]); sc = float(norm[r, c])
            tc  = "white" if sc < 0.23 or sc > 0.80 else "#111111"
            ax.text(c, r, f"{raw:.3f}", ha="center", va="center",
                    fontsize=6.8, color=tc)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=_FS_TICK)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=_FS_TICK + 0.5)
    ax.tick_params(length=0)
    ax.set_title("Normalised Performance Heatmap (green = better)", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.03)
    cbar.ax.set_ylabel("Relative Score", fontsize=_FS_LEGEND, labelpad=4)
    cbar.ax.tick_params(labelsize=_FS_TICK - 0.5)

    fig.tight_layout(pad=0.6)
    fig.savefig(out)
    plt.close(fig)


def _fig_pnl_quantiles(results, out: str) -> None:
    """Horizontal quantile fan: Q5-Q25-median-Q75-Q95."""
    methods = _methods(results)
    labels  = [_lbl(k) for k in methods]
    q05 = _arr(results, "pnl_q05")
    q25 = _arr(results, "pnl_q25")
    med = _arr(results, "pnl_median")
    q75 = _arr(results, "pnl_q75")
    q95 = _arr(results, "pnl_q95")
    y   = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(_W2, max(2.8, 0.62 * len(methods) + 1.4)))
    _clean(ax)

    for i, key in enumerate(methods):
        c = _s(key)[0]
        ax.plot([q05[i], q95[i]], [y[i], y[i]], color=c, lw=1.6, alpha=0.45)
        ax.plot([q25[i], q75[i]], [y[i], y[i]], color=c, lw=4.5,
                alpha=0.72, solid_capstyle="butt")
        ax.scatter(med[i], y[i], s=52, color=c, zorder=5,
                   linewidths=0.8, edgecolors="white")

    ax.axvline(0, color="#888888", lw=0.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Terminal PnL")
    ax.set_title("PnL Quantile Fan  (Q5 \u2013 Q25\u2501Q75 \u2013 Q95, dot = median)",
                 fontweight="bold")
    proxies = [
        mpatches.Patch(color="#888888", alpha=0.45, label="Q5\u2013Q95 range"),
        mpatches.Patch(color="#888888", alpha=0.80, label="IQR (Q25\u2013Q75)"),
    ]
    ax.legend(handles=proxies, fontsize=_FS_LEGEND, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _fig_table(results, out: str) -> None:
    """Summary table as a figure."""
    df  = _pres_df(results)
    fig, ax = plt.subplots(figsize=(_W2 + 10.3, max(3.6, 0.52 * (len(df) + 1))))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values, colLabels=df.columns,
        loc="center", cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.65)
    try:
        tbl.auto_set_column_width(list(range(len(df.columns))))
    except Exception:
        pass
    _style_table(tbl, len(df), len(df.columns))
    fig.tight_layout(pad=0.2)
    fig.savefig(out)
    plt.close(fig)


# ===========================================================================
# Multi-page A4-landscape PDF  (5 pages)
# ===========================================================================

def _pdf_p1_table(results, pdf: PdfPages) -> None:
    """Page 1: title block + summary table."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    # title / subtitle
    fig.text(0.03, 0.97,
             "Hybrid Market-Making \u2014 Performance Summary",
             fontsize=16, fontweight="bold", va="top", color="#1A3A5C")
    fig.text(0.03, 0.93,
             "Comparison of Avellaneda\u2013Stoikov baselines, "
             "Model-Free PPO, and Hybrid AS+RL (proposed method).",
             fontsize=9, va="top", color="#555555")

    df  = _pres_df(results)
    ax  = fig.add_axes([0.02, 0.07, 0.96, 0.80])
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values, colLabels=df.columns,
        loc="upper center", cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.70)
    try:
        tbl.auto_set_column_width(list(range(len(df.columns))))
    except Exception:
        pass
    _style_table(tbl, len(df), len(df.columns))

    _add_caption(fig,
        "Table 1. Composite score = weighted z-score across PnL, Sharpe, "
        "drawdown, inventory risk, and fill rate. "
        "Green row = top-ranked method.")
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _pdf_p2_trajectories(results, pdf: PdfPages) -> None:
    """Page 2: PnL + inventory trajectory curves."""
    fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
    fig.subplots_adjust(hspace=0.40, top=0.93, bottom=0.10, left=0.07, right=0.97)

    for key in _methods(results):
        c, ls, _, _ = _s(key)
        t_p, mu_p, sg_p = _mean_std_curve(results[key].pnl_episodes)
        t_i, mu_i, sg_i = _mean_std_curve(results[key].inv_episodes)
        axes[0].plot(t_p, mu_p, color=c, ls=ls, lw=_LW, label=_lbl(key))
        axes[0].fill_between(t_p, mu_p - sg_p, mu_p + sg_p, color=c, alpha=0.12)
        axes[1].plot(t_i, mu_i, color=c, ls=ls, lw=_LW, label=_lbl(key))
        axes[1].fill_between(t_i, mu_i - sg_i, mu_i + sg_i, color=c, alpha=0.12)

    spec = [
        (axes[0], "Mean Episode PnL Trajectory (\u00b11 s.d.)",        "Cumulative PnL"),
        (axes[1], "Mean Episode Inventory Trajectory (\u00b11 s.d.)", "Inventory (shares)"),
    ]
    for ax, title, ylabel in spec:
        _clean(ax)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time Step")
        ax.set_ylabel(ylabel)
        ax.legend(ncol=3, fontsize=_FS_LEGEND)

    axes[1].axhline(0, color="#888888", lw=0.8, ls=":")
    fig.suptitle("Episode Trajectory Analysis", fontsize=13, fontweight="bold", y=0.98)
    _add_caption(fig,
        "Figure 1. Shaded bands denote \u00b11 s.d. across evaluation episodes. "
        "Inventory converges faster for the constrained and hybrid methods.")
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _pdf_p3_bar_metrics(results, pdf: PdfPages) -> None:
    """Page 3: 2x2 horizontal bar panel."""
    methods = _methods(results)
    colours = [_s(k)[0] for k in methods]
    hatches = [_s(k)[3] for k in methods]
    labels  = [_lbl(k) for k in methods]
    y = np.arange(len(methods))
    h = 0.55

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(hspace=0.48, wspace=0.24, top=0.92, bottom=0.10,
                        left=0.14, right=0.97)
    fig.suptitle("Aggregate Performance Metrics", fontsize=13,
                 fontweight="bold", y=0.97)

    # Mean PnL
    ax = axes[0, 0]; _clean(ax)
    vals = _arr(results, "mean_pnl"); errs = _arr(results, "std_pnl")
    bars = ax.barh(y, vals, h, color=colours, edgecolor="white", linewidth=0.5,
                   xerr=errs, capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#555"})
    [b.set_hatch(hatch) for b, hatch in zip(bars, hatches)]
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("PnL")
    ax.set_title("Mean Terminal PnL (\u00b11 s.d.)", fontweight="bold")
    ax.axvline(0, color="#888", lw=0.7, ls=":")
    ax.invert_yaxis()

    # Sharpe
    ax = axes[0, 1]; _clean(ax)
    vals = _arr(results, "sharpe")
    bars = ax.barh(y, vals, h, color=colours, edgecolor="white", linewidth=0.5)
    [b.set_hatch(hatch) for b, hatch in zip(bars, hatches)]
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Annualised Sharpe Ratio", fontweight="bold")
    ax.axvline(0, color="#888", lw=0.7, ls=":")
    ax.invert_yaxis()

    # Tail Risk
    ax = axes[1, 0]; _clean(ax)
    dd = _arr(results, "max_drawdown"); cvar = _arr(results, "cvar_5")
    w2 = 0.32
    ax.barh(y - w2/2, dd,   w2, color=_OI[0], label="Max Drawdown",
            edgecolor="white", linewidth=0.5)
    ax.barh(y + w2/2, cvar, w2, color=_OI[3], label="CVaR 5\u0025",
            edgecolor="white", linewidth=0.5, hatch="..")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Magnitude")
    ax.set_title("Tail Risk Metrics", fontweight="bold")
    ax.legend(ncol=1, fontsize=_FS_LEGEND, loc="lower right")
    ax.invert_yaxis()

    # Mean |Inv|
    ax = axes[1, 1]; _clean(ax)
    vals = _arr(results, "mean_abs_inv"); errs = _arr(results, "std_inv")
    bars = ax.barh(y, vals, h, color=colours, edgecolor="white", linewidth=0.5,
                   xerr=errs, capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#555"})
    [b.set_hatch(hatch) for b, hatch in zip(bars, hatches)]
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("Mean |Inventory| (shares)")
    ax.set_title("Mean Absolute Inventory (\u00b11 s.d.)", fontweight="bold")
    ax.invert_yaxis()

    _add_caption(fig,
        "Figure 2. Error bars denote \u00b11 s.d. across episodes. "
        "Hatches distinguish methods; colours follow the Okabe\u2013Ito palette.")
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _pdf_p4_distributions(results, pdf: PdfPages) -> None:
    """Page 4: risk-return scatter | violin | quantile fan."""
    methods = _methods(results)
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.07, right=0.97,
                        hspace=0.44, wspace=0.34)
    gs = GridSpec(2, 2, figure=fig)

    # --- Risk-return scatter (left column, full height) ---
    ax_rr = fig.add_subplot(gs[:, 0]); _clean(ax_rr)
    pnl    = _arr(results, "mean_pnl")
    risk   = _arr(results, "std_pnl")
    sharpe = _arr(results, "sharpe")
    sizes  = np.clip(70 + 30 * np.maximum(sharpe, 0), 50, 300)
    offsets = {
        "static_as":       (16, 14),
        "constrained_as":  (-68, 10),
        "model_free_rl":   (12, 8),
        "hybrid_tuning":   (-68, -16),
        "hybrid_as_rl":    (12, 10),
    }
    for i, key in enumerate(methods):
        c = _s(key)[0]; mk = _s(key)[2]
        dx, dy = offsets.get(key, (8, 4))
        ax_rr.scatter(risk[i], pnl[i], s=sizes[i], color=c, marker=mk,
                      zorder=4, linewidths=0.8, edgecolors="white")
        ax_rr.annotate(
            _lbl(key), (risk[i], pnl[i]),
            fontsize=6.5, xytext=(dx, dy), textcoords="offset points",
            color=c,
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": c, "alpha": 0.6},
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.82},
        )
    _axis_pad(ax_rr, risk, pnl, xp=0.14, yp=0.12)
    ax_rr.set_xlabel("Std PnL (Risk)")
    ax_rr.set_ylabel("Mean Terminal PnL (Return)")
    ax_rr.set_title("Risk\u2013Return Map\n(marker area \u221d Sharpe)", fontweight="bold")

    # --- Violin + IQR (top-right) ---
    ax_v = fig.add_subplot(gs[0, 1]); _clean(ax_v)
    term_data = []
    for key in methods:
        eps = results[key].pnl_episodes
        td  = [float(np.asarray(ep).ravel()[-1]) for ep in eps if len(ep) > 0]
        term_data.append(td if td else [0.0])
    colours = [_s(k)[0] for k in methods]
    parts = ax_v.violinplot(term_data, positions=range(len(methods)),
                             showmeans=False, showmedians=False,
                             showextrema=False, widths=0.70)
    for body, c in zip(parts["bodies"], colours):
        body.set_facecolor(c); body.set_edgecolor("white"); body.set_alpha(0.55)
    bp = ax_v.boxplot(
        term_data, positions=range(len(methods)), widths=0.20,
        patch_artist=True, showfliers=False,
        medianprops={"color": "#111111", "linewidth": 2.0},
        whiskerprops={"linewidth": 0.9, "linestyle": "--"},
        capprops={"linewidth": 0.9}, boxprops={"linewidth": 0.5},
    )
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c); patch.set_alpha(0.80)
    ax_v.axhline(0, color="#888", lw=0.8, ls=":")
    ax_v.set_xticks(range(len(methods)))
    ax_v.set_xticklabels([_slbl(k) for k in methods], rotation=14,
                          ha="right", fontsize=_FS_TICK - 0.5)
    ax_v.set_ylabel("Terminal PnL")
    ax_v.set_title("Terminal PnL Distribution", fontweight="bold")

    # --- Quantile fan (bottom-right) ---
    ax_q = fig.add_subplot(gs[1, 1]); _clean(ax_q)
    q05 = _arr(results, "pnl_q05"); q25 = _arr(results, "pnl_q25")
    med = _arr(results, "pnl_median"); q75 = _arr(results, "pnl_q75")
    q95 = _arr(results, "pnl_q95")
    yy  = np.arange(len(methods))
    for i, key in enumerate(methods):
        c = _s(key)[0]
        ax_q.plot([q05[i], q95[i]], [yy[i], yy[i]], color=c, lw=1.6, alpha=0.45)
        ax_q.plot([q25[i], q75[i]], [yy[i], yy[i]], color=c, lw=4.5,
                  alpha=0.72, solid_capstyle="butt")
        ax_q.scatter(med[i], yy[i], s=46, color=c, zorder=5,
                     edgecolors="white", linewidths=0.8)
    ax_q.axvline(0, color="#888", lw=0.8, ls=":")
    ax_q.set_yticks(yy)
    ax_q.set_yticklabels([_slbl(k) for k in methods], fontsize=_FS_TICK)
    ax_q.set_xlabel("Terminal PnL")
    ax_q.set_title("PnL Quantile Fan  (Q5 \u2013 IQR \u2013 Q95)", fontweight="bold")

    fig.suptitle("Return Distribution Analysis", fontsize=13, fontweight="bold", y=0.97)
    _add_caption(fig,
        "Figure 3. Left: risk\u2013return frontier; marker area \u221d Sharpe ratio. "
        "Top-right: violin + IQR box overlay. "
        "Bottom-right: quantile fan (thick bar = IQR, thin lines = Q5\u2013Q95, dot = median).")
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _pdf_p5_heatmap_inv(results, pdf: PdfPages) -> None:
    """Page 5: heatmap + inventory histogram."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.97,
                        hspace=0.42, wspace=0.32)
    gs = GridSpec(1, 2, width_ratios=[2.3, 1.0], figure=fig)

    # --- Heatmap ---
    ax_h = fig.add_subplot(gs[0, 0]); _clean(ax_h)
    ax_h.spines["bottom"].set_visible(False); ax_h.spines["left"].set_visible(False)

    dfn     = to_numeric_df(results).copy()
    metrics = list(_HEATMAP_COLS.keys())
    mat     = dfn[metrics].to_numpy(float)
    mn      = mat.min(0, keepdims=True); mx = mat.max(0, keepdims=True)
    denom   = np.where((mx - mn) < 1e-12, 1.0, mx - mn)
    norm    = (mat - mn) / denom
    for j, name in enumerate(metrics):
        if name in _RISK_COLS:
            norm[:, j] = 1.0 - norm[:, j]

    row_labels = dfn["label"].tolist()
    col_labels = [_HEATMAP_COLS[m] for m in metrics]

    im = ax_h.imshow(norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            raw = float(mat[r, c]); sc = float(norm[r, c])
            tc  = "white" if sc < 0.23 or sc > 0.80 else "#111111"
            ax_h.text(c, r, f"{raw:.3f}", ha="center", va="center",
                      fontsize=6.5, color=tc)
    ax_h.set_xticks(np.arange(len(col_labels)))
    ax_h.set_xticklabels(col_labels, rotation=0, ha="center", fontsize=_FS_TICK)
    ax_h.set_yticks(np.arange(len(row_labels)))
    ax_h.set_yticklabels(row_labels, fontsize=_FS_TICK + 0.5)
    ax_h.tick_params(length=0)
    ax_h.set_title("Normalised Performance Heatmap (green = better)", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.022, pad=0.03)
    cbar.ax.set_ylabel("Relative Score", fontsize=_FS_LEGEND, labelpad=4)
    cbar.ax.tick_params(labelsize=_FS_TICK - 0.5)

    # --- Inventory histogram ---
    ax_i = fig.add_subplot(gs[0, 1]); _clean(ax_i)
    for key in _methods(results):
        c = _s(key)[0]
        arrs = [np.asarray(ep).ravel() for ep in results[key].inv_episodes if len(ep) > 0]
        if not arrs:
            continue
        inv = np.concatenate(arrs)
        ax_i.hist(inv, bins=40, density=True, histtype="stepfilled",
                  alpha=0.20, color=c, linewidth=0.4)
        ax_i.hist(inv, bins=40, density=True, histtype="step",
                  alpha=0.90, color=c, linewidth=1.0, label=_lbl(key))
    ax_i.axvline(0, color="#444", lw=0.8, ls=":")
    ax_i.set_xlabel("Inventory (shares)")
    ax_i.set_ylabel("Density")
    ax_i.set_title("Inventory\nDistribution", fontweight="bold")
    ax_i.legend(fontsize=6.0, ncol=1, loc="upper center",
                bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Diagnostics: Metric Heatmap & Inventory Profile",
                 fontsize=13, fontweight="bold", y=0.97)
    _add_caption(fig,
        "Figure 4. Left: each cell shows the raw metric value; "
        "colour normalised column-wise (green = best in column). "
        "Right: overlaid inventory density histograms across all evaluation episodes.")
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


# ===========================================================================
# Additional PDF pages for new tables
# ===========================================================================

def _render_df_as_table(
    fig: plt.Figure,
    ax: plt.Axes,
    df: pd.DataFrame,
    fontsize: float = 8.0,
    row_height: float = 1.55,
) -> None:
    """Render a DataFrame as a styled matplotlib table on a given axes."""
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="upper center",
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, row_height)
    try:
        tbl.auto_set_column_width(list(range(len(df.columns))))
    except Exception:
        pass
    _style_table(tbl, len(df), len(df.columns))


def _pdf_p6_new_tables_1(results, pdf: PdfPages) -> None:
    """
    Page 6: Statistical significance + Risk-adjusted performance.
    """
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("Statistical Analysis", fontsize=13, fontweight="bold", y=0.98)
    gs = GridSpec(2, 1, figure=fig,
                  top=0.93, bottom=0.08, left=0.03, right=0.97, hspace=0.55)

    # --- Statistical significance ---
    ax1 = fig.add_subplot(gs[0])
    sig_df = to_statistical_significance_df(results)
    fig.text(0.5, 0.935, "Table A. Statistical Significance (Welch t-test vs. Static AS baseline)",
             ha="center", fontsize=_FS_TITLE, fontweight="bold", color="#1A3A5C")
    _render_df_as_table(fig, ax1, sig_df, fontsize=8.2, row_height=1.60)

    # --- Risk-adjusted performance ---
    ax2 = fig.add_subplot(gs[1])
    ra_df = to_risk_adjusted_df(results)
    fig.text(0.5, 0.485, "Table B. Risk-Adjusted Performance Metrics",
             ha="center", fontsize=_FS_TITLE, fontweight="bold", color="#1A3A5C")
    _render_df_as_table(fig, ax2, ra_df, fontsize=8.2, row_height=1.60)

    _add_caption(fig,
        "Table A: Welch two-sample t-test; *p<0.05, **p<0.01, ***p<0.001, n.s. = not significant. "
        "Cohen's d: |d|<0.2 small, 0.2\u20130.5 medium, >0.8 large. "
        "Table B: Omega ratio = E[gains]/E[losses]; Sortino uses downside deviation.")
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _pdf_p7_new_tables_2(results, pdf: PdfPages) -> None:
    """
    Page 7: PnL distribution moments + Per-metric rank + Constraint satisfaction.
    """
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("Distribution Analysis & Ranking", fontsize=13, fontweight="bold", y=0.98)
    gs = GridSpec(3, 1, figure=fig,
                  top=0.93, bottom=0.06, left=0.03, right=0.97, hspace=0.72)

    # --- PnL distribution moments ---
    ax1 = fig.add_subplot(gs[0])
    dist_df = to_pnl_distribution_df(results)
    fig.text(0.5, 0.940, "Table C. Terminal PnL Distribution Moments",
             ha="center", fontsize=_FS_TITLE, fontweight="bold", color="#1A3A5C")
    _render_df_as_table(fig, ax1, dist_df, fontsize=7.8, row_height=1.50)

    # --- Per-metric rank ---
    ax2 = fig.add_subplot(gs[1])
    rank_df = to_per_metric_rank_df(results)
    fig.text(0.5, 0.645, "Table D. Per-Metric Rank (1 = best in column)",
             ha="center", fontsize=_FS_TITLE, fontweight="bold", color="#1A3A5C")
    _render_df_as_table(fig, ax2, rank_df, fontsize=7.5, row_height=1.50)

    # --- Constraint satisfaction ---
    ax3 = fig.add_subplot(gs[2])
    cst_df = to_constraint_satisfaction_df(results)
    fig.text(0.5, 0.345, "Table E. CMDP Constraint Satisfaction Rate",
             ha="center", fontsize=_FS_TITLE, fontweight="bold", color="#1A3A5C")
    _render_df_as_table(fig, ax3, cst_df, fontsize=8.0, row_height=1.55)

    _add_caption(fig,
        "Table C: distributional statistics of terminal PnL across evaluation episodes. "
        "Table D: rank 1 = best; lower Avg Rank = more consistently strong. "
        "Table E: fraction of episodes where the realised constraint metric stays within "
        "the CMDP training bound; slack > 0 means the constraint is comfortably satisfied.")
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


# ===========================================================================
# Public entry point
# ===========================================================================

def generate_figure_report(
    results: Dict[str, AgentMetrics],
    out_dir: str = "outputs/figures",
) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # CSV / LaTeX tables
    _export_tables(results, out_dir)

    # Individual figure PDFs
    _fig_table(           results, os.path.join(out_dir, "table_overview.pdf"))
    _fig_pnl_trajectory(  results, os.path.join(out_dir, "pnl_trajectory.pdf"))
    _fig_inventory_trajectory(results, os.path.join(out_dir, "inventory_trajectory.pdf"))
    _fig_bar_metrics(     results, os.path.join(out_dir, "bar_metrics.pdf"))
    _fig_risk_return(     results, os.path.join(out_dir, "risk_return_scatter.pdf"))
    _fig_violin(          results, os.path.join(out_dir, "terminal_pnl_violin.pdf"))
    _fig_inv_hist(        results, os.path.join(out_dir, "inventory_hist.pdf"))
    _fig_heatmap(         results, os.path.join(out_dir, "metrics_heatmap.pdf"))
    _fig_pnl_quantiles(   results, os.path.join(out_dir, "pnl_quantile_fan.pdf"))

    # Multi-page A4-landscape PDF
    pdf_path = os.path.join(out_dir, "hybrid_mm_results.pdf")
    with PdfPages(pdf_path) as pdf:
        meta = pdf.infodict()
        meta["Title"]   = "Hybrid Market-Making: Performance Report"
        meta["Subject"] = "Avellaneda-Stoikov + PPO residual control"
        meta["Creator"] = "figures.py"
        _pdf_p1_table(         results, pdf)
        _pdf_p2_trajectories(  results, pdf)
        _pdf_p3_bar_metrics(   results, pdf)
        _pdf_p4_distributions( results, pdf)
        _pdf_p5_heatmap_inv(   results, pdf)
        _pdf_p6_new_tables_1(  results, pdf)
        _pdf_p7_new_tables_2(  results, pdf)

    # Alias for backward compatibility
    try:
        import shutil
        shutil.copyfile(pdf_path, os.path.join(out_dir, "summary_report.pdf"))
    except Exception:
        pass

    return pdf_path
