from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT_DIR = os.path.join("outputs", "figures")
DPI = 300

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": DPI,
        "figure.dpi": DPI,
    }
)


COLORS = {
    "obs": "#D7EAF8",
    "as": "#FCE6C9",
    "ppo": "#E6D8F5",
    "act": "#D9F0D2",
    "risk": "#F8D6D6",
    "lob_bid": "#B8D9FF",
    "lob_ask": "#FFD0D0",
    "mid": "#333333",
}


def _canvas(title: str, w: float = 12, h: float = 4.8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    return fig, ax


def _box(ax, x, y, w, h, text, fc, ec="#444444", lw=1.2, fontsize=10, bold=False):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2, text=None, color="#444444", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.4,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, text, ha="center", va="center", fontsize=9)


def fig_static_as():
    fig, ax = _canvas("Figure A1. Static AS Market-Making Architecture")
    _box(ax, 0.04, 0.32, 0.18, 0.34, "Market State\n(mid, inventory,\ntime to maturity)", COLORS["obs"], bold=True)
    _box(ax, 0.30, 0.32, 0.18, 0.34, "Avellaneda-Stoikov\nClosed-Form Backbone\n\nreservation price\noptimal spread", COLORS["as"], bold=True)
    _box(ax, 0.56, 0.32, 0.16, 0.34, "Quote Engine\n\nbid depth\nask depth", COLORS["act"], bold=True)
    _box(ax, 0.80, 0.32, 0.16, 0.34, "Limit Order Book\nExecution\n\nfills / no fills", COLORS["risk"], bold=True)

    _arrow(ax, 0.22, 0.49, 0.30, 0.49)
    _arrow(ax, 0.48, 0.49, 0.56, 0.49)
    _arrow(ax, 0.72, 0.49, 0.80, 0.49)
    _arrow(ax, 0.88, 0.32, 0.13, 0.24, "inventory and PnL feedback", rad=0.15)

    ax.text(0.50, 0.12, "No learning module: all quoting decisions come from the analytical AS policy.", ha="center", fontsize=10)
    return fig


def fig_model_free():
    fig, ax = _canvas("Figure A2. Model-Free PPO Market-Making Architecture")
    _box(ax, 0.04, 0.30, 0.18, 0.38, "State Vector\n(mid, inventory,\ntime, cash, etc.)", COLORS["obs"], bold=True)
    _box(ax, 0.30, 0.30, 0.18, 0.38, "Policy Network\n(PPO Actor)\n\n2-layer MLP", COLORS["ppo"], bold=True)
    _box(ax, 0.54, 0.30, 0.18, 0.38, "Direct Action Output\n\nbid depth\nask depth", COLORS["act"], bold=True)
    _box(ax, 0.80, 0.30, 0.16, 0.38, "Environment\nReward\nPnL and fills", COLORS["risk"], bold=True)

    _arrow(ax, 0.22, 0.49, 0.30, 0.49)
    _arrow(ax, 0.48, 0.49, 0.54, 0.49)
    _arrow(ax, 0.72, 0.49, 0.80, 0.49)
    _arrow(ax, 0.88, 0.30, 0.39, 0.22, "policy gradient update", rad=0.18)

    ax.text(0.50, 0.12, "The RL agent learns quotes from scratch without analytical structure.", ha="center", fontsize=10)
    return fig


def fig_hybrid():
    fig, ax = _canvas("Figure A3. Hybrid AS+RL Residual Architecture")
    _box(ax, 0.03, 0.28, 0.16, 0.40, "State Vector\n(mid, inventory,\ntime, features)", COLORS["obs"], bold=True)
    _box(ax, 0.25, 0.28, 0.18, 0.40, "AS Backbone\n\nbaseline bid/ask\nbaseline spread", COLORS["as"], bold=True)
    _box(ax, 0.49, 0.28, 0.18, 0.40, "PPO Residual Head\n\nspread residual\nskew residual", COLORS["ppo"], bold=True)
    _box(ax, 0.73, 0.28, 0.20, 0.40, "Hybrid Quote Layer\n\nbaseline + residual\ninventory shield\nrisk shaping", COLORS["act"], bold=True)

    _arrow(ax, 0.19, 0.55, 0.25, 0.55)
    _arrow(ax, 0.19, 0.41, 0.49, 0.41, "shared state input")
    _arrow(ax, 0.43, 0.55, 0.73, 0.55, "AS baseline quotes")
    _arrow(ax, 0.67, 0.41, 0.73, 0.41, "residual correction")
    _arrow(ax, 0.83, 0.28, 0.58, 0.17, "reward / constraints", rad=-0.18)

    ax.text(0.50, 0.10, "The analytical AS policy provides structure, while PPO only learns bounded corrections.", ha="center", fontsize=10)
    return fig


def fig_lob_levels():
    fig, ax = _canvas("Figure A4. Order Book Snapshot with Bid and Ask Sides", h=5.2)
    y0 = 0.20
    level_h = 0.10
    prices_bid = [99.97, 99.98, 99.99]
    prices_ask = [100.01, 100.02, 100.03]
    vols_bid = [180, 120, 80]
    vols_ask = [90, 130, 170]

    ax.plot([0.5, 0.5], [0.14, 0.84], color=COLORS["mid"], linewidth=2.0)
    ax.text(0.5, 0.87, "Mid Price = 100.00", ha="center", va="bottom", fontsize=11, fontweight="bold")

    for i, (p, v) in enumerate(zip(prices_bid[::-1], vols_bid[::-1])):
        y = y0 + i * 0.13
        width = 0.10 + 0.18 * (v / max(vols_bid))
        _box(ax, 0.5 - width, y, width, level_h, f"{p:.2f}\nVol {v}", COLORS["lob_bid"])

    for i, (p, v) in enumerate(zip(prices_ask, vols_ask)):
        y = y0 + i * 0.13
        width = 0.10 + 0.18 * (v / max(vols_ask))
        _box(ax, 0.5, y, width, level_h, f"{p:.2f}\nVol {v}", COLORS["lob_ask"])

    ax.text(0.23, 0.09, "Bid side\n(limit buy orders)", ha="center", fontsize=11, fontweight="bold", color="#1D5FA7")
    ax.text(0.77, 0.09, "Ask side\n(limit sell orders)", ha="center", fontsize=11, fontweight="bold", color="#B44444")
    ax.text(0.50, 0.03, "Best bid is the highest buy price; best ask is the lowest sell price.", ha="center", fontsize=10)
    return fig


def fig_bid_ask_dynamics():
    fig, ax = _canvas("Figure A5. Bid-Ask Spread, Inventory Skew, and Execution Logic", h=5.0)
    ax.plot([0.10, 0.90], [0.55, 0.55], color="#666666", linewidth=1.2, linestyle="--")
    ax.text(0.50, 0.59, "Mid price", ha="center", fontsize=11)

    ax.plot([0.24, 0.40], [0.55, 0.55], color="#1D5FA7", linewidth=6, solid_capstyle="butt")
    ax.plot([0.60, 0.76], [0.55, 0.55], color="#B44444", linewidth=6, solid_capstyle="butt")
    ax.text(0.32, 0.62, "Bid quote", ha="center", fontsize=10)
    ax.text(0.68, 0.62, "Ask quote", ha="center", fontsize=10)
    ax.text(0.50, 0.48, "Spread", ha="center", fontsize=10, fontweight="bold")

    _arrow(ax, 0.50, 0.35, 0.66, 0.35, "long inventory -> skew ask closer", color="#B44444")
    _arrow(ax, 0.50, 0.25, 0.34, 0.25, "short inventory -> skew bid closer", color="#1D5FA7")
    _arrow(ax, 0.32, 0.72, 0.32, 0.58, "market sell hits bid", color="#1D5FA7")
    _arrow(ax, 0.68, 0.72, 0.68, 0.58, "market buy lifts ask", color="#B44444")

    _box(ax, 0.08, 0.08, 0.25, 0.10, "Tighter quote\n-> higher fill probability", COLORS["lob_bid"])
    _box(ax, 0.38, 0.08, 0.25, 0.10, "Wider spread\n-> higher margin", COLORS["as"])
    _box(ax, 0.68, 0.08, 0.24, 0.10, "Inventory skew\n-> faster rebalancing", COLORS["risk"])
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    figures = [
        ("architecture_static_as.pdf", fig_static_as()),
        ("architecture_model_free_ppo.pdf", fig_model_free()),
        ("architecture_hybrid_as_rl.pdf", fig_hybrid()),
        ("lob_bid_ask_snapshot.pdf", fig_lob_levels()),
        ("lob_bid_ask_dynamics.pdf", fig_bid_ask_dynamics()),
    ]

    for filename, fig in figures:
        fig.savefig(os.path.join(OUT_DIR, filename))
        plt.close(fig)

    summary_path = os.path.join(OUT_DIR, "architecture_and_lob_diagrams.pdf")
    with PdfPages(summary_path) as pdf:
        for _, fig_builder in [
            ("architecture_static_as.pdf", fig_static_as),
            ("architecture_model_free_ppo.pdf", fig_model_free),
            ("architecture_hybrid_as_rl.pdf", fig_hybrid),
            ("lob_bid_ask_snapshot.pdf", fig_lob_levels),
            ("lob_bid_ask_dynamics.pdf", fig_bid_ask_dynamics),
        ]:
            fig = fig_builder()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved diagrams to: {OUT_DIR}")
    print(f"Bundle PDF: {summary_path}")


if __name__ == "__main__":
    main()
