#!/usr/bin/env python3
"""Restyle Fig. 1 (configuration robustness) for the ICASSP 2027 camera-ready.

Content is unchanged from revision_asset_builder.py: the same 96 configuration
estimates, configuration means, and descriptive 95% class-index intervals, read
from the frozen source-data snapshot. Only the visual style changes:

  * Times New Roman (matches the newtx Times body text), every text >= 9 pt
  * reference-figure look: dashed coloured frames group the primary (linear)
    and secondary (centroid) readouts, legend on top, coloured bold values
  * panel letters (a)-(d) so they cannot be confused with cells A-D
  * printed at natural size: 178 mm wide for a figure* float
"""

import csv
import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "figure_config_robustness_source_data.csv"
OUT = HERE / "figure_config_robustness"

MIN_PT = 9.0
MM = 1 / 25.4
PANEL_ORDER = [("birdnet", "linear"), ("perch", "linear"),
               ("birdnet", "centroid"), ("perch", "centroid")]
PIPELINE_LABEL = {"birdnet": "BirdNET", "perch": "Perch"}
CONTRASTS = ("B_minus_A", "B_minus_C")
CONTRAST_LABEL = {"B_minus_A": "B − A", "B_minus_C": "B − C"}

INK = "#222222"
RED = "#D9453D"      # B - C (direct same-session contrast)
BLUE = "#3E6DB5"     # B - A
TEAL = "#2E9C96"     # frame: linear readout (primary)
PURPLE = "#8E6BBF"   # frame: centroid readout (secondary)
COLORS = {"B_minus_A": BLUE, "B_minus_C": RED}
MARKERS = {"B_minus_A": "o", "B_minus_C": "s"}
FRAME = {"linear": TEAL, "centroid": PURPLE}

# Values printed in Table 1 / Sec. 4; the figure must agree with them.
EXPECTED_MEANS = {("birdnet", "linear", "B_minus_A"): 0.624, ("birdnet", "linear", "B_minus_C"): 0.705,
                  ("perch", "linear", "B_minus_A"): 0.735, ("perch", "linear", "B_minus_C"): 0.782,
                  ("birdnet", "centroid", "B_minus_A"): 0.525, ("birdnet", "centroid", "B_minus_C"): 0.559,
                  ("perch", "centroid", "B_minus_A"): 0.515, ("perch", "centroid", "B_minus_C"): 0.527}


def load():
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8")))
    if len(rows) != 96:
        raise RuntimeError(f"expected 96 rows, found {len(rows)}")
    data = {}
    for r in rows:
        data.setdefault((r["pipeline"], r["readout"], r["contrast"]), []).append(r)
    for combo, rs in data.items():
        est = np.array([float(r["estimate"]) for r in rs])
        mean = float(rs[0]["config_mean"])
        if len(rs) != 12 or not np.isclose(est.mean(), mean, atol=1e-12):
            raise RuntimeError(f"snapshot inconsistent for {combo}")
        if round(mean, 3) != EXPECTED_MEANS[combo]:
            raise RuntimeError(f"{combo}: mean {mean:.3f} disagrees with paper {EXPECTED_MEANS[combo]}")
        if not (est > 0).all():
            raise RuntimeError(f"{combo}: not 12/12 positive")
    return data


def draw(data):
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "icassp2027-fig1",
    })
    W, H = 178 * MM, 77 * MM
    fig, axes = plt.subplots(2, 2, figsize=(W, H), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.785, bottom=0.125, wspace=0.13, hspace=0.62)

    offsets = np.linspace(-0.14, 0.14, 12)
    ypos = {"B_minus_A": 1.0, "B_minus_C": 0.0}
    for ax, letter, (pipeline, readout) in zip(axes.flat, "abcd", PANEL_ORDER):
        ax.axvline(0.0, color="#7A7A7A", linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
        ax.grid(axis="x", color="#E3E3E3", linewidth=0.5, zorder=0)
        for contrast in CONTRASTS:
            rs = data[(pipeline, readout, contrast)]
            est = [float(r["estimate"]) for r in rs]
            mean = float(rs[0]["config_mean"])
            lo, hi = float(rs[0]["conditional_ci_low"]), float(rs[0]["conditional_ci_high"])
            col, y = COLORS[contrast], ypos[contrast]
            ax.scatter(est, y + offsets, s=15, marker=MARKERS[contrast], facecolor="white",
                       edgecolor=col, linewidth=0.8, zorder=3)
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="D", markersize=5,
                        markerfacecolor=col, markeredgecolor=INK, markeredgewidth=0.5,
                        ecolor=col, elinewidth=1.7, capsize=2.8, capthick=1.0, zorder=5)
            ax.text(0.40, y, f"{mean:.3f}", color=col, fontsize=9.5, fontweight="bold",
                    va="center", ha="right", zorder=6)
        ax.set_title(f"({letter}) {PIPELINE_LABEL[pipeline]}, {readout}", loc="left",
                     color=INK, pad=3.5, fontweight="bold")
        ax.set_xlim(-0.035, 0.885)
        ax.set_ylim(-0.45, 1.45)
        ax.set_yticks([1.0, 0.0], [CONTRAST_LABEL[c] for c in CONTRASTS])
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(axis="both", width=0.6, length=2.6, color="#666666", pad=2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#777777")
        ax.tick_params(labelbottom=True)

    # Dashed rounded frames around each readout row, as in the reference figure.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for row, readout in enumerate(("linear", "centroid")):
        boxes = [inv.transform_bbox(ax.get_tightbbox(renderer)) for ax in axes[row]]
        x0 = min(b.x0 for b in boxes) - 0.006
        x1 = max(b.x1 for b in boxes) + 0.006
        y0 = min(b.y0 for b in boxes) - 0.012
        y1 = max(b.y1 for b in boxes) + 0.012
        fig.patches.append(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.012",
            transform=fig.transFigure, fill=False, edgecolor=FRAME[readout],
            linewidth=1.0, linestyle=(0, (4, 2.5)), zorder=0))

    fig.supxlabel("Macro-F1 contrast", x=0.525, y=0.005, fontsize=9.5, color=INK)

    def frame_handle(col):
        return Rectangle((0, 0), 1, 1, fill=False, edgecolor=col, linewidth=1.0, linestyle=(0, (3, 2)))

    handles = [
        frame_handle(TEAL), frame_handle(PURPLE),
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.2, markerfacecolor="white",
               markeredgecolor="#555555", markeredgewidth=0.8),
        Line2D([0], [0], marker="D", linestyle="-", markersize=4.6, color="#555555",
               markerfacecolor="#555555", linewidth=1.6),
    ]
    labels = ["Linear readout (primary)", "Centroid readout (secondary)",
              "12 configuration estimates", "Mean, 95% class-index interval"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1.0), ncol=2,
               frameon=False, handlelength=1.6, handletextpad=0.45, columnspacing=1.1,
               borderaxespad=0.2)
    return fig


def check_text(fig):
    small = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_text().strip() and t.get_visible() and t.get_fontsize() < MIN_PT - 1e-6:
            small.append((t.get_text(), t.get_fontsize()))
    if small:
        raise RuntimeError(f"text below {MIN_PT} pt: {small}")
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    for t in fig.findobj(mpl.text.Text):
        if t.get_text().strip() and t.get_visible():
            b = t.get_window_extent(r)
            if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y0 < fb.y0 - 0.5 or b.y1 > fb.y1 + 0.5:
                raise RuntimeError(f"text outside figure: {t.get_text()!r}")


def main():
    data = load()
    fig = draw(data)
    check_text(fig)
    meta = {"Title": "Configuration robustness of the two macro-F1 contrasts", "Author": "Song Dong",
            "CreationDate": None}
    fig.savefig(f"{OUT}.pdf", metadata=meta)
    fig.savefig(f"{OUT}.svg", metadata={"Title": meta["Title"], "Date": None})
    fig.savefig(f"{OUT}.png", dpi=300)
    plt.close(fig)
    sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    print(f"source snapshot sha256 {sha}")
    print(f"wrote {OUT}.svg / .pdf / .png")


if __name__ == "__main__":
    main()
