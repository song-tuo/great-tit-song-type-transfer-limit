"""Figure 2: repertoire-coverage curve with the size-neutral control (single column)."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures" / "figure_coverage"
MIN_PT, MM = 9.0, 1 / 25.4
INK = "#222222"
COL = {"birdnet": "#3E6DB5", "perch": "#D9453D"}
LAB = {"birdnet": "BirdNET", "perch": "Perch"}


def read(name):
    p = HERE / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def main():
    r2, r2b = read("r2_coverage_curve.csv"), read("r2b_size_control.csv")
    mpl.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.titlesize": 9.5,
                         "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "icassp2027-fig2"})
    fig, ax = plt.subplots(figsize=(86 * MM, 58 * MM))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.965, bottom=0.175)
    cov, seen = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for r in r2:
        cov[r["pipeline"]][int(r["m_exposed"])].append(float(r["macro_f1_C"]))
        seen[r["pipeline"]].append(float(r["macro_f1_B"]))
    ctrl = defaultdict(lambda: defaultdict(list))
    for r in r2b:
        ctrl[r["pipeline"]][int(r["m_swapped"])].append(float(r["macro_f1_C"]))
    ms = sorted({m for p in cov for m in cov[p]})
    xpos = {m: i for i, m in enumerate(ms)}
    for pipe in ("birdnet", "perch"):
        if not cov[pipe]:
            continue
        x = [xpos[m] for m in ms]
        mean = [np.mean(cov[pipe][m]) for m in ms]
        lo = [np.min(cov[pipe][m]) for m in ms]
        hi = [np.max(cov[pipe][m]) for m in ms]
        ax.fill_between(x, lo, hi, color=COL[pipe], alpha=0.15, lw=0)
        ax.plot(x, mean, marker="o" if pipe == "birdnet" else "s", ms=4, lw=1.4, color=COL[pipe])
        ax.axhline(np.mean(seen[pipe]), color=COL[pipe], lw=0.9, ls=(0, (4, 2.5)), alpha=0.75)
        for m, v in ctrl[pipe].items():
            ax.plot([xpos[m]], [np.mean(v)], marker="^", ms=5, mfc="white", mec=COL[pipe], mew=1.2)
    ax.set_xticks(list(xpos.values()), [str(m) for m in ms])
    ax.set_xlabel("Held-out-type clips per bird added to training")
    ax.set_ylabel("Held-out-type cell C (macro-F1)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#E3E3E3", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Line2D([0], [0], marker="o", color=COL["birdnet"], lw=1.4, ms=4, label="BirdNET"),
               Line2D([0], [0], marker="s", color=COL["perch"], lw=1.4, ms=4, label="Perch"),
               Line2D([0], [0], marker="^", color="#777777", lw=0, mfc="white", mec="#777777",
                      ms=5, label="Size-neutral swap"),
               Line2D([0], [0], color="#777777", lw=0.9, ls=(0, (4, 2.5)), label="Seen-type cell B")]
    ax.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.7,
              labelspacing=0.25, borderpad=0.2)
    for t in fig.findobj(mpl.text.Text):
        if t.get_text().strip() and t.get_visible() and t.get_fontsize() < MIN_PT - 1e-6:
            raise RuntimeError(f"text under {MIN_PT} pt: {t.get_text()!r}")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
