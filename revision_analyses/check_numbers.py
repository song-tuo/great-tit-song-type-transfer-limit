"""Audit: every number in the paper's Results must exist in the generated CSVs.

Builds the set of legitimate values (3 decimals) from every run CSV plus the frozen
per-panel metrics, then scans the Results section of main.tex and reports numbers that
are not traceable. Structural numbers (counts like 12, 107, dimensions) are allowed via
an explicit list.
"""
import csv, glob, re, sys
from pathlib import Path
import probe_lib as L

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "paper" / "main.tex"
STRUCTURAL = {"0", "1", "2", "3", "4", "5", "10", "12", "20", "24", "40", "96", "100",
              "106", "107", "118", "242", "300", "512", "918", "1024", "1284", "1536",
              "1070", "3166", "47242", "74048", "109963", "22.05", "2.52", "16", "32",
              "48", "95", "10000", "0.5", "2027", "2.4", "96.8", "69.2", "3072", "6144"}

vals = set()


def add(x):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return
    vals.add(f"{f:.3f}")
    vals.add(f"{f:.2f}")


for path in glob.glob(str(HERE / "r*.csv")):
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                if any(s in k for s in ("macro_f1", "B_minus", "acc", "mean", "corr", "estimate", "ci_")):
                    add(v)
# frozen per-panel metrics (cells and contrasts, both readouts)
for panel in [f"panel{i:03d}" for i in range(1, 13)]:
    for pipe in ("birdnet", "perch"):
        try:
            _, _, _, frozen = L.panel_split(panel, pipe)
        except Exception:
            continue
        for v in frozen.values():
            add(v)
        for ro in ("linear", "centroid"):
            add(frozen[(ro, "B")] - frozen[(ro, "C")])
            add(frozen[(ro, "B")] - frozen[(ro, "A")])
            add(frozen[(ro, "D")] - frozen[(ro, "B")])
            add(frozen[(ro, "A")] - frozen[(ro, "C")])

import json
sm = HERE / "summary.json"
if sm.exists():
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            add(o)
    walk(json.load(sm.open()))

text = TEX.read_text()
start = text.index("\\section{Results}")
end = text.index("\\section{Discussion}")
results = text[start:end]
bad = []
for m in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w])", results):
    n = m.group(1)
    if n in STRUCTURAL or n in vals:
        continue
    f = float(n)
    if any(abs(f - float(v)) < 5e-4 for v in vals):
        continue
    line = results[:m.start()].count("\n") + 1
    bad.append((line, n, results[max(0, m.start() - 60):m.start() + 20].replace("\n", " ")))
print(f"{len(vals)} traceable values; {len(bad)} untraceable numbers in Results")
for line, n, ctx in bad:
    print(f"  line +{line}: {n}   ...{ctx}...")
sys.exit(1 if bad else 0)
