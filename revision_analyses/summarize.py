"""Every new number quoted in the paper is produced here, from the run CSVs."""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def read(name):
    p = HERE / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def stats(vals):
    v = np.asarray(vals, dtype=float)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                min=float(v.min()), max=float(v.max()), positive=int((v > 0).sum()))


out = {}
r1 = read("r1_decoder_strength.csv")
if r1:
    d = defaultdict(list)
    for r in r1:
        d[(r["pipeline"], r["readout"])].append(float(r["B_minus_C"]))
        d[(r["pipeline"], r["readout"], "BA")].append(float(r["B_minus_A"]))
    out["r1_decoder_B_minus_C"] = {f"{k[0]}|{k[1]}": stats(v) for k, v in d.items() if len(k) == 2}
    out["r1_decoder_B_minus_A"] = {f"{k[0]}|{k[1]}": stats(v) for k, v in d.items() if len(k) == 3}
    cells = defaultdict(list)
    for r in r1:
        for c in "DABC":
            cells[(r["pipeline"], r["readout"], c)].append(float(r[f"macro_f1_{c}"]))
    out["r1_cell_means"] = {f"{k[0]}|{k[1]}|{k[2]}": round(float(np.mean(v)), 4) for k, v in cells.items()}

r2 = read("r2_coverage_curve.csv")
if r2:
    cov = defaultdict(lambda: defaultdict(list))
    for r in r2:
        cov[r["pipeline"]][int(r["m_exposed"])].append(float(r["macro_f1_C"]))
    gap = defaultdict(lambda: defaultdict(list))
    for r in r2:
        gap[r["pipeline"]][int(r["m_exposed"])].append(float(r["B_minus_C"]))
    out["r2_cellC_by_m"] = {p: {m: stats(v) for m, v in sorted(d.items())} for p, d in cov.items()}
    out["r2_gap_by_m"] = {p: {m: stats(v) for m, v in sorted(d.items())} for p, d in gap.items()}

r3 = read("r3_pooling_ablation.csv")
if r3:
    d = defaultdict(list)
    for r in r3:
        d[r["pooling"]].append(float(r["B_minus_C"]))
    out["r3_pooling_B_minus_C"] = {k: stats(v) for k, v in d.items()}
    out["r3_frozen_reference"] = stats([float(r["frozen_B_minus_C"]) for r in r3 if r["pooling"] == "unweighted_mean"])

r4 = read("r4_per_bird.csv")
if r4:
    per = defaultdict(list)
    for r in r4:
        per[r["pipeline"]].append((float(r["mean_B_minus_C"]), float(r["mean_train_clips"]), float(r["train_song_types"])))
    res = {}
    for p, v in per.items():
        g = np.array([x[0] for x in v]); t = np.array([x[1] for x in v]); ty = np.array([x[2] for x in v])
        res[p] = dict(**stats(g), corr_train_clips=float(np.corrcoef(g, t)[0, 1]),
                      corr_song_types=float(np.corrcoef(g, ty)[0, 1]))
    out["r4_per_bird"] = res


r5 = read("r5_multilayer.csv")
if r5:
    d = defaultdict(lambda: defaultdict(list))
    for r in r5:
        for k in ("macro_f1_B", "macro_f1_C", "B_minus_C"):
            d[r["representation"]][k].append(float(r[k]))
    out["r5_representations"] = {
        k: dict(dim=int([r["dim"] for r in r5 if r["representation"] == k][0]),
                B=round(float(np.mean(v["macro_f1_B"])), 4),
                C=round(float(np.mean(v["macro_f1_C"])), 4),
                **{"B_minus_C": stats(v["B_minus_C"])}) for k, v in d.items()}

r2b = read("r2b_size_control.csv")
if r2b:
    d = defaultdict(lambda: defaultdict(list))
    for r in r2b:
        d[(r["pipeline"], int(r["m_swapped"]))]["C"].append(float(r["macro_f1_C"]))
        d[(r["pipeline"], int(r["m_swapped"]))]["BC"].append(float(r["B_minus_C"]))
    out["r2b_size_neutral"] = {f"{k[0]}|m={k[1]}": dict(cellC=stats(v["C"]), gap=stats(v["BC"]))
                               for k, v in sorted(d.items())}

(HERE / "summary.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print("##", k)
    print(json.dumps(v, indent=1)[:1200])
