"""R2: repertoire-coverage curve (see PREREG.md).

Adds m held-out-type clips per bird to training, keeping the held-out session held out,
then re-selects C with the frozen rule and refits.
"""
import csv, glob, time, warnings
import numpy as np, probe_lib as L
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold

PREP = glob.glob(str(L.ART / "config_robustness_v6/preparation_bundles/*/panels"))[0]
import sys
PANELS = (sys.argv[1].split(",") if len(sys.argv) > 1
          else [f"panel{i:03d}" for i in range(1, 13)])
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
PIPES = ("birdnet", "perch")
MS = (1, 2, 5, 10, 20)
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
SEED = 20260916


def eligible_pool(panel):
    """Per bird: held-out-type clips that are not on the held-out-session day."""
    pool = {}
    with open(f"{PREP}/{panel}/membership.csv") as fh:
        for r in csv.DictReader(fh):
            if r["split_reason"] == "excluded_tU":
                pool.setdefault(r["bird_id"], []).append(r["segment_id"])
    return {b: sorted(v) for b, v in pool.items()}


def select_c(X, y, classes):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    means = {}
    splits = list(folds.split(X, y))
    for c in C_GRID:
        vals = []
        for tr, va in splits:
            sc, m = L.fit_linear(X[tr], y[tr], c, classes)
            vals.append(L.macro_f1(y[va], m.predict(sc.transform(X[va])), classes))
        means[c] = float(np.mean(vals))
    best = max(means, key=lambda c: (means[c], -c))
    return best, means[best]


rows = []
t0 = time.time()
for pipe in PIPES:
    for panel in PANELS:
        train, test, cval, frozen = L.panel_split(panel, pipe)
        classes = sorted({b for _, b, _ in train})
        base_ids = [s for s, _, _ in train]
        base_y = [b for _, b, _ in train]
        pool = eligible_pool(panel)
        rng = np.random.Generator(np.random.PCG64(SEED))
        cells = {c: (L.matrix(pipe, [s for s, _ in test[c]]), np.array([b for _, b in test[c]]))
                 for c in L.CELLS}
        for m in MS:
            add_ids, add_y = [], []
            for bird in sorted(pool):
                avail = pool[bird]
                take = min(m, len(avail))
                pick = rng.choice(len(avail), size=take, replace=False)
                add_ids += [avail[i] for i in sorted(pick)]
                add_y += [bird] * take
            ids = base_ids + add_ids
            y = np.array(base_y + add_y)
            X = L.matrix(pipe, ids)
            best_c, val = select_c(X, y, classes)
            sc, model = L.fit_linear(X, y, best_c, classes)
            f1 = {c: L.macro_f1(cells[c][1], model.predict(sc.transform(cells[c][0])), classes)
                  for c in L.CELLS}
            rows.append(dict(panel=panel, pipeline=pipe, m_exposed=m, added_clips=len(add_ids),
                             selected_c=best_c, val_macro_f1=val,
                             **{f"macro_f1_{c}": f1[c] for c in L.CELLS},
                             B_minus_C=f1["B"] - f1["C"], B_minus_A=f1["B"] - f1["A"]))
            print(f"{pipe} {panel} m={m} (+{len(add_ids)}) C={best_c} C-cell={f1['C']:.3f} "
                  f"B-C={rows[-1]['B_minus_C']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        rows.append(dict(panel=panel, pipeline=pipe, m_exposed=0, added_clips=0,
                         selected_c=cval, val_macro_f1="",
                         **{f"macro_f1_{c}": frozen[("linear", c)] for c in L.CELLS},
                         B_minus_C=frozen[("linear", "B")] - frozen[("linear", "C")],
                         B_minus_A=frozen[("linear", "B")] - frozen[("linear", "A")]))
        with open(f"r2_coverage_curve{SUFFIX}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote r2_coverage_curve.csv", len(rows), "rows")
