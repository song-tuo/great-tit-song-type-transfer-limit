"""R2b: size-neutral coverage control.

R2 grows the training set, so part of any recovery could be extra data rather than
exposure to the held-out type. Here the training set keeps its size: for each bird, m
held-out-type clips are added AND m seen-type clips are removed. Any recovery is then
attributable to composition, not volume.
"""
import csv, glob, time, warnings
import numpy as np, probe_lib as L
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold

PREP = glob.glob(str(L.ART / "config_robustness_v6/preparation_bundles/*/panels"))[0]
import sys
PANELS = (sys.argv[1].split(",") if len(sys.argv) > 1 else [f"panel{i:03d}" for i in range(1, 13)])
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
MS = (5, 20)
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
SEED = 20260916


def pools(panel):
    """Per bird: (held-out-type clips off the held-out day, seen-type training clips)."""
    add, drop = {}, {}
    with open(f"{PREP}/{panel}/membership.csv") as fh:
        for r in csv.DictReader(fh):
            if r["split_reason"] == "excluded_tU":
                add.setdefault(r["bird_id"], []).append(r["segment_id"])
            elif r["cell"] == "TRAIN":
                drop.setdefault(r["bird_id"], []).append(r["segment_id"])
    return ({b: sorted(v) for b, v in add.items()}, {b: sorted(v) for b, v in drop.items()})


def select_c(X, y, classes):
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))
    means = {}
    for c in C_GRID:
        vals = []
        for tr, va in splits:
            sc, m = L.fit_linear(X[tr], y[tr], c, classes)
            vals.append(L.macro_f1(y[va], m.predict(sc.transform(X[va])), classes))
        means[c] = float(np.mean(vals))
    best = max(means, key=lambda c: (means[c], -c))
    return best, means[best]


rows, t0 = [], time.time()
for pipe in ("birdnet", "perch"):
    for panel in PANELS:
        train, test, cval, frozen = L.panel_split(panel, pipe)
        classes = sorted({b for _, b, _ in train})
        add_pool, drop_pool = pools(panel)
        base = [(s, b) for s, b, _ in train]
        cells_test = {c: (L.matrix(pipe, [s for s, _ in test[c]]),
                          np.array([b for _, b in test[c]])) for c in L.CELLS}
        for m in MS:
            rng = np.random.Generator(np.random.PCG64(SEED))
            add_ids, drop_ids = [], set()
            for bird in sorted(add_pool):
                avail = add_pool[bird]
                take = min(m, len(avail))
                for i in sorted(rng.choice(len(avail), size=take, replace=False)):
                    add_ids.append((avail[i], bird))
                own = drop_pool.get(bird, [])
                ndrop = min(take, max(0, len(own) - 10))
                for i in sorted(rng.choice(len(own), size=ndrop, replace=False)):
                    drop_ids.add(own[i])
            kept = [(s, b) for s, b in base if s not in drop_ids]
            ids = [s for s, _ in kept] + [s for s, _ in add_ids]
            y = np.array([b for _, b in kept] + [b for _, b in add_ids])
            X = L.matrix(pipe, ids)
            best_c, val = select_c(X, y, classes)
            sc, model = L.fit_linear(X, y, best_c, classes)
            f1 = {c: L.macro_f1(cells_test[c][1], model.predict(sc.transform(cells_test[c][0])), classes)
                  for c in L.CELLS}
            rows.append(dict(panel=panel, pipeline=pipe, m_swapped=m,
                             added=len(add_ids), removed=len(drop_ids), train_size=len(ids),
                             base_train_size=len(base), selected_c=best_c, val_macro_f1=val,
                             **{f"macro_f1_{c}": f1[c] for c in L.CELLS},
                             B_minus_C=f1["B"] - f1["C"],
                             frozen_B_minus_C=frozen[("linear", "B")] - frozen[("linear", "C")]))
            print(f"{pipe} {panel} swap m={m} (+{len(add_ids)}/-{len(drop_ids)}, "
                  f"size {len(base)}->{len(ids)}) C={best_c} C-cell={f1['C']:.3f} "
                  f"B-C={rows[-1]['B_minus_C']:.3f} (frozen {rows[-1]['frozen_B_minus_C']:.3f}) "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            with open(f"r2b_size_control{SUFFIX}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"wrote r2b_size_control{SUFFIX}.csv", len(rows))
