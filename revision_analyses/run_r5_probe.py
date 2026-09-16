"""R5 probes: final pooled embedding vs pre-pooling spatial features, same splits."""
import csv, json, sys, time, warnings
from pathlib import Path
import numpy as np, probe_lib as L
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold

OUT = Path("r5_spatial")
PANELS = json.load(open(OUT / "extraction_report.json"))["panels"]
TRAIN_PER_BIRD = json.load(open(OUT / "extraction_report.json"))["train_per_bird"]
SEED = 20260916
C_GRID = (1e-2, 1e-1, 1.0, 10.0)
REPS = ("final", "spatial_meanmax", "freq_bands", "time_meanmax")

ids = list(np.load(OUT / "segment_ids.npy"))
row = {s: i for i, s in enumerate(ids)}
mats = {k: np.load(OUT / f"{k}.npy") for k in REPS}


def train_subsample(panel):
    train, _, _, _ = L.panel_split(panel, "perch")
    by_bird = {}
    for s, b, _ in train:
        by_bird.setdefault(b, []).append(s)
    rng = np.random.Generator(np.random.PCG64(SEED))
    out = []
    for b in sorted(by_bird):
        pool = sorted(by_bird[b])
        take = min(TRAIN_PER_BIRD, len(pool))
        out += [(pool[i], b) for i in sorted(rng.choice(len(pool), take, replace=False))]
    return out


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


rows = []
t0 = time.time()
for panel in PANELS:
    tr = train_subsample(panel)
    _, test, _, frozen = L.panel_split(panel, "perch")
    classes = sorted({b for _, b in tr})
    yt = np.array([b for _, b in tr])
    idx_tr = np.array([row[s] for s, _ in tr])
    for rep in REPS:
        X = mats[rep][idx_tr].astype(np.float64)
        best_c, val = select_c(X, yt, classes)
        sc, model = L.fit_linear(X, yt, best_c, classes)
        f1 = {}
        for c in L.CELLS:
            idx = np.array([row[s] for s, _ in test[c]])
            yc = np.array([b for _, b in test[c]])
            f1[c] = L.macro_f1(yc, model.predict(sc.transform(mats[rep][idx].astype(np.float64))), classes)
        rows.append(dict(panel=panel, representation=rep, dim=int(mats[rep].shape[1]),
                         selected_c=best_c, val_macro_f1=val,
                         **{f"macro_f1_{c}": f1[c] for c in L.CELLS},
                         B_minus_C=f1["B"] - f1["C"], B_minus_A=f1["B"] - f1["A"],
                         frozen_full_train_B_minus_C=frozen[("linear", "B")] - frozen[("linear", "C")]))
        print(f"{panel} {rep} (d={mats[rep].shape[1]}) C={best_c} B={f1['B']:.3f} C={f1['C']:.3f} "
              f"B-C={rows[-1]['B_minus_C']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        with open("r5_multilayer.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote r5_multilayer.csv", len(rows))
