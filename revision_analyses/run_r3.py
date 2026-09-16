"""R3: does the pooling rule explain the gap? BirdNET only (see PREREG.md)."""
import csv, time, warnings
import numpy as np, probe_lib as L
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold

EMB_DIR = L.ART / "stage2/bundles/stage2-birdnet-20260912T151520743945Z-27261ba29bd0/outputs/embeddings"
import sys
PANELS = (sys.argv[1].split(",") if len(sys.argv) > 1 else [f"panel{i:03d}" for i in range(1, 13)])
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
VARIANTS = ("unweighted_mean", "max", "first_window")


def build_variants():
    """Re-pool cached window embeddings under three alternative rules."""
    rows = list(csv.DictReader((EMB_DIR / "birdnet_window_index.csv").open()))
    shards = {}
    per_seg = {}
    for r in rows:
        per_seg.setdefault(r["segment_id"], []).append(r)
    out = {v: {} for v in VARIANTS}
    order = sorted(per_seg)
    for seg in order:
        ws = sorted(per_seg[seg], key=lambda r: int(r["window_index_in_segment"]))
        vecs = []
        for w in ws:
            sh = w["window_shard"]
            if sh not in shards:
                shards[sh] = np.load(EMB_DIR / "window_shards" / sh, mmap_mode="r")
            vecs.append(np.asarray(shards[sh][int(w["window_row_in_shard"])], dtype=np.float64))
        V = np.stack(vecs)
        out["unweighted_mean"][seg] = V.mean(axis=0)
        out["max"][seg] = V.max(axis=0)
        out["first_window"][seg] = V[0]
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


t0 = time.time()
print("re-pooling window embeddings ...", flush=True)
VAR = build_variants()
print("done %.0fs" % (time.time() - t0), flush=True)
rows = []
for variant in VARIANTS:
    table = VAR[variant]
    for panel in PANELS:
        train, test, cval, frozen = L.panel_split(panel, "birdnet")
        classes = sorted({b for _, b, _ in train})
        X = np.stack([table[s] for s, _, _ in train]); y = np.array([b for _, b, _ in train])
        best_c, val = select_c(X, y, classes)
        sc, model = L.fit_linear(X, y, best_c, classes)
        f1 = {}
        for c in L.CELLS:
            Xc = np.stack([table[s] for s, _ in test[c]]); yc = np.array([b for _, b in test[c]])
            f1[c] = L.macro_f1(yc, model.predict(sc.transform(Xc)), classes)
        rows.append(dict(panel=panel, pipeline="birdnet", pooling=variant, selected_c=best_c,
                         **{f"macro_f1_{c}": f1[c] for c in L.CELLS},
                         B_minus_C=f1["B"] - f1["C"], B_minus_A=f1["B"] - f1["A"],
                         frozen_B_minus_C=frozen[("linear", "B")] - frozen[("linear", "C")]))
        print(f"{variant} {panel} C={best_c} B-C={rows[-1]['B_minus_C']:.3f} "
              f"(frozen {rows[-1]['frozen_B_minus_C']:.3f}) [{time.time()-t0:.0f}s]", flush=True)
        with open(f"r3_pooling_ablation{SUFFIX}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote r3_pooling_ablation.csv", len(rows), "rows")
