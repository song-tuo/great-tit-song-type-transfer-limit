"""R1: does a stronger decoder close the held-out-type gap? (see PREREG.md)"""
import csv, json, time, warnings
import numpy as np, probe_lib as L
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder

PANELS = [f"panel{i:03d}" for i in range(1, 13)]
PIPES = ("birdnet", "perch")
ALPHAS = (1e-4, 1e-3, 1e-2)
KS = (1, 5, 10)
rows = []
t_start = time.time()
for pipe in PIPES:
    for panel in PANELS:
        train, test, cval, frozen = L.panel_split(panel, pipe)
        classes = sorted({b for _, b, _ in train})
        le = LabelEncoder().fit(classes)
        ids = [s for s, _, _ in train]; y = np.array([b for _, b, _ in train])
        fold = np.array([f for _, _, f in train])
        X = L.matrix(pipe, ids)
        cells = {c: (L.matrix(pipe, [s for s, _ in test[c]]), np.array([b for _, b in test[c]]))
                 for c in L.CELLS}
        # --- MLP: select alpha on the frozen folds
        scores = {}
        for a in ALPHAS:
            vals = []
            for f in sorted(set(fold)):
                tr, va = fold != f, fold == f
                sc = StandardScaler().fit(X[tr])
                m = MLPClassifier(hidden_layer_sizes=(512,), alpha=a, early_stopping=True,
                                  n_iter_no_change=10, max_iter=300, random_state=0)
                m.fit(sc.transform(X[tr]), le.transform(y[tr]))
                vals.append(L.macro_f1(y[va], le.inverse_transform(m.predict(sc.transform(X[va]))), classes))
            scores[a] = float(np.mean(vals))
        best_a = max(sorted(ALPHAS), key=lambda a: scores[a])
        sc = StandardScaler().fit(X)
        mlp = MLPClassifier(hidden_layer_sizes=(512,), alpha=best_a, early_stopping=True,
                            n_iter_no_change=10, max_iter=300, random_state=0).fit(sc.transform(X), le.transform(y))
        # --- kNN: select k on the frozen folds
        Xn = normalize(X)
        kscores = {}
        for k in KS:
            vals = []
            for f in sorted(set(fold)):
                tr, va = fold != f, fold == f
                kn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance").fit(Xn[tr], y[tr])
                vals.append(L.macro_f1(y[va], kn.predict(Xn[va]), classes))
            kscores[k] = float(np.mean(vals))
        best_k = max(sorted(KS), key=lambda k: kscores[k])
        knn = KNeighborsClassifier(n_neighbors=best_k, metric="cosine", weights="distance").fit(Xn, y)
        for name, predict, hp in (("mlp", lambda M: le.inverse_transform(mlp.predict(sc.transform(M))), best_a),
                                  ("knn", lambda M: knn.predict(normalize(M)), best_k)):
            cellf1 = {}
            for c in L.CELLS:
                Xc, yc = cells[c]
                cellf1[c] = L.macro_f1(yc, predict(Xc), classes)
            rows.append(dict(panel=panel, pipeline=pipe, readout=name, hyperparameter=hp,
                             **{f"macro_f1_{c}": cellf1[c] for c in L.CELLS},
                             B_minus_C=cellf1["B"] - cellf1["C"], B_minus_A=cellf1["B"] - cellf1["A"]))
        # frozen linear for reference
        rows.append(dict(panel=panel, pipeline=pipe, readout="linear_frozen", hyperparameter=cval,
                         **{f"macro_f1_{c}": frozen[("linear", c)] for c in L.CELLS},
                         B_minus_C=frozen[("linear", "B")] - frozen[("linear", "C")],
                         B_minus_A=frozen[("linear", "B")] - frozen[("linear", "A")]))
        print(f"{pipe} {panel} alpha={best_a} k={best_k} "
              f"mlp B-C={rows[-3]['B_minus_C']:.3f} knn B-C={rows[-2]['B_minus_C']:.3f} "
              f"lin B-C={rows[-1]['B_minus_C']:.3f} [{time.time()-t_start:.0f}s]", flush=True)
with open("r1_decoder_strength.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote r1_decoder_strength.csv", len(rows), "rows")
