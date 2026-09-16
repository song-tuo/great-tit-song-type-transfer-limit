"""Reuse the frozen config-robustness splits with alternative decoders.

Reads only frozen artifacts; writes nothing into the experiment tree.
"""
import csv, os, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

EXP = Path(os.environ.get("WYTHAM_EXP_ROOT", "."))
ART = EXP / "artifacts"
BATCH = ART / "config_robustness_v6/batch_result_bundles/batch-launch-59afe248905046f6958e6f8567ca5e51/panels"
EMB = {
    "birdnet": (ART / "stage2/bundles/stage2-birdnet-20260912T151520743945Z-27261ba29bd0/outputs/embeddings/birdnet_embeddings.npy",
                ART / "stage2/bundles/stage2-birdnet-20260912T151520743945Z-27261ba29bd0/outputs/embeddings/birdnet_index.csv",
                "row_index"),
    "perch":   (ART / "stage4/representation_bundles/stage4c-perch-20260912T202124Z-fd7436472ce1/outputs/perch_embeddings.npy",
                ART / "stage4/representation_bundles/stage4c-perch-20260912T202124Z-fd7436472ce1/outputs/perch_index.csv",
                "global_segment_row"),
}
CELLS = ("D", "A", "B", "C")
_cache = {}


def load_embeddings(pipeline):
    if pipeline not in _cache:
        path, index_path, rowkey = EMB[pipeline]
        mat = np.load(path, mmap_mode="r")
        rows = {r["segment_id"]: int(r[rowkey]) for r in csv.DictReader(index_path.open())}
        _cache[pipeline] = (mat, rows)
    return _cache[pipeline]


def panel_split(panel, pipeline):
    """Training segments/labels and per-cell test segments, exactly as frozen."""
    out = BATCH / panel / "attempt001/outputs" / pipeline
    train = [(r["segment_id"], r["bird_id"], int(r["validation_fold"]))
             for r in csv.DictReader((out / "cv_folds.csv").open())]
    test = {c: [] for c in CELLS}
    for r in csv.DictReader((out / "predictions.csv").open()):
        if r["readout"] == "linear":
            test[r["condition"]].append((r["segment_id"], r["true_bird_id"]))
    selected_c = None
    import json
    selected_c = json.load((out / "cv_selection.json").open())["selected_c"]
    frozen = {}
    for r in csv.DictReader((out / "aggregate_metrics.csv").open()):
        frozen[(r["readout"], r["condition"])] = float(r["macro_f1"])
    return train, test, selected_c, frozen


def matrix(pipeline, segment_ids):
    mat, rows = load_embeddings(pipeline)
    idx = np.array([rows[s] for s in segment_ids])
    return np.asarray(mat[idx], dtype=np.float64)


def fit_linear(Xtr, ytr, c_value, classes):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(Xtr)
    model = LogisticRegression(C=float(c_value), penalty="l2", solver="lbfgs",
                               class_weight="balanced", fit_intercept=True, tol=1e-6,
                               max_iter=5000, random_state=0)
    model.fit(Xs, ytr)
    return scaler, model


def macro_f1(y_true, y_pred, classes):
    return f1_score(y_true, y_pred, labels=list(classes), average="macro", zero_division=0)
