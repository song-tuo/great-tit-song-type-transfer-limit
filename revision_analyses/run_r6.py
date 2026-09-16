"""R6: within-bird decodability of song type vs source session (frozen embeddings).

Type task: two song types inside ONE session (session held constant).
Session task: two sessions inside ONE song type (type held constant).
Both are balanced binary problems, so the two numbers are comparable.
"""
import csv, json, warnings
from collections import defaultdict
import numpy as np, probe_lib as L
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_CLIPS = 10
SEED = 20260916
man = list(csv.DictReader((L.ART / "manifests/primary_song_manifest.csv").open()))
by_bird = defaultdict(list)
for r in man:
    by_bird[r["bird_id"]].append(r)


def balanced_binary(groups, rng):
    """Pick the two largest groups and subsample both to the same size."""
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if len(ordered) < 2 or len(ordered[1][1]) < MIN_CLIPS:
        return None
    (ka, a), (kb, b) = ordered[0], ordered[1]
    n = min(len(a), len(b))
    a = [a[i] for i in sorted(rng.choice(len(a), n, replace=False))]
    b = [b[i] for i in sorted(rng.choice(len(b), n, replace=False))]
    return a + b, [0] * n + [1] * n


def score(pipeline, ids, y):
    X = L.matrix(pipeline, ids)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000,
                                                            class_weight="balanced", random_state=0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return float(np.mean(cross_val_score(clf, X, np.asarray(y), cv=cv, scoring="accuracy")))


rows = []
for pipe in ("birdnet", "perch"):
    for bird in sorted(by_bird):
        rng = np.random.Generator(np.random.PCG64(SEED))
        clips = by_bird[bird]
        # type task: inside the session with the most types that have >= MIN_CLIPS
        best = None
        per_session = defaultdict(lambda: defaultdict(list))
        for r in clips:
            per_session[r["source_session"]][r["class_id"]].append(r["segment_id"])
        for sess, types in per_session.items():
            ok = {k: v for k, v in types.items() if len(v) >= MIN_CLIPS}
            if len(ok) >= 2:
                cand = balanced_binary(ok, rng)
                if cand and (best is None or len(cand[1]) > len(best[1])):
                    best = cand
        type_acc = score(pipe, *best) if best else None
        # session task: inside the type with two sessions that have >= MIN_CLIPS
        best2 = None
        per_type = defaultdict(lambda: defaultdict(list))
        for r in clips:
            per_type[r["class_id"]][r["source_session"]].append(r["segment_id"])
        for typ, sessions in per_type.items():
            ok = {k: v for k, v in sessions.items() if len(v) >= MIN_CLIPS}
            if len(ok) >= 2:
                cand = balanced_binary(ok, rng)
                if cand and (best2 is None or len(cand[1]) > len(best2[1])):
                    best2 = cand
        sess_acc = score(pipe, *best2) if best2 else None
        if type_acc is None and sess_acc is None:
            continue
        rows.append(dict(pipeline=pipe, bird_id=bird,
                         type_task_acc="" if type_acc is None else type_acc,
                         type_task_n=0 if not best else len(best[1]),
                         session_task_acc="" if sess_acc is None else sess_acc,
                         session_task_n=0 if not best2 else len(best2[1])))
    done = [r for r in rows if r["pipeline"] == pipe]
    ta = [r["type_task_acc"] for r in done if r["type_task_acc"] != ""]
    sa = [r["session_task_acc"] for r in done if r["session_task_acc"] != ""]
    print(f"{pipe}: type task n={len(ta)} median acc {np.median(ta):.3f} | "
          f"session task n={len(sa)} median acc {np.median(sa):.3f}", flush=True)
with open("r6_within_bird_decodability.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote r6_within_bird_decodability.csv", len(rows))
