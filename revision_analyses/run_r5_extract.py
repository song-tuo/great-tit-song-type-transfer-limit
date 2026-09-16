"""R5: extract Perch pre-pooling spatial features for a matched subset (see PREREG.md).

The SavedModel returns spatial_embedding (16 time x 4 freq x 1536) in the same forward
pass as the pooled embedding, so the richer representation costs one extra pass, not a
different model. The frozen frontend functions are imported, and the pooled embedding is
checked against the frozen matrix before anything else is used.
"""
import csv, os, json, os, sys, time
from pathlib import Path
import numpy as np

EXP = Path(os.environ.get("WYTHAM_EXP_ROOT", "."))
sys.path.insert(0, str(EXP / "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("KAGGLEHUB_OFFLINE", "1")
import probe_lib as L
from wytham_aiid.stage4_frontend import (decode_wav, resample_waveform, frame_waveform,
                                         SOURCE_FRAMES_PER_WINDOW, TARGET_SAMPLES_PER_WINDOW)

AUDIO = EXP / "data/derived/wytham_audio_v1"
MODEL = EXP / "artifacts/stage4/models/perch_v2_cpu_1-8d8582f33472"
PANELS = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["panel001"])
TRAIN_PER_BIRD = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED = 20260916
OUT = Path("r5_spatial")
OUT.mkdir(exist_ok=True)


def clip_list():
    """Test cells for each panel plus a deterministic training subsample."""
    wanted, meta = set(), {}
    for panel in PANELS:
        train, test, _, _ = L.panel_split(panel, "perch")
        for c in L.CELLS:
            for s, b in test[c]:
                wanted.add(s); meta[s] = b
        by_bird = {}
        for s, b, _ in train:
            by_bird.setdefault(b, []).append(s)
        rng = np.random.Generator(np.random.PCG64(SEED))
        for b in sorted(by_bird):
            pool = sorted(by_bird[b])
            take = min(TRAIN_PER_BIRD, len(pool))
            for i in sorted(rng.choice(len(pool), take, replace=False)):
                wanted.add(pool[i]); meta[pool[i]] = b
    return sorted(wanted), meta


def normalize(frames):
    f = frames.astype(np.float32).copy()
    f -= np.mean(f, axis=-1, keepdims=True)
    peak = np.max(np.abs(f), axis=-1, keepdims=True)
    f = np.divide(f, peak, where=(peak > 0.0))
    return f * 0.25


def main():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    fn = tf.saved_model.load(str(MODEL)).signatures["serving_default"]
    ids, meta = clip_list()
    print(f"{len(ids)} clips for {PANELS}", flush=True)
    frozen_mat, frozen_rows = L.load_embeddings("perch")
    reps = {"final": [], "spatial_meanmax": [], "freq_bands": [], "time_meanmax": []}
    checks = []
    t0 = time.time()
    for n, seg in enumerate(ids, 1):
        decoded = decode_wav(AUDIO / "WAV" / f"{seg}.wav")
        resampled = resample_waveform(decoded)
        frames, wrows = frame_waveform(resampled, decoded.shape[0])
        out = fn(inputs=tf.constant(normalize(frames)))
        emb = np.asarray(out["embedding"])                  # (W, 1536)
        sp = np.asarray(out["spatial_embedding"])           # (W, 16, 4, 1536)
        wt = np.asarray([r["valid_source_frames"] for r in wrows], dtype=np.float64)
        wt /= wt.sum()
        pooled_final = (emb.astype(np.float64) * wt[:, None]).sum(0)
        sp64 = sp.astype(np.float64)
        mm = np.concatenate([sp64.mean(axis=(1, 2)), sp64.max(axis=(1, 2))], axis=1)
        fb = sp64.mean(axis=1).reshape(sp64.shape[0], -1)                   # freq bands x 1536
        tf_mean = sp64.mean(axis=2)                                          # (W, 16, 1536)
        tm = np.concatenate([tf_mean.mean(axis=1), tf_mean.max(axis=1)], axis=1)
        reps["final"].append(pooled_final)
        reps["spatial_meanmax"].append((mm * wt[:, None]).sum(0))
        reps["freq_bands"].append((fb * wt[:, None]).sum(0))
        reps["time_meanmax"].append((tm * wt[:, None]).sum(0))
        ref = np.asarray(frozen_mat[frozen_rows[seg]], dtype=np.float64)
        checks.append(float(np.max(np.abs(pooled_final - ref))))
        if n % 250 == 0:
            print(f"  {n}/{len(ids)} [{time.time()-t0:.0f}s] max|repro-frozen| so far "
                  f"{max(checks):.2e}", flush=True)
    np.save(OUT / "segment_ids.npy", np.array(ids))
    for k, v in reps.items():
        np.save(OUT / f"{k}.npy", np.stack(v).astype(np.float32))
    json.dump({"panels": PANELS, "train_per_bird": TRAIN_PER_BIRD, "clips": len(ids),
               "max_abs_diff_vs_frozen_embedding": max(checks),
               "median_abs_diff": float(np.median(checks))},
              open(OUT / "extraction_report.json", "w"), indent=1)
    print("max |repro - frozen| =", max(checks), " (float32 eps ~1e-7 scale)")


if __name__ == "__main__":
    main()
