# Revision analyses, pre-specified 2026-09-16

Scope: answer the three criticisms shared by the four simulated reviews, reusing the
frozen config_robustness_v6 splits. Nothing in the experiment tree is modified; all
runs read frozen artifacts and write only into this directory.

Reproduction check (passed before any new analysis): refitting the frozen linear probe
on panel001/BirdNET reproduced the frozen cell macro-F1 exactly (diff 0.00e+00 in D, A,
B, C).

## R1 Decoder strength (reviewer: "only linear and centroid readouts")
Question: does a stronger decoder on the same frozen embeddings close the held-out-type
gap?
- Decoders: (a) MLP, one hidden layer of 512 ReLU units, early stopping, alpha selected
  from {1e-4, 1e-3, 1e-2}; (b) cosine k-NN, k selected from {1, 5, 10}, distance
  weights, L2-normalised embeddings.
- Selection uses the frozen 5 validation folds and validation macro-F1, the rule already
  used for C. Test cells are untouched.
- Run over all 12 configurations x 2 pipelines. Report B-C and B-A per configuration.
- Exploratory note: MLP and k-NN were run once on panel001/BirdNET before this file was
  written; both showed a gap at least as large as the linear probe. The 12-configuration
  run is confirmatory.
- Prediction: the gap persists; a stronger decoder is not the missing ingredient.

## R2 Repertoire-coverage curve (reviewer: "is it a type boundary or data coverage?")
Question: how much exposure to the held-out type is needed to close the gap?
- For each bird, add m in {1, 2, 5, 10, 20} clips of its held-out type t_U to training.
- Eligible pool: clips marked `excluded_tU` (held-out type, not on the held-out-session
  day d_U), so the held-out session stays held out and no test clip is ever added.
- Deterministic selection: eligible ids sorted, NumPy PCG64(20260916), sampled without
  replacement per bird.
- For each m, folds are regenerated (StratifiedKFold, 5 splits, shuffle, random_state=0)
  and C is re-selected over the frozen grid by validation macro-F1, then refitted; the
  same procedure as the main experiment.
- Report cell C and B-C as a function of m, for both pipelines, over all 12
  configurations. m = 0 is the frozen result.

## R3 Aggregation ablation (reviewer: "the gap is an artifact of duration pooling")
- BirdNET only; Perch forms a single window for 96.8% of clips, so pooling is nearly
  the identity there.
- Recompute segment embeddings from the cached window embeddings with unweighted mean,
  max pooling, and first-window-only, then refit the linear probe with re-selected C.
- Report B-C per variant over the 12 configurations.

## Reporting rules
- Every new number enters the paper from the generated CSVs, never by hand.
- The frozen 12-configuration linear and centroid results stay the primary analysis;
  R1-R3 are declared secondary and are labelled as such in the paper.
- If any new analysis contradicts the main claim, it is reported, not dropped.

## Added the same day, before their runs produced any output

### R2b Size-neutral coverage control
R2 grows the training set, so recovery could reflect extra data rather than exposure to
the held-out type. R2b keeps the training-set size fixed: per bird it adds m in {5, 20}
held-out-type clips and removes the same number of seen-type training clips, never
dropping a bird below 10 training clips. Specified after R2 was launched and before any
R2 outcome was inspected.

### R5 Pre-pooling spatial features (Perch)
The Perch SavedModel returns `spatial_embedding` (16 time x 4 frequency x 1536) in the
same forward pass as the pooled embedding. Extraction reuses the frozen frontend
functions, and the recomputed pooled embedding is checked against the frozen matrix
before use (observed maximum absolute difference 3.6e-07, float32 precision).
Representations compared on identical splits: final pooled (reference), mean-max over
the spatial map, per-frequency-band means, and temporal mean-max. Scope: three
configurations with a fixed 40-clips-per-bird training subsample, so absolute scores are
not comparable with the 12-configuration analysis; only the comparison between
representations on the same clips is interpreted.

### R6 What the embeddings encode
Within each bird, a balanced binary probe separates (a) two song types recorded inside
one session and (b) two sessions of one song type, each with at least 10 clips per side,
5-fold stratified cross-validation, C = 1. Because both tasks are balanced and binary,
the two accuracies are directly comparable.
