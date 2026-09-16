# Supplementary Protocol S2: secondary analyses

Status: supplementary protocol for the ICASSP 2027 paper. Every analysis below reuses
the frozen 12-configuration splits of Protocol S1: the same training membership, the
same test clips, and the same 107-identity roster. A "panel" here is a "matched
configuration" in the paper.

## Reproduction gate

Before any secondary analysis, the frozen linear probe was refitted from the cached
embeddings on one configuration and reproduced the frozen cell macro-F1 exactly
(difference 0.00e+00 in all four cells). For the Perch spatial-feature analysis, the
pooled embedding was recomputed from audio through the frozen frontend and matched the
frozen embedding matrix to float32 precision.

## Decoder strength

Two decoders join the frozen linear and cosine-centroid readouts:

- MLP, one hidden layer of 512 ReLU units, early stopping, L2 penalty selected from
  {1e-4, 1e-3, 1e-2}.
- Cosine k-nearest neighbours on L2-normalised embeddings, distance weights, k selected
  from {1, 5, 10}.

Selection uses the frozen five validation folds and validation macro-F1, the rule
already used for the linear penalty C. Test cells are never used for selection.

## Pooling

For BirdNET, segment embeddings were recomputed from the cached window embeddings under
three alternative rules: unweighted mean over windows, element-wise maximum, and
first-window only. C was re-selected for each variant. Perch was not re-pooled because
96.8% of its clips form a single window, where every pooling rule is the identity.

## Pre-pooling spatial features

The Perch SavedModel returns `spatial_embedding` with shape 16 (time) x 4 (frequency) x
1536 in the same forward pass as the pooled embedding. Four representations were
compared on identical splits: the final pooled embedding, mean-max over the spatial map,
per-frequency-band means, and temporal mean-max after averaging frequency. Scope: three
configurations with a fixed subsample of 40 training clips per bird, so absolute scores
are not comparable with the 12-configuration analysis; only the comparison between
representations on the same clips is interpreted.

## Repertoire coverage

For each bird, m held-out-type clips were moved into training, with m in {1, 2, 5, 10,
20}. The eligible pool is clips of the held-out type that are not on the held-out-session
day, so the held-out session stays held out and no test clip is ever added. Selection is
deterministic: eligible identifiers sorted, NumPy PCG64(20260916), sampled without
replacement per bird. Folds were regenerated and C re-selected for every m.

A size-neutral control repeats the same procedure at m in {5, 20} while removing the
same number of seen-type training clips per bird, never dropping a bird below 10
training clips, so the training set keeps its size and only its composition changes.

## What the embeddings encode

Within each bird, two balanced binary probes were fitted on the frozen embeddings:
two song types recorded inside one session, and two sessions of one song type. Each side
needs at least 10 clips and both sides are subsampled to equal size. Five-fold
stratified cross-validation, C = 1, standardised features, class-balanced weights.

## Reporting

Every number quoted in the paper is produced by `summarize.py` from the run CSVs in this
directory; none is transcribed by hand. Analyses that contradict the main claim are
reported rather than dropped.
