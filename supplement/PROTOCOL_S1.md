# Supplementary Protocol S1: deterministic panel and interval construction

Status: supplementary protocol for the ICASSP 2027 paper. This note summarizes
the frozen rules; it does not redefine the experiment. A "panel" here is a
"matched configuration" in the paper.

## Chronology and legacy boundary

The original D/A/B/C configuration and its outcomes were already known when
the configuration-sensitivity analysis was designed. The B-A and B-C endpoints,
the panel generator, the number of panels, the completion rules, and the wording
gate were then frozen before any new-panel outcome was inspected.

Panel000 replayed the exact legacy membership, model selection, predictions,
and metrics as a preflight gate. It is excluded from the 12-panel distribution,
means, quantiles, positive counts, intervals, and claim gate. For each of the
106 birds with at least one alternative, the exact legacy tuple was excluded
from the alternative pool. The remaining bird had one legal tuple and retained
it in every new panel.

## Roster and identity labels

The 107 identities follow the public dataset's breeding-attempt/nestbox linkage
to a nonmissing `father` field, defined as the social father or male observed
feeding young. Each segment inherits that field-linked singer identity; clips
were not independently assigned from a separate visual observation of every
vocalization.

Eligibility required at least three song types, three recording days, 10 clips
in each D/A/B/C cell, and 10 clips in each of three post-exclusion training
margins: seen type, seen source session, and total training support. If one bird
had multiple eligible years, selection maximized the minimum capacity across
the four cells and three training margins, then preferred more total songs and
finally the earlier year.

## Source session and legal tuple

`source_session` is the `YYYYMMDD_HHMMSS` source-recording identifier parsed
from a released segment ID of the form
`<pnum>_<YYYYMMDD_HHMMSS>_<start_frame>`. Its date prefix must agree with the
recording day extracted from the CSV `datetime` field.

A tuple `(tU,tS,dU,sS,sU)` is legal when the two types differ; the two sessions
and their recording days differ; both types occur in both sessions; each cell
has at least 10 distinct eligible segments; and at least 10 seen-type,
seen-session, and total training rows remain after removing all held-out-type
rows, all rows on `dU`, and 10 D rows. The metadata-only enumeration found
3,166 legal bird-level exact tuples in total across the 107 fixed bird-years,
including the legacy tuple for each bird. This is the candidate-universe sum;
each global panel instead contains one bird-level tuple per bird.

## Global-panel sampling

Exactly 12 new global panels were generated with NumPy 2.1.0
`PCG64(20260913)`, processing birds in ascending Unicode `bird_id` order. For
each bird, alternative tuples were grouped by `(tU,tS,dU)`. One independently
permuted ordering of these structural keys was cycled as evenly as the 12-panel
budget allowed. Within each structural key, one independently permuted ordering
of legal `(sS,sU)` pairs was likewise cycled by visit count. The 12 resulting
107-bird tuple manifests were pairwise distinct. This equal-bird,
equal-structure procedure produced 1,284 bird-level selections comprising 918
unique exact tuples and 366 repeated selections. It was frozen to give balanced
coverage of feasible alternatives.

## Cell sampling and panel-specific fitting

After selected `(bird_id, tuple)` pairs were deduplicated, eligible segment IDs
were sorted. A single NumPy 2.1.0 `PCG64(20260914)` stream traversed
`bird_id -> tuple -> cell`, with cell order D, A, B, C, and sampled exactly 10
segments without replacement per cell. Repeated tuples reused the same local
test rows. Every panel excludes all `tU` rows, all rows on `dU`, and sampled D
rows from training.

For each panel and representation pipeline, five stratified training-only
folds were regenerated with `random_state=0`; scaling was fitted within fold;
regularization was selected from `{1e-4,1e-3,1e-2,1e-1,1,10,100}`; and the
scaler, linear classifier, and centroid readout were refitted from that panel's
training membership. All 24 pipeline-panel units had one successful registered
execution, with no failed or replacement panel.

## Contribution intervals

For each cell, class-F1 is computed from the full 107-way confusion matrix. A
bird-indexed class-F1 value therefore includes true positives and false
negatives for that true class plus false positives produced by segments whose
true class is another bird. The analysis aligns 107 class-F1 contrast
contributions across cells and panels.

For the conditional configuration-mean interval, each indexed contribution is
first averaged across the fixed 12 panels. A shared PCG64(20260912)
10,000-by-107 index matrix then selects 107 contribution indices per draw, and
their mean is recorded. The reported limits are the NumPy linear 2.5 and 97.5
percentiles. Panels remain fixed. The procedure does not refit a model or
recompute a confusion matrix from resampled segments. All intervals are
pointwise and unadjusted.

Machine-readable Table S1, `table_s1_panel_contrasts.csv`, contains every panel,
pipeline, readout, contrast, endpoint role, point estimate, planned pointwise
interval, and sign. Rows for `B_minus_A` and `B_minus_C` provide the primary
linear and secondary centroid values used in the manuscript figure and table.

## Post-hoc descriptive bird contributions

After the primary configuration analysis was frozen, each bird-indexed
class-F1 contrast contribution was averaged across the 12 panels to inspect
how broadly the mean ordering was distributed across the roster. For the
primary linear readout, BirdNET yielded 106 positive, zero zero, and one
negative contribution mean for B-A, and 106 positive, one zero, and zero
negative contribution means for B-C. Perch yielded 107 positive contribution
means for each contrast.

Supplementary Table S2,
`table_s2_posthoc_bird_mean_contributions.csv`, gives the complete 107-bird
values for all pipelines, readouts, and contrasts;
`table_s2_posthoc_bird_summary.csv` gives their descriptive summaries. These
tables are post-hoc descriptive outputs and contain no p-values or
population-level inference. `tuple_selection_summary.json` records the panel
and unique-tuple counts reported above.
