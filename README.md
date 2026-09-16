# Held-Out Song Types in Closed-Set Great Tit Identification: Decoders Do Not Close the Gap, Coverage Does

Supplementary material for the ICASSP 2027 paper by Song Dong (independent researcher, Hong Kong SAR, China).

The paper evaluates a fixed roster of 107 Wytham great tits with a matched 2×2 design that crosses song-type exposure (seen / held-out type) and source-session exposure (seen / held-out session), over 12 feasible matched configurations, with frozen BirdNET V2.4 and Perch 2.0 embeddings and linear and cosine-centroid readouts.

## Contents

| Path | Description |
|---|---|
| `supplement/PROTOCOL_S1.md` | Supplementary Protocol S1: deterministic configuration ("panel") sampling, cell membership, legacy gate, and contribution-interval rules |
| `supplement/table_s1_panel_contrasts.csv` | Table S1: every configuration, pipeline, readout, and contrast with its point estimate, pointwise descriptive interval, and sign |
| `supplement/table_s2_posthoc_bird_mean_contributions.csv` | Table S2: post-hoc bird-indexed class-F1 contrast contributions averaged across the 12 configurations (107 birds) |
| `supplement/table_s2_posthoc_bird_summary.csv` | Descriptive summaries of Table S2 |
| `supplement/tuple_selection_summary.json` | Configuration and unique-tuple counts (3,166 feasible tuples; 1,284 selections; 918 unique tuples) |
| `supplement/PROTOCOL_S2.md` | Supplementary Protocol S2: the secondary analyses (decoders, pooling, pre-pooling features, coverage curve, within-bird decodability) |
| `supplement/PREREGISTRATION_revision.md` | Design of the secondary analyses, fixed before their outcomes were inspected |
| `figure/figure_config_robustness_source_data.csv` | Source data for Fig. 1 |
| `figure/make_figure_config_robustness.py` | Script that draws Fig. 1 |
| `figure/make_coverage_figure.py` | Script that draws Fig. 2 from the coverage CSVs |
| `revision_analyses/*.py` | Runners for the secondary analyses; they read the frozen experiment tree (set `WYTHAM_EXP_ROOT`) and write only CSVs |
| `revision_analyses/r1_decoder_strength.csv` | Cell scores and contrasts for four decoders, 12 configurations x 2 pipelines |
| `revision_analyses/r2_coverage_curve.csv` | Coverage curve: held-out-type clips added per bird |
| `revision_analyses/r2b_size_control.csv` | Size-neutral control for the coverage curve |
| `revision_analyses/r3_pooling_ablation.csv` | Three alternative pooling rules (BirdNET) |
| `revision_analyses/r5_multilayer.csv` | Perch pre-pooling spatial representations |
| `revision_analyses/r6_within_bird_decodability.csv` | Within-bird song-type and session decodability |
| `revision_analyses/r4_per_bird.csv` | Per-bird averaged contrast contributions |
| `revision_analyses/summary.json` | Every number quoted in the paper, generated from the CSVs |

In the supplementary files, a "panel" corresponds to a "matched configuration" in the paper. Contrasts are named `B_minus_A` (B − A) and `B_minus_C` (B − C).

## Reproducing the figures

```bash
cd figure
python3 make_figure_config_robustness.py     # Fig. 1
python3 make_coverage_figure.py              # Fig. 2
```

Requires Python 3 with NumPy and matplotlib, and the Times New Roman font. The script checks that the plotted means match the values reported in the paper and that no text is smaller than 9 pt.

## Data source

The analyses use the publicly released Wytham Great Tit Song Dataset:

- N. Merino Recalde, A. Estandía, L. Pichot, A. Vansse, E. F. Cole, and B. C. Sheldon, "A densely sampled and richly annotated acoustic data set from a wild bird population," *Animal Behaviour*, vol. 211, pp. 111–122, 2024. doi:10.1016/j.anbehav.2024.02.008
- N. Merino Recalde, "Wytham Great Tit Song Dataset," OSF, 2023. doi:10.17605/OSF.IO/N8AC9

Bird identifiers in the tables follow the released dataset.
