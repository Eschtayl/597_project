# Phase 3 — Supervised Flow-Level IDS (False-Positive Reduction)

Second stage of the two-stage intrusion-detection system. Phase 2 (unsupervised,
packet-level, recall-tuned) flags anomalous packets; Phase 3 re-checks each flagged
packet against a supervised flow-level classifier and suppresses false positives
while retaining true detections. The cascade can only *remove* alerts, never add:

```
ŷ_cascade = ŷ_phase2 AND ŷ_phase3
```

## Data requirements

CSVs are gitignored. Place the CIC IoT captures under the project root:

```
flow_and_packet/
  flow_based/     *.pcap_Flow.csv   (CICFlowMeter flow statistics, 84 columns)
  packet_based/   *.csv             (per-packet features, 135 columns)
```

(`config.py` falls back to root-level `flow_based/` / `packet_based/` if
`flow_and_packet/` is absent.) Labels are derived from filenames — the flow
`Label` column is `NeedManualLabel` and is never used.

## Pipeline & run order

Run everything from the project root (or from inside `phase_3/`). Each modelling
script prints PASS/FAIL checkpoint verifications and writes a `*_report.txt`
(committed) plus machine-readable outputs in `phase_3/data/` (gitignored).

| # | Command | What it does |
|---|---------|--------------|
| 1 | `python phase_3/check_aggregation.py` | Verify LogicalFlowID construction + aggregation rules (10 integrity tests) |
| 2 | `python phase_3/check_mapping.py` | Packet→flow mapping quality report (canonical 5-tuple key) |
| 3 | `python phase_3/check_sampling.py` | Build/cache the 204k unified-flow sample; verify sampler + splits |
| 4 | `python phase_3/check_features.py` | Verify feature builders + leakage rules; save feature manifests |
| 5 | `python phase_3/baselines.py` | Dummy / Logistic Regression / Random Forest baselines (validation only) |
| 6 | `python phase_3/xgboost_model.py` | XGBoost hyperparameter search (params reused by everything downstream) |
| 7 | `python phase_3/weighting_comparison.py` | none/balanced/dampened class weighting — **none wins (locked)** |
| 8 | `python phase_3/heads.py` | Train the four heads: {multiclass, binary} × {behaviour-only, service-aware} |
| 9 | `python phase_3/phase2_cohort.py` | Reproduce Phase 2 (Isolation Forest) packet alerts with a train/val/test split |
| 10 | `python phase_3/cascade_alignment.py` | Map val/test alert packets to the full 986k-flow index; run the leakage guard |
| 11 | `python phase_3/cascade_heads.py` | Remove leaking flows, retrain heads (locked params), score candidate flows |
| 12 | `python phase_3/cascade_eval.py` | Threshold sweep + Policy A/B bake-off on val alerts; single sealed-test evaluation |
| 13 | `python phase_3/ablations.py` | Model-progression ladder + engineered-feature ablation + overhead timing |
| 14 | `python phase_3/temporal_robustness.py` | Expanding-window temporal folds (robustness view) |
| 15 | `python phase_3/interpretability.py` | Gain/permutation importance, per-class SHAP, four cascade case studies |

## File map

**Configuration & shared libraries** (imported, not run directly)

- `config.py` — all paths, labels, frozen constants (gap cutoff, sample sizes, seed).
- `flow_aggregation.py` — LogicalFlowID assignment (Flow ID + 300 s temporal
  clustering) and per-feature segment aggregation (sum / min / max / weighted mean /
  pooled std / recomputed rates). `feature_dictionary()` documents every rule.
- `packet_flow_mapping.py` — canonical bidirectional 5-tuple key; maps packets to
  candidate logical flows and classifies match ambiguity.
- `flow_sampling.py` — 200k benign + ~4k attack sample drawn over *unified* flows.
- `splits.py` — grouped stratified train/val/test split (primary) and blocked
  temporal folds (robustness).
- `features.py` — behaviour-only and service-aware feature matrices, engineered
  features, frozen diagnostic definitions (service alignment, Brute Force buckets), and
  train-only preprocessing builders.
- `model_eval.py` — shared multiclass + binary-collapse metric computation.

**Verification scripts** — `check_aggregation.py`, `check_mapping.py`,
`check_sampling.py`, `check_features.py` (all must PASS before modelling).

**Models & experiments** — `baselines.py`, `xgboost_model.py`,
`weighting_comparison.py`, `heads.py`.

**Cascade** — `phase2_cohort.py`, `cascade_alignment.py`, `cascade_heads.py`,
`cascade_eval.py`.

**Analysis** — `ablations.py`, `temporal_robustness.py`, `interpretability.py`.

## Core conventions (leakage rules)

- **Proxy labels**: filename-derived, capture-level. Contamination is measured and
  reported, never used to relabel.
- **Sealed test sets**: the Phase 3 flow test split and the Phase 2 test alerts are
  never read during tuning or threshold selection; each is evaluated exactly once
  with a frozen configuration.
- **Train-only fitting**: imputation, scalers, encoders, and diagnostic quantile
  thresholds are fitted on the training partition only.
- **No identity features**: IPs, exact ports, MACs, timestamps, and capture
  filenames never enter a model matrix (`features.FORBIDDEN_IN_MATRIX`);
  the service-aware variant uses only universal port *semantics* (ranges,
  well-known service categories).
- **Leakage guard**: any flow reachable from a Phase 2 val/test alert is excluded
  from supervised training (`cascade_alignment.py` detects, `cascade_heads.py`
  remediates and retrains with locked hyperparameters).

## Headline result (sealed test)

Frozen cascade = multiclass service-aware head, Policy B (candidate-set
consensus), τ = 0.0166: **59.6 % FP-reduction at 92.7 % TP-retention**
(precision 0.070 → 0.148; recall ceiling preserved). Full numbers, CIs, and
breakdowns: `cascade_report.txt` and `data/cascade_results.json`.
