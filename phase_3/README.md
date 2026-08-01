# Phase 3 — Supervised Flow-Level IDS (False-Positive Reduction)

> **New here?** Start with [Mental model](#mental-model) and [Glossary](#glossary),
> then skim the [pipeline diagram](#end-to-end-pipeline). The run-order table
> further down is what you execute; the file map tells you what each module owns.

Second stage of the two-stage intrusion-detection system. Phase 2 (unsupervised,
packet-level, recall-tuned) flags anomalous packets; Phase 3 re-checks each flagged
packet against a supervised flow-level classifier and suppresses false positives
while retaining true detections. The cascade can only *remove* alerts, never add:

```
ŷ_cascade = ŷ_phase2 AND ŷ_phase3
```

Historical sealed-test result from the earlier Isolation Forest cohort:
**59.6 % FP-reduction at 92.7 % TP-retention** (precision 0.070 → 0.148).
Those archived numbers live in `cascade_report.txt` and
`data/cascade_results.json`. Run `python run_pipeline.py` to regenerate the
artifacts for the finalized Autoencoder cohort.

---

## Mental model

Think of Phase 3 as three layers stacked on top of Phase 2:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Flow identity & features                             │
│  Raw CICFlowMeter segments → LogicalFlowID → one row per flow   │
│  → behaviour-only / service-aware feature matrices              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  LAYER 2 — Supervised heads                                      │
│  XGBoost (multiclass or binary) scores each flow:                │
│  AttackScore = 1 − P(benign)   (or P(attack) for binary heads)   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  LAYER 3 — Cascade                                               │
│  Phase 2 alert packet → map to candidate flow(s) → score →       │
│  if score < τ suppress alert; else keep. No-match → keep.        │
└─────────────────────────────────────────────────────────────────┘
```

**Why two stages?** Phase 2 is cheap and high-recall but noisy (many false
positives). Phase 3 looks at the *conversation* (flow statistics) that the
packet belonged to, which is a much stronger signal for “is this really an
attack?” — but only for packets Phase 2 already flagged.

**Why flow-level, not packet-level again?** A single packet rarely carries
enough context (duration, byte ratios, flag counts, rates). A logical flow
aggregates that context across all segments of a conversation.

---

## Glossary

| Term | Meaning |
|------|---------|
| **LogicalFlowID** | One conversation = directional `Flow ID` + temporal cluster. Same 5-tuple reused hours later becomes a *new* logical flow once the gap exceeds 300 s. Format: `source_file\|FlowID#cluster`. |
| **Unified flow** | One aggregated row per `LogicalFlowID` (segments collapsed by per-feature rules in `flow_aggregation.py`). |
| **Proxy label** | Attack/benign label inferred from the **filename** (e.g. `DDoS-HTTP_Flood-…`). The CSV `Label` column is `NeedManualLabel` and is never used as a target or feature. |
| **Behaviour-only features** | CICFlowMeter stats + engineered ratios/logs. No IPs, ports, protocol, timestamps. The honest detector. |
| **Service-aware features** | Behaviour-only **plus** universal port *semantics* (HTTP/DNS flags, well-known/registered/ephemeral ranges, protocol). Never exact ports. |
| **Head** | One trained scorer: `{multiclass, binary} × {behaviour, service}` → four heads. |
| **AttackScore / s_*** | Higher = more attack-like. Multiclass: `1 − P(benign)`. Binary: `P(attack)`. |
| **Phase 2 alert** | Packet with `y2_alert == 1` from the finalized Autoencoder cohort. |
| **Mapping status** | How cleanly a packet maps to flow(s): `unique_match`, `multiple_same_capture`, `multiple_same_label`, `multiple_conflicting_labels`, `no_match`, `invalid_key`. |
| **Policy A** | Strict cascade: only `unique_match` may suppress; everything else retains the Phase 2 alert. |
| **Policy B** | Like A, but for `multiple_same_capture` suppresses only if *every* candidate flow scores below τ (consensus “benign”). |
| **τ (tau)** | Cascade threshold chosen on **validation alerts only**. Alert kept unless a usable score is finite and `< τ`. |
| **FP-reduction** | Fraction of Phase 2 false-positive alerts that Phase 3 suppresses. |
| **TP-retention** | Fraction of Phase 2 true-positive alerts that Phase 3 keeps. Floor was 95 % on val; sealed test landed at 92.7 %. |
| **Leakage guard** | Any flow reachable from a Phase 2 val/test alert must not sit in Phase 3 train (and test candidates must not sit in Phase 3 val). `cascade_heads.py` remediates by removing + resampling. |
| **Sealed test** | Test split / Phase 2 test alerts are never used for tuning, threshold selection, or model choice. Evaluated once with a frozen config. |

---

## End-to-end pipeline

```
flow CSVs                    packet CSVs
    │                             │
    ▼                             ▼
flow_aggregation.py          phase2_cohort.py
 (LogicalFlowID +               (AE alerts on
  segment aggregation)           packet sample)
    │                             │
    ▼                             │
flow_sampling.py                  │
 (200k benign + ~5k attack)       │
    │                             │
    ▼                             │
features.py + splits.py           │
 (matrices, train/val/test)       │
    │                             │
    ▼                             │
baselines → xgboost tune →        │
weighting bake-off → heads.py     │
    │                             │
    └────────────┬────────────────┘
                 ▼
        cascade_alignment.py
         (map alerts → candidate flows;
          leakage guard)
                 │
                 ▼
        cascade_heads.py
         (clean sample, retrain heads,
          score candidate flows)
                 │
                 ▼
        cascade_eval.py
         (τ sweep on val alerts;
          freeze winner; sealed test)
                 │
                 ▼
        ablations / temporal_robustness /
        interpretability  (analysis only)
```

---

## Data requirements

CSVs are gitignored. Place the CIC IoT captures under the project root:

```
flow_and_packet/
  flow_based/     *.pcap_Flow.csv   (CICFlowMeter flow statistics, 84 columns)
  packet_based/   *.csv             (per-packet features, 135 columns)
```

(`config.py` falls back to root-level `flow_based/` / `packet_based/` if
`flow_and_packet/` is absent.)

Important schema quirks (already measured — do not re-derive):

- Flow `Label` = `NeedManualLabel` for every row → **filename proxy labels only**.
- Flow ID is directional **5-field** (`SrcIP-DstIP-SrcPort-DstPort-Protocol`).
- Timestamps are `DD/MM/YYYY hh:MM:SS AM/PM` (second granularity).
- Packets have **no absolute timestamp** → mapper cannot disambiguate by time;
  ambiguity is classified, not resolved.

---

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
| 9 | `python phase_3/phase2_cohort.py` | Runs preprocessing + finalized Autoencoder packet alerts with a train/val/test split |
| 10 | `python phase_3/cascade_alignment.py` | Map val/test alert packets to the full 986k-flow index; run the leakage guard |
| 11 | `python phase_3/cascade_heads.py` | Remove leaking flows, retrain heads (locked params), score candidate flows |
| 12 | `python phase_3/cascade_eval.py` | Threshold sweep + Policy A/B bake-off on val alerts; single sealed-test evaluation |
| 13 | `python phase_3/ablations.py` | Model-progression ladder + engineered-feature ablation + overhead timing |
| 14 | `python phase_3/temporal_robustness.py` | Expanding-window temporal folds (robustness view) |
| 15 | `python phase_3/interpretability.py` | Gain/permutation importance, per-class SHAP, four cascade case studies |

Steps 1–4 are gates: do not train until they PASS. Steps 5–8 build the
standalone flow classifier. Steps 9–12 wire it into the cascade. Steps 13–15
are report/analysis only and assume the cascade artifacts already exist.

---

## File map

### Configuration & shared libraries (imported, not usually run directly)

| File | Responsibility |
|------|----------------|
| `config.py` | Paths, labels, frozen constants (gap cutoff, sample sizes, seed). Inserts project root on `sys.path`. |
| `flow_aggregation.py` | `LogicalFlowID` assignment + per-feature segment aggregation. `feature_dictionary()` documents every rule. |
| `packet_flow_mapping.py` | Canonical bidirectional 5-tuple key; maps packets → candidate flows; status taxonomy. |
| `flow_sampling.py` | 200k benign + ~4–6.2k attack sample over *unified* flows; parquet cache. |
| `splits.py` | Grouped stratified train/val/test (primary) and blocked temporal folds (robustness). |
| `features.py` | Behaviour-only / service-aware matrices, engineered features, diagnostics, train-only preprocessors. |
| `model_eval.py` | Shared multiclass + binary-collapse metrics (`AttackScore = 1 − P(benign)`). |

### Verification scripts (must PASS before modelling)

`check_aggregation.py`, `check_mapping.py`, `check_sampling.py`, `check_features.py`.

### Models & experiments

| File | Responsibility |
|------|----------------|
| `baselines.py` | Dummy / LR / RF on both feature variants (val only). |
| `xgboost_model.py` | Randomized hyperparameter search; locks `best_params` in `data/xgb_results.json`. |
| `weighting_comparison.py` | Pre-registered none/balanced/dampened bake-off → **none locked**. |
| `heads.py` | Trains the four unweighted heads; saves val scores (standalone view). |

### Cascade (the Phase 3 contribution)

| File | Responsibility |
|------|----------------|
| `phase2_cohort.py` | Memory-bounded packet sample + preprocessing + finalized Autoencoder → `phase2_cohort.parquet`. |
| `cascade_alignment.py` | Alert→flow alignment against the *full* flow index + leakage guard. |
| `cascade_heads.py` | Remediate leakage, retrain heads with fitted preprocessors, score candidates. |
| `cascade_eval.py` | τ + Policy A/B bake-off on val; freeze winner; sealed test metrics/CIs. |

### Analysis (report support)

`ablations.py`, `temporal_robustness.py`, `interpretability.py`.

---

## Cascade decision logic (detail)

For each Phase 2 **alert** packet:

1. Build a canonical 5-tuple key (endpoint-sorted, so direction does not matter).
2. Look up all logical flows sharing that key → **candidate set**.
3. Classify mapping status (unique / multi same-capture / … / no-match).
4. Under the chosen policy, either:
   - obtain a decision score `s` from the winning head, or
   - fall back to **retain** (score is NaN → alert stays).
5. Keep the alert unless `s` is finite **and** `s < τ`.

```
Policy A:  usable = unique_match only
Policy B:  usable = unique_match
                    OR (multiple_same_capture AND max(candidate scores) is used;
                        suppress only if that max < τ, i.e. ALL candidates look benign)
```

Conservative defaults by design:

- No match / invalid key / conflicting labels → **retain** Phase 2 alert.
- Missing prediction for a candidate → **retain**.
- Phase 3 never creates a new alert that Phase 2 did not raise.

---

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

---

## Key artifacts under `phase_3/data/` (gitignored)

| Path | Produced by | Contents |
|------|-------------|----------|
| `sampled_unified_flows.parquet` | `flow_sampling` / `check_sampling` | ~204k-row modelling sample |
| `unified_flows_full.parquet` | `cascade_alignment` | All ~986k unified flows (mapping index) |
| `phase2_cohort.parquet` | `phase2_cohort` | Packet rows with `y2_alert`, split, 5-tuple |
| `cascade_alignment.parquet` | `cascade_alignment` | One row per val/test alert + candidates |
| `cascade_leakage.json` | `cascade_alignment` | Guard verdict + overlapping LFIDs |
| `cascade_candidate_scores.parquet` | `cascade_heads` | Four head scores per candidate flow |
| `models/head_*_cascade.joblib` | `cascade_heads` | Fitted preprocessor + model per head |
| `cascade_results.json` | `cascade_eval` | Frozen config + val/test metrics |
| `xgb_results.json` | `xgboost_model` | Locked `best_params` for both variants |

Committed human-readable reports (`*_report.txt`) sit next to the scripts.

---

## Headline result (sealed test)

Frozen cascade = multiclass service-aware head, Policy B (candidate-set
consensus), τ = 0.0166: **59.6 % FP-reduction at 92.7 % TP-retention**
(precision 0.070 → 0.148; recall ceiling preserved). Full numbers, CIs, and
breakdowns: `cascade_report.txt` and `data/cascade_results.json`.

---

## Reading order for newcomers

If you are reviewing or extending this code without prior context:

1. This README (mental model + glossary).
2. `config.py` — every frozen constant and why it exists.
3. `flow_aggregation.py` — how a conversation becomes one row.
4. `features.py` — what the model is allowed to see.
5. `packet_flow_mapping.py` + `cascade_alignment.py` — how packets find flows.
6. `cascade_eval.py` — how τ / Policy A/B turn scores into suppress/keep.
7. The matching `*_report.txt` for any script whose numbers you need.
