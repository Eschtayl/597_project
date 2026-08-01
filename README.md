# ECE 597 — Cascaded Network Intrusion Detection

Two-stage IDS on CIC IoT traffic:

1. **Phase 1** — preprocessing (load, sample, clean, scale).
2. **Phase 2** — unsupervised packet-level anomaly detection (Isolation Forest,
   autoencoder, ensemble / latent variants), tuned for recall.
3. **Phase 3** — supervised flow-level second stage that suppresses false
   positives from Phase 2. Cascade can only remove alerts:

   ```
   ŷ_cascade = ŷ_phase2 AND ŷ_phase3
   ```

More detail on Phase 3 is in `phase_3/README.md`.

---

## Data layout

CSVs are gitignored. Put them at the project root next to `helpers.py`:

```
flow_and_packet/          # preferred for Phase 3
  flow_based/             *.pcap_Flow.csv
  packet_based/           *.csv

# Phase 2 also works with the flatter layout:
packet_based/
flow_based/
```

Files starting with `Benign` are labelled benign; attack labels come from the
filename. The flow CSV `Label` column (`NeedManualLabel`) is not used as a target.

---

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Phase 1 — Preprocessing

`helpers.py` holds the shared pipeline; `PreprocessingPipeline.py` runs it.

Steps: load + label → sample (200k benign + 4–6.2k attack) → shuffle → handle
missing/infinite values → one-hot encode → drop identifiers → `log1p` +
`RobustScaler` (reuse the fitted scaler on held-out data).

```bash
# edit `path` in PreprocessingPipeline.py to switch packet_based / flow_based
python PreprocessingPipeline.py
```

---

## Phase 2 — Unsupervised packet detection

Code lives in `phase_2/`. Models train on benign rows only; labels are used for
threshold selection and evaluation. Higher anomaly score = more anomalous.

```bash
python phase_2/main.py eda
python phase_2/main.py if
python phase_2/main.py ae
python phase_2/main.py tune
python phase_2/main.py ensemble
python phase_2/main.py latent
python phase_2/main.py all
```

| Path | Role |
|------|------|
| `phase_2/config.py` | Seeds, paths, hyperparameters |
| `phase_2/data_prep.py` | Prep, split, threshold, alerts |
| `phase_2/models/` | Isolation Forest, autoencoder, ensemble |
| `phase_2/evaluation.py` | Metrics and figures → `saved_figs/` |
| `phase_2/main.py` | Entry point |

Results append to `phase_2/phase_2_results.txt` — delete it for a clean report.

---

## Phase 3 — Supervised flow-level FP reduction

Code lives in `phase_3/`. Builds unified logical flows, trains XGBoost heads,
maps Phase 2 alert packets to flows, then picks a cascade threshold on
validation alerts and evaluates once on the sealed test set.

Sealed-test result: **59.6% FP-reduction at 92.7% TP-retention**
(precision 0.070 → 0.148).

```bash
# checks
python phase_3/check_aggregation.py
python phase_3/check_mapping.py
python phase_3/check_sampling.py
python phase_3/check_features.py

# flow classifier
python phase_3/baselines.py
python phase_3/xgboost_model.py
python phase_3/weighting_comparison.py
python phase_3/heads.py

# cascade
python phase_3/phase2_cohort.py
python phase_3/cascade_alignment.py
python phase_3/cascade_heads.py
python phase_3/cascade_eval.py

# analysis
python phase_3/ablations.py
python phase_3/temporal_robustness.py
python phase_3/interpretability.py
```

| Area | Files |
|------|-------|
| Config & data | `config.py`, `flow_aggregation.py`, `flow_sampling.py`, `splits.py`, `features.py` |
| Mapping | `packet_flow_mapping.py`, `cascade_alignment.py` |
| Models | `baselines.py`, `xgboost_model.py`, `heads.py`, `cascade_heads.py` |
| Cascade | `phase2_cohort.py`, `cascade_eval.py` |
| Analysis | `ablations.py`, `temporal_robustness.py`, `interpretability.py` |

Reports are in `phase_3/*_report.txt`.
