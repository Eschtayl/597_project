# ECE 597 Multi-Stage Network Intrusion Detection

This repository implements a packet-to-flow intrusion-detection cascade. The
standard execution path is now:

1. **Phase 1 — behavioral packet preprocessing:** sample and label packets, clean missing
   and infinite values, remove identity fields, apply `log1p`, and robust-scale.
2. **Phase 2 — finalized Autoencoder:** train the tuned `(128, 64, 32)`
   Autoencoder on benign packets and generate packet-level anomaly alerts.
3. **Phase 3 — XGBoost:** map alerts to unified flows and use the finalized
   service-aware multiclass XGBoost cascade to suppress false positives.

The historical experiments are preserved. Isolation Forest, K-Means, score
fusion, Gradient Boosting, and neural-network implementations can still be run
independently, but they are not part of the default pipeline.

## Quick start

Create an environment and install the full project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the CSV files in the preferred layout:

```text
flow_and_packet/
├── packet_based/
│   └── *.csv
└── flow_based/
    └── *.pcap_Flow.csv
```

The root-level `packet_based/` and `flow_based/` layout is also supported when
using the default configuration generated in code.

Review `pipeline_config.json`, then run:

```powershell
python run_pipeline.py --dry-run
python run_pipeline.py
```

The dry run validates configuration and prints the exact stage sequence without
training. A full run writes cached data, models, scores, and evaluation JSON
under `phase_3/data/`; human-readable Phase 3 reports remain under `phase_3/`.

## Standard project structure

```text
ids_pipeline/
├── config.py          # typed configuration for all three phases
├── contracts.py       # small preprocessing/detector/classifier contracts
├── components.py      # preprocessing and finalized-AE adapters + registries
├── orchestrator.py    # ordered end-to-end execution
└── cli.py             # `python -m ids_pipeline`

phase_1/
├── helpers.py         # preserved Phase 1 function-based implementation
└── PreprocessingPipeline.py

phase_2/
├── data_prep.py       # shared splits, threshold selection, alert generation
└── models/
    └── autoencoder.py # finalized configurable AE implementation

phase_3/
├── phase2_cohort.py   # behavioral preprocessing + AE alerts with mapping IDs
├── flow_aggregation.py
├── flow_sampling.py
├── xgboost_model.py
├── cascade_alignment.py
├── cascade_heads.py
└── cascade_eval.py

phase_3_gb_nn/         # preserved earlier GB/NN experimental pipeline
pipeline_config.json   # default component and hyperparameter selection
run_pipeline.py        # primary entry point
```

## Configuration

`pipeline_config.json` centralizes:

- packet, flow, and artifact paths;
- preprocessing, feature-extractor/detector, and classifier names;
- random seed and sample sizes;
- Autoencoder architecture and training parameters;
- XGBoost tuning candidates and tuned-parameter reuse.

Paths are resolved relative to `project_root`. The default finalized components
are explicitly named:

```json
{
  "preprocessing": "behavioral_packet",
  "feature_extractor": "autoencoder",
  "classifier": "xgboost"
}
```

To use another configuration:

```powershell
python run_pipeline.py --config path/to/config.json
```

## Replacing a component

The component boundaries deliberately stay small:

- A preprocessor implements `PacketPreprocessor.prepare(...)`.
- A packet detector implements `AnomalyDetector.fit(...)` and `.score(...)`.
  Scores must use the shared convention: higher means more anomalous.
- A supervised stage implements `ClassifierStage.run(...)`.

Add a preprocessor or detector implementation and register its configuration
name in `ids_pipeline/components.py`. Add a supervised stage and register it in
`CLASSIFIER_STAGES` in `ids_pipeline/orchestrator.py`. Then select that name in
`pipeline_config.json`; the orchestration and downstream artifact contracts do
not need to change.

The Phase 3 boundary is the packet cohort schema:

```text
mapping identifiers + label + y_true + phase2_split + anomaly_score + y2_alert
```

Any replacement detector that produces this schema can use the existing flow
mapping, leakage guard, XGBoost heads, and cascade evaluation unchanged.

## Running legacy experiments

Legacy code is intentionally retained for comparison:

```powershell
# Phase 1 preprocessing only
python phase_1/PreprocessingPipeline.py

# Phase 2 experiments
python phase_2/main.py eda
python phase_2/main.py if
python phase_2/main.py ae
python phase_2/main.py tune
python phase_2/main.py ensemble
python phase_2/main.py latent
python phase_2/main.py kmeans

# Detailed Phase 3 checks and analyses
python phase_3/check_aggregation.py
python phase_3/check_mapping.py
python phase_3/check_sampling.py
python phase_3/check_features.py
python phase_3/temporal_robustness.py
python phase_3/interpretability.py
```

See `phase_3/README.md` for the detailed flow identity, leakage, threshold, and
cascade policies.

## Verification

Fast component tests do not require the full dataset:

```powershell
python -m pytest -q
python -m compileall -q ids_pipeline phase_1 phase_2 phase_3
```
