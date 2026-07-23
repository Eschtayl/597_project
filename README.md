.csv files must be in a folder named "flow_based" or "packet_based" respectively.

The folders must be in the project root next to helpers.py and PreprocessingPipeline.py

## Phase 1
- `helpers.py` — load, sample, preprocess
- `PreprocessingPipeline.py` — run Phase 1 on flow or packet data

## Phase 2
All Phase 2 code lives in `phase_2/`:
- `config.py` — paths, seeds, model settings
- `data_prep.py` — packet prep, train/test split, threshold, alerts
- `models/isolation_forest.py` — Isolation Forest
- `models/autoencoder.py` — autoencoder
- `models/kmeans.py` — K-Means (PCA + cluster attack-purity scoring)
- `evaluation.py` — metrics and plots
- `eda.py` — EDA plots
- `main.py` — Phase 2 entry point
- `saved_figs/` — figures

```bash
source .venv/bin/activate
pip install -r requirements.txt
cd phase_2
python main.py eda
python main.py if
python main.py ae
python main.py kmeans
python main.py all
```

Or from the project root:

```bash
python phase_2/main.py eda
python phase_2/main.py if
python phase_2/main.py ae
python phase_2/main.py kmeans
python phase_2/main.py all
```
