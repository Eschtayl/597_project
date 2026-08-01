# Phase 3 — Supervised Flow Classification
### Logistic Regression, Decision Tree, AdaBoost, and Gradient Boosting on Taylor's shared Phase 1 preprocessing

## Recent addition: two ensemble models

Random Forest and XGBoost were already claimed by teammates, so two
different ensemble strategies were added instead:

- **AdaBoost**: trains many very shallow "stump" trees (depth 1) in
  sequence, reweighting misclassified rows after each one so the next
  stump focuses on current mistakes. Final prediction is a weighted
  vote across all stumps.
- **Gradient Boosting**: also trains trees sequentially, but each new
  tree predicts the *residual error* of the ensemble so far, gradually
  correcting it. This is the same algorithmic family XGBoost belongs
  to, without XGBoost's specific engineering optimizations -- so
  comparing your Gradient Boosting result to Vansh's XGBoost result
  shows how much those optimizations actually add.

**A real, documented limitation**: neither model supports
`class_weight` the way Logistic Regression and Decision Tree do.
- **AdaBoost**: combining `class_weight="balanced"` on its base stump
  with AdaBoost's own reweighting mechanism caused a hard failure
  during testing (`"BaseClassifier in AdaBoostClassifier ensemble is
  worse than random, ensemble can not be fit"`) -- a known,
  reproducible interaction, not a bug. AdaBoost is run **unweighted**,
  relying entirely on its own built-in reweighting to attend to the
  rare class. State this explicitly if asked, rather than presenting
  it as an oversight.
- **Gradient Boosting**: has no `class_weight` parameter at all;
  weighting is approximated via per-row `sample_weight`, computed the
  same way sklearn's `"balanced"` setting would internally.

All four models now appear in every report: the primary run, the
5-seed robustness check, and the class-weight ablation (AdaBoost
appears in the ablation output too, for comparison, even though its
own weighting doesn't actually change -- see the printed `[NOTE]` in
its output).

**Runtime note**: with four models across 8 total experiment runs
(1 primary + 5 robustness seeds + 2 ablation settings), a full
`python main.py` run takes a few minutes -- longer than the earlier
two-model version. This is expected, not a hang.



## Phase numbering (please read first)

Per the team repo (`github.com/Eschtayl/597_project`), the correct project
structure is:

| Phase | What it is | Owner |
|---|---|---|
| Phase 1 | Data preprocessing (load, sample, clean, encode, scale) | Taylor |
| Phase 2 | Unsupervised anomaly detection (Isolation Forest, Autoencoder) | — |
| Phase 3 | **Supervised flow classification** (this work + Vansh's Dummy/LR/RF/XGBoost) | You + Vansh |

Earlier drafts of this work called it "Phase 2" — that was incorrect and
has been corrected throughout this pipeline and its outputs. Make sure
your slides say **Phase 3**.

## What this is

A modular, documented Phase 3 pipeline that:
1. Runs Taylor's actual Phase 1 preprocessing code (ported, not
   re-implemented, from his repo) to produce a shared, team-consistent
   feature set.
2. Trains and compares three models on that output: a Dummy baseline,
   Logistic Regression, and a Decision Tree.
3. Evaluates with a leakage-safe, group-aware train/test split.
4. Checks robustness across multiple seeds and runs a class-weight
   ablation, rather than assuming any single run or setting is "the" answer.

## File structure

```
phase2_pipeline/            (folder name predates the Phase 3 correction;
│                             contents are Phase 3)
├── config.py               All settings: seeds, paths, task mode, model
│                            hyperparameters, group column, scope-exception
│                            values. Change values here, not in logic files.
├── taylor_pipeline.py       Ported from Taylor's helpers.py: load, label,
│                            sample, clean, encode, log-transform, scale.
│                            This is the CURRENT, primary preprocessing path.
├── data_loader.py           LEGACY: our own preprocessing, built before
│                            Taylor's code was available. Kept for reference
│                            only — do not use for reported results.
├── preprocessing.py         Adapters: preprocess_taylor() (current) reads
│                            Taylor's output; preprocess() (legacy) reads
│                            data_loader.py's output.
├── models.py                Training functions for Dummy, Logistic
│                            Regression (scaling-aware — skips re-scaling
│                            Taylor's already-scaled features), and
│                            Decision Tree.
├── evaluate.py               Metric computation, group-aware split,
│                            multi-seed robustness check, class-weight
│                            ablation. pipeline="taylor" is the default.
├── main.py                  Runs everything end to end, saves results.
└── flow_based/               Folder Taylor's loaders expect: real,
                             flow-level CSVs ONLY (not the raw packet-level
                             files, which have a different schema).
```

## How to run

1. Create a `flow_based/` folder next to these files (or point
   `config.TAYLOR_FLOW_DIR` elsewhere) containing only the real
   flow-level CSVs — files with names starting `Benign...` for benign
   traffic, and attack files whose names contain one of: `DDoS-HTTP`,
   `DoS-HTTP`, `Spoofing`, `XSS`, `BruteForce`.
2. Run:
   ```bash
   python main.py
   ```
3. Results (metrics including a Dummy baseline, top features per model,
   robustness check, class-weight ablation) print to console and save
   as CSVs in `config.OUTPUT_DIR`, all prefixed `phase3_`.
4. For the multiclass (macro-F1) view directly comparable to Vansh's
   reporting, set `config.TASK_MODE = "multiclass"` and re-run.

## The pipeline, step by step

### 1. Taylor's Phase 1 preprocessing (`taylor_pipeline.py`)
Ported near-verbatim from `helpers.py` in the team repo, so features are
byte-for-byte consistent with Taylor's (and by extension, comparable to
Vansh's Phase 3 work built on the same foundation):
- **Load & label**: benign files (matched by filename prefix `Benign`)
  and attack files (matched by filename substring — `DDoS-HTTP`,
  `DoS-HTTP`, `Spoofing`, `XSS`, `BruteForce`) are pooled and labeled.
- **Sample**: benign and attack rows are drawn per Phase 1's spec —
  **with one documented exception, below**.
- **Shuffle & segregate**: split into features and labels.
- **Handle missing data**: `-1` placeholders and infinities are treated
  as missing; numeric NaNs filled with median, categorical with
  `'unknown'`; `_is_missing` / `_is_infinite` indicator columns preserve
  *where* imputation happened, rather than silently hiding it.
- **One-hot encode**: packet-level categorical columns (`http_request_method`,
  etc.) — a safe no-op on our flow-level data, since those columns don't exist here.
- **Split identifiers**: keyword-based exclusion of IP, port, MAC,
  timestamp, protocol, and other non-behavioral columns from the
  feature set. Notably, **this excludes Src/Dst Port and Protocol** —
  fixing a gap in our earlier, pre-Taylor pipeline, where ports were
  left in as numeric features (flagged then as a leakage-adjacent risk).
- **Log + scale**: `log1p(abs(x))` compresses heavy-tailed flow features
  (byte counts, durations), then `RobustScaler` (median/IQR-based, more
  outlier-resistant than mean/std-based `StandardScaler`) scales everything.

### 2. THE DOCUMENTED SCOPE EXCEPTION — benign sample size

**Taylor's original code hard-requires exactly 200,000 benign rows**
(`benign_sampler_200k` raises `ValueError` below that). The real benign
pool currently available to this team (pooled across 4 files) is
**~44,895 rows** — a genuine data availability shortfall, not a bug in
anyone's code.

`taylor_pipeline.benign_sampler_scope_adjusted` replaces the hard
200,000 requirement with `config.BENIGN_TARGET` (currently 40,000) as a
ceiling, and **prints this adjustment explicitly every run** — never a
silent substitution. State this exactly this way in your report:

> "Phase 1 specifies 200,000 benign rows; the team's currently available
> real benign data supports at most ~44,895, so this run scope-adjusts
> to 40,000."

**Side effect worth flagging too**: Taylor's `attack_sampler` is left
completely unmodified and independently draws 4,000–6,200 attack rows
regardless of benign count. With benign scope-adjusted down to 40,000
(rather than 200,000), the resulting ratio is closer to **~90/10**
benign/attack rather than the originally-intended ~97/3. This is an
honest consequence of the real shortfall — not a further adjustment —
and is worth mentioning if asked, since it means this run's class
imbalance is milder than the original spec envisioned.

### 3. Group-aware split (`evaluate.py`)
Every flow from a given `Src IP` stays entirely in train or entirely in
test (`GroupShuffleSplit`), preventing a model from partially learning
"this specific host's behavior" instead of a generalizable attack
pattern. Confirmed every run via an explicit zero-overlap check printed
to console.

### 4. Models (`models.py`)
- **Dummy baseline**: predicts the majority class, ignoring all
  features — makes concrete how much real signal the other two models
  add, or don't.
- **Logistic Regression**: trained directly on Taylor's already-scaled
  features (no redundant re-scaling) when using the Taylor pipeline.
- **Decision Tree**: `max_depth=8`, `min_samples_leaf=50` to limit
  overfitting.
- Both support `class_weight="balanced"` or unweighted — see the
  ablation below for why this isn't assumed.

### 5. Multi-seed robustness check & class-weight ablation (`evaluate.py`)
- Re-runs the full pipeline across 5 seeds, reporting mean ± std per
  metric — tells you whether a result is a stable pattern or a fluke.
- Runs each model with **both** `class_weight="balanced"` and
  unweighted, and reports both — rather than assuming balanced
  weighting is always correct.

## Key finding from the class-weight ablation

**Unweighted Logistic Regression substantially outperformed the
balanced version** on this run (precision 0.70 vs. 0.10, accuracy 0.94
vs. 0.44) — matching what teammate Vansh independently found on his own
models ("no class weighting works better than weighting it"). This is a
genuinely corroborated, cross-validated finding across two independent
pipelines and worth highlighting in your presentation as a point of
agreement with your teammate's results.

## Known limitations / discussion points for your report

1. **Benign sample size scope exception** (detailed above) — states
   itself in console output every run; not a silent deviation.
2. **Resulting ~90/10 ratio** rather than ~97/3, as a side effect of #1.
3. **High run-to-run variance**: the robustness check shows some
   metrics swinging widely across seeds (e.g. Logistic Regression F1
   ranging 0.13–0.57) — with only ~1,200–1,240 rows per attack category,
   individual seeds can land on meaningfully different, small samples.
   This variance is itself worth reporting, not hidden behind a single
   seed's number.
4. **Taylor's scaler is fit before the train/test split** (on the full
   sampled dataset), which is a minor leakage risk in the original
   Phase 1 code — not something this pipeline attempts to silently fix,
   since doing so would break consistency with the shared preprocessing
   standard, but worth flagging as a discussion point if asked.
5. **Binary framing** — set `config.TASK_MODE = "multiclass"` for the
   per-category view directly comparable to Vansh's macro-F1 reporting.

## Connecting to Phase 2 → Phase 3 (per your professor's cascade design)

Once a final Phase 3 model is chosen, its `predict_proba()` output
becomes an input signal alongside Phase 2's unsupervised alerts. You'll
need to pick a probability **threshold** balancing recall (catching
attacks) against precision (avoiding false alarms) — typically chosen
via a precision-recall curve.
