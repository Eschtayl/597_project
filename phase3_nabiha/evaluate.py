"""
evaluate.py
===========
Everything related to measuring how good the models actually are, and
whether that goodness is a stable pattern or a fluke of one random sample
or an artifact of leakage.

WHY NOT JUST REPORT ACCURACY?
Our dataset is ~97% benign, ~3% attack. A model that predicts "benign"
for every single row, without looking at any feature, would score ~97%
accuracy while being completely useless -- it would catch zero attacks.
This is exactly the trap accuracy sets for imbalanced classification
problems, and exactly why we report precision, recall, F1, and ROC-AUC
alongside it (and why we include a Dummy baseline below to make this
concrete rather than just asserted).

BINARY vs MULTICLASS METRICS
- Binary mode: precision/recall/F1 are computed for the positive
  ("Attack") class specifically.
- Multiclass mode: we report macro-averaged precision/recall/F1 --
  computed per class, then averaged with equal weight per class
  regardless of how many rows that class has. This matches teammate
  Vansh's Phase 3 reporting (macro-F1), making the two directly
  comparable.

GROUP-AWARE SPLITTING
A plain random train/test split can place some flows from a given
source IP into training and others into testing. A model could then
partly learn "this specific host's behavior" rather than a
generalizable attack pattern -- inflating test performance in a way
that would not hold up on genuinely new traffic. We use GroupShuffleSplit
so every flow from a given IP (config.GROUP_COL) stays entirely on one
side of the split.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import config
import data_loader
import models
import preprocessing
import taylor_pipeline


def _group_aware_split(X, y, groups, seed):
    """
    Split into train/test while keeping every row that shares the same
    `groups` value (e.g. Src IP) entirely on one side of the split.

    NOTE ON STRATIFICATION TRADEOFF
    GroupShuffleSplit does NOT guarantee the same class balance in train
    vs. test the way stratified splitting does -- it can only guarantee
    group separation. With ~97/3 imbalance and enough distinct IPs per
    class (verified before this was implemented: our smallest attack
    category still has 40+ distinct Src IPs), this is a reasonable
    tradeoff, but it's worth spot-checking the resulting train/test
    attack rate (printed below) each run to confirm neither split
    ended up with an unreasonably skewed class balance by chance.

    Falls back to a plain stratified split if `groups` is unavailable
    (e.g. if Src IP was missing from the data), printing a warning so
    the gap is visible rather than silently skipped.
    """
    if groups is None:
        print("  [WARNING] No group column available -- falling back to "
              "plain stratified split (group-leakage risk not addressed)")
        return train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=seed,
            stratify=y if config.STRATIFY else None,
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups & test_groups
    if overlap:
        # Should never happen with GroupShuffleSplit, but verify anyway --
        # an assumption worth checking rather than trusting blindly.
        print(f"  [ERROR] {len(overlap)} groups leaked across train/test!")
    else:
        print(f"  [OK] Group-aware split confirmed: 0 overlapping "
              f"{config.GROUP_COL} values between train and test "
              f"({len(train_groups)} train groups, {len(test_groups)} test groups)")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, needs_scaling=False, scaler=None):
    """
    Run a fitted model against the held-out test set and compute the
    full metric suite plus a confusion matrix. Automatically adapts
    metric averaging to config.TASK_MODE (binary vs. multiclass).
    """
    X_eval = scaler.transform(X_test) if needs_scaling else X_test

    t0 = time.time()
    y_pred = model.predict(X_eval)
    infer_time_sec = time.time() - t0

    is_multiclass = config.TASK_MODE == "multiclass"
    average = "macro" if is_multiclass else "binary"
    pos_label = None if is_multiclass else 1

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average,
                                      pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average,
                                pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average,
                        pos_label=pos_label, zero_division=0),
        "infer_time_sec": infer_time_sec,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }

    # ROC-AUC needs predict_proba; guard against models/edge cases where
    # it's undefined (e.g. a Dummy classifier with constant predictions
    # can make ROC-AUC mathematically undefined for a degenerate split).
    try:
        y_proba = model.predict_proba(X_eval)
        if is_multiclass:
            metrics["roc_auc"] = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro"
            )
        else:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        metrics["y_proba"] = y_proba
    except (ValueError, AttributeError) as e:
        metrics["roc_auc"] = float("nan")
        print(f"  [NOTE] ROC-AUC undefined for this model/split: {e}")

    return metrics


def run_single_experiment(seed=None, verbose=True, class_weight_override=None,
                           pipeline="taylor"):
    """
    Run the FULL pipeline once, end to end, for a single seed:
    build dataset -> preprocess -> group-aware split -> train all models
    (Dummy, Logistic Regression, Decision Tree) -> evaluate.

    Parameters
    ----------
    class_weight_override : str, dict, "no_weight", or None
        Passed through to model training. "no_weight" explicitly forces
        class_weight=None (used by the ablation check); None means "use
        config.CLASS_WEIGHT as normal".
    pipeline : "taylor" or "legacy"
        "taylor" (default): uses taylor_pipeline.py -- the team's shared
        Phase 1 preprocessing (log1p + RobustScaler, ported from Taylor's
        repo). This is the current, primary path.
        "legacy": uses our own original data_loader.py/preprocessing.py
        (pre-Taylor-integration). Kept only for comparison/debugging --
        do NOT use this for reported results, since it predates the
        team's shared preprocessing standard.

    This function is the single source of truth for "how do we run one
    experiment" -- main.py calls it for the primary report, and both
    run_robustness_check() and run_class_weight_ablation() below call it
    repeatedly with different seeds/settings, so all experiment variants
    can never silently drift out of sync with each other.
    """
    active_seed = seed if seed is not None else config.SEED
    if verbose:
        print(f"\n{'=' * 60}\nRUNNING EXPERIMENT (seed = {active_seed}, "
              f"mode = {config.TASK_MODE}, pipeline = {pipeline})\n{'=' * 60}")

    if pipeline == "taylor":
        df_final, taylor_scaler = taylor_pipeline.build_preprocessed_dataset(seed=active_seed)
        X, y, labels_raw, groups = preprocessing.preprocess_taylor(df_final)
        lr_already_scaled = True
    elif pipeline == "legacy":
        df = data_loader.build_dataset(seed=active_seed)
        X, y, labels_raw, groups = preprocessing.preprocess(df)
        lr_already_scaled = False
    else:
        raise ValueError(f"Unknown pipeline: {pipeline!r}")

    X_train, X_test, y_train, y_test = _group_aware_split(X, y, groups, active_seed)

    if verbose:
        print(f"\nTrain shape: {X_train.shape}  Test shape: {X_test.shape}")
        if config.TASK_MODE == "binary":
            print(f"Train attack rate: {y_train.mean():.4f}  "
                  f"Test attack rate: {y_test.mean():.4f}")
        else:
            print("Train class counts:\n", y_train.value_counts())
            print("Test class counts:\n", y_test.value_counts())

    if class_weight_override is None:
        cw = models._UNSET          # no override requested -> use config default
    elif class_weight_override == "no_weight":
        cw = None                    # explicit request for NO weighting
    else:
        cw = class_weight_override   # explicit override value (e.g. "balanced")

    # --- Dummy baseline ---
    dummy_model, dummy_train_time = models.train_dummy_classifier(X_train, y_train)
    dummy_results = evaluate_model(dummy_model, X_test, y_test, needs_scaling=False)
    dummy_results["train_time_sec"] = dummy_train_time

    # --- Logistic Regression ---
    lr_model, lr_scaler, lr_train_time = models.train_logistic_regression(
        X_train, y_train, class_weight=cw, already_scaled=lr_already_scaled
    )
    lr_results = evaluate_model(
        lr_model, X_test, y_test,
        needs_scaling=(lr_scaler is not None), scaler=lr_scaler
    )
    lr_results["train_time_sec"] = lr_train_time
    lr_results["top_features"] = models.get_logistic_regression_coefficients(
        lr_model, X.columns
    )

    # --- Decision Tree ---
    dt_model, dt_train_time = models.train_decision_tree(
        X_train, y_train, class_weight=cw
    )
    dt_results = evaluate_model(dt_model, X_test, y_test, needs_scaling=False)
    dt_results["train_time_sec"] = dt_train_time
    dt_results["top_features"] = models.get_decision_tree_importances(
        dt_model, X.columns
    )

    # --- AdaBoost ---
    ada_model, ada_train_time = models.train_adaboost(X_train, y_train, class_weight=cw)
    ada_results = evaluate_model(ada_model, X_test, y_test, needs_scaling=False)
    ada_results["train_time_sec"] = ada_train_time
    ada_results["top_features"] = models.get_feature_importances(ada_model, X.columns)

    # --- Gradient Boosting ---
    gb_model, gb_train_time = models.train_gradient_boosting(X_train, y_train, class_weight=cw)
    gb_results = evaluate_model(gb_model, X_test, y_test, needs_scaling=False)
    gb_results["train_time_sec"] = gb_train_time
    gb_results["top_features"] = models.get_feature_importances(gb_model, X.columns)

    return {
        "seed": active_seed,
        "pipeline": pipeline,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "dummy": {"model": dummy_model, **dummy_results},
        "logistic_regression": {"model": lr_model, "scaler": lr_scaler, **lr_results},
        "decision_tree": {"model": dt_model, **dt_results},
        "adaboost": {"model": ada_model, **ada_results},
        "gradient_boosting": {"model": gb_model, **gb_results},
    }


def print_comparison_report(results):
    """Pretty-print the metric table, confusion matrices, and top features
    for a single experiment's results (as returned by run_single_experiment)."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc",
               "train_time_sec", "infer_time_sec"]

    summary = pd.DataFrame({
        "Dummy (baseline)": {m: results["dummy"].get(m, float("nan")) for m in metrics},
        "Logistic Regression": {m: results["logistic_regression"][m] for m in metrics},
        "Decision Tree": {m: results["decision_tree"][m] for m in metrics},
        "AdaBoost": {m: results["adaboost"][m] for m in metrics},
        "Gradient Boosting": {m: results["gradient_boosting"][m] for m in metrics},
    }).T

    print("\n" + "=" * 60)
    print(f"MODEL COMPARISON SUMMARY (mode = {config.TASK_MODE})")
    print("=" * 60)
    print(summary.round(4).to_string())

    for name, key in [("Logistic Regression", "logistic_regression"),
                       ("Decision Tree", "decision_tree"),
                       ("AdaBoost", "adaboost"),
                       ("Gradient Boosting", "gradient_boosting")]:
        r = results[key]
        print(f"\n--- {name}: Confusion Matrix ---")
        print(r["confusion_matrix"])
        print(f"\n--- {name}: Top influential features ---")
        print(r["top_features"])

    return summary


def run_robustness_check(seeds=None):
    """
    Re-run the entire pipeline once per seed in `seeds` and report the
    mean +/- standard deviation of each metric across runs.

    WHY THIS MATTERS
    A single run's numbers could be an accident of which specific rows
    got randomly sampled into the dataset, and which specific IPs'
    groups landed in the test split. If Logistic Regression "beats"
    Decision Tree by 0.02 F1 in one run but the gap flips sign in
    another, that tells you the difference isn't a real, stable pattern
    -- it's noise. Reporting mean +/- std across multiple seeds (as the
    professor's spec explicitly asks for) is what lets you make a
    defensible claim like "Model A reliably outperforms Model B" rather
    than "Model A happened to score higher once."
    """
    seeds = seeds if seeds is not None else config.ROBUSTNESS_SEEDS
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    model_keys = [
        ("Logistic Regression", "logistic_regression"),
        ("Decision Tree", "decision_tree"),
        ("AdaBoost", "adaboost"),
        ("Gradient Boosting", "gradient_boosting"),
    ]
    records = {name: [] for name, _ in model_keys}

    for seed in seeds:
        results = run_single_experiment(seed=seed, verbose=False)
        for name, key in model_keys:
            row = {m: results[key][m] for m in metrics}
            row["seed"] = seed
            records[name].append(row)
        print(f"  Completed seed {seed}")

    print("\n" + "=" * 60)
    print(f"ROBUSTNESS CHECK ACROSS {len(seeds)} SEEDS: {seeds}")
    print("=" * 60)

    summary_rows = []
    for name in records:
        df = pd.DataFrame(records[name]).set_index("seed")
        print(f"\n--- {name}: per-seed results ---")
        print(df.round(4).to_string())
        mean_std = pd.DataFrame({"mean": df.mean(), "std": df.std()})
        print(f"\n--- {name}: mean +/- std across seeds ---")
        print(mean_std.round(4).to_string())
        for m in metrics:
            summary_rows.append({
                "model": name, "metric": m,
                "mean": mean_std.loc[m, "mean"],
                "std": mean_std.loc[m, "std"],
            })

    return pd.DataFrame(summary_rows)


def run_class_weight_ablation(seed=None):
    """
    Run each model BOTH with class_weight="balanced" and with
    class_weight=None, and report both -- rather than assuming balanced
    weighting is the right call.

    WHY THIS MATTERS
    Teammate Vansh found the opposite of the "obvious" assumption on his
    data: "no class weighting works better than weighting it." Class
    weighting is meant to help a model pay attention to a rare class,
    but it can also push a model toward too many false positives if
    overcorrected -- whether it helps is an empirical question for YOUR
    specific data and models, not something to assume from general
    imbalanced-learning advice. This function tests it directly.
    """
    active_seed = seed if seed is not None else config.SEED
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    model_keys = [
        ("Logistic Regression", "logistic_regression"),
        ("Decision Tree", "decision_tree"),
        ("AdaBoost", "adaboost"),
        ("Gradient Boosting", "gradient_boosting"),
    ]

    rows = []
    for weight_setting, cw in [("balanced", "balanced"), ("no_weight", "no_weight")]:
        results = run_single_experiment(
            seed=active_seed, verbose=False, class_weight_override=cw
        )
        for name, key in model_keys:
            row = {m: results[key][m] for m in metrics}
            row["model"] = name
            row["weighting"] = weight_setting
            rows.append(row)

    ablation_df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("CLASS WEIGHT ABLATION (balanced vs. no weighting)")
    print("=" * 60)
    print(ablation_df.round(4).to_string(index=False))

    return ablation_df
