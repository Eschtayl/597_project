"""
main.py
=======
Entry point for PHASE 3: Supervised Flow Classification
(Logistic Regression vs. Decision Tree, benchmarked against a Dummy
baseline, on top of Taylor's shared Phase 1 preprocessing).

PHASE NUMBERING (see config.py's docstring for full context):
    Phase 1 -> data preprocessing (Taylor)          [taylor_pipeline.py]
    Phase 2 -> unsupervised anomaly detection         (Isolation Forest,
               Autoencoder -- not in this pipeline)
    Phase 3 -> supervised flow classification         [THIS pipeline]

RUN THIS WITH:  python main.py

To get the multiclass (macro-F1) view comparable to teammate Vansh's
Phase 3 results, set config.TASK_MODE = "multiclass" and re-run.
"""

import os

import pandas as pd

import config
import evaluate


def save_outputs(results, robustness_summary, ablation_summary):
    """Persist metrics and feature importances to disk so they can be
    dropped straight into a report or slide deck, and so you have a
    record of exactly what a given run produced."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc",
               "train_time_sec", "infer_time_sec"]
    summary = pd.DataFrame({
        "Dummy (baseline)": {m: results["dummy"].get(m, float("nan")) for m in metrics},
        "Logistic Regression": {m: results["logistic_regression"][m] for m in metrics},
        "Decision Tree": {m: results["decision_tree"][m] for m in metrics},
        "AdaBoost": {m: results["adaboost"][m] for m in metrics},
        "Gradient Boosting": {m: results["gradient_boosting"][m] for m in metrics},
    }).T
    summary.to_csv(os.path.join(
        config.OUTPUT_DIR, f"phase3_primary_run_metrics_{config.TASK_MODE}.csv"
    ))

    for name, key in [("logistic_regression", "logistic_regression"),
                       ("decision_tree", "decision_tree"),
                       ("adaboost", "adaboost"),
                       ("gradient_boosting", "gradient_boosting")]:
        results[key]["top_features"].to_csv(
            os.path.join(config.OUTPUT_DIR, f"phase3_{name}_top_features_{config.TASK_MODE}.csv"),
            header=["importance_or_coefficient"],
        )

    robustness_summary.to_csv(
        os.path.join(config.OUTPUT_DIR, f"phase3_robustness_check_{config.TASK_MODE}.csv"),
        index=False,
    )
    ablation_summary.to_csv(
        os.path.join(config.OUTPUT_DIR, f"phase3_class_weight_ablation_{config.TASK_MODE}.csv"),
        index=False,
    )

    print(f"\nAll Phase 3 outputs saved to: {config.OUTPUT_DIR}")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"# {config.PHASE_LABEL}: SUPERVISED FLOW CLASSIFICATION")
    print(f"# Pipeline: Taylor's Phase 1 preprocessing -> "
          f"Dummy / Logistic Regression / Decision Tree")
    print(f"{'#' * 60}")

    # ---- Primary experiment (the numbers you report as "the result") ----
    results = evaluate.run_single_experiment(seed=config.SEED, pipeline="taylor")
    evaluate.print_comparison_report(results)

    # ---- Robustness check (proves the result isn't a lucky sample) ----
    print("\n\nRunning multi-seed robustness check -- this re-runs the "
          "entire pipeline several times and will take a bit longer...")
    robustness_summary = evaluate.run_robustness_check()

    # ---- Class-weight ablation (tests the assumption, doesn't presume it) ----
    print("\n\nRunning class-weight ablation (balanced vs. unweighted)...")
    ablation_summary = evaluate.run_class_weight_ablation()

    # ---- Persist everything ----
    save_outputs(results, robustness_summary, ablation_summary)

    # ---- Explicit scope-adjustment notes for your report ----
    print("\n" + "=" * 60)
    print("SCOPE NOTES FOR YOUR REPORT")
    print("=" * 60)
    print(
        f"1. Phase numbering: this is PHASE 3 (supervised flow classification), "
        f"not Phase 2 (which is unsupervised anomaly detection -- Isolation "
        f"Forest / Autoencoder -- per the team repo's structure).\n\n"
        f"2. Benign sample size: Phase 1's spec (Taylor's helpers.py) requires "
        f"exactly 200,000 benign rows, hard-enforced via ValueError. The real "
        f"benign pool currently available to the team is ~44,895 rows -- a data "
        f"availability blocker, not a code issue. This run scope-adjusts to "
        f"{config.BENIGN_TARGET:,} benign rows (see taylor_pipeline.py's "
        f"benign_sampler_scope_adjusted for the documented exception).\n\n"
        f"3. Side effect of the above: Taylor's attack_sampler independently "
        f"draws 4,000-6,200 attack rows regardless of benign count. With "
        f"benign scope-adjusted down to {config.BENIGN_TARGET:,}, the "
        f"resulting benign/attack ratio is closer to ~90/10 than the ~97/3 "
        f"originally intended -- this is an honest consequence of the real "
        f"data shortfall, not a further silent adjustment, and is worth "
        f"stating explicitly if asked."
    )


if __name__ == "__main__":
    main()
