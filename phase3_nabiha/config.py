"""
config.py
=========
All tunable settings for the Phase 3 pipeline live here, in one place,
so you can change an assumption once and have it propagate everywhere
instead of hunting through multiple files.

PHASE NUMBERING CORRECTION
Earlier work in this project referred to this stage as "Phase 2." After
reviewing the team's actual repo structure (github.com/Eschtayl/597_project),
the correct numbering is:
    Phase 1 -> data preprocessing (Taylor)
    Phase 2 -> unsupervised anomaly detection: Isolation Forest, Autoencoder
    Phase 3 -> supervised flow classification (THIS work: Logistic
               Regression vs. Decision Tree, alongside Vansh's
               Dummy/LR/Random Forest/XGBoost comparison)
Everything in this pipeline and its outputs now refers to "Phase 3."

WHY A SEPARATE CONFIG FILE?
In a real ML pipeline, "which files go where" and "how much do we sample"
are decisions you will revisit constantly during development (new data
arrives, your professor changes the spec, you want to test a different
seed). Keeping them out of the logic files means you never have to touch
tested code just to change a number.
"""

import os

PHASE_LABEL = "Phase 3"

# ============================================================
# REPRODUCIBILITY
# ============================================================
# A "seed" initializes the random number generator to a fixed starting
# point. Every operation that involves randomness (sampling rows,
# shuffling, splitting train/test, initializing model weights) will
# produce IDENTICAL results every time you re-run the code with the
# same seed. This matters for two reasons:
#   1. You can reproduce your own results exactly (important for
#      debugging: if a number changes between runs, you know it's a
#      real code change, not random noise).
#   2. Your professor's spec explicitly asks for a *configurable* seed
#      so you can test whether your results are stable across different
#      random samples of the data, not a fluke of one lucky draw.
#
# CHANGE THIS to any integer to get a different random composition of
# the dataset (different rows sampled), while keeping everything else
# (ratios, categories, logic) identical.
SEED = 42

# ============================================================
# DATA LOCATIONS
# ============================================================
# Where your raw flow-based CSV files live. In Colab this might be
# "/content" (direct upload) or "/content/drive/MyDrive/your_folder"
# (Google Drive). Locally, this is wherever you've placed the files.
DATA_DIR = "C:/Users/nabih/OneDrive/Desktop/phase2_pipeline"

# Taylor's Phase 1 loaders (taylor_pipeline.py) expect a dedicated folder
# containing ONLY real flow-level CSVs -- per his repo's README convention
# ("csv files must be in a folder named 'flow_based' or 'packet_based'").
# This must NOT include the raw packet-level files (e.g. the original
# XSS.csv, DoS-HTTP_Flood.csv, DictionaryBruteForce.csv) -- those have an
# entirely different schema and would corrupt the pooled load.
TAYLOR_FLOW_DIR = os.path.join(DATA_DIR, "flow_based")

# Where processed outputs (metrics, plots, saved datasets) get written.
OUTPUT_DIR = "C:/Users/nabih/OneDrive/Desktop/phase2_pipeline/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SAMPLING TARGETS
# ============================================================
# ORIGINAL SPEC (Phase 1, Taylor's helpers.py):
#   Exactly 200,000 benign rows (hard-enforced in his original code).
#   4,000-6,200 attack rows, split as evenly as possible across 5
#   categories (handled by taylor_pipeline.attack_sampler, unmodified).
#
# SCOPE EXCEPTION (documented blocker, not a silent substitution):
# The real benign pool currently available to this team (pooled across
# 4 files) is ~44,895 rows -- well short of 200,000. This is a data
# availability gap, not a code bug. taylor_pipeline.benign_sampler_
# scope_adjusted uses BENIGN_TARGET below as a ceiling instead of the
# hard 200,000 requirement, and prints this adjustment explicitly every
# run. State this exactly this way in your report: "Phase 1 specifies
# 200,000 benign rows; our team's currently available real data supports
# at most ~44,895, so this run scope-adjusts to that ceiling."
BENIGN_TARGET = 40_000

# NOTE: attack sampling is now handled directly by
# taylor_pipeline.attack_sampler (ported verbatim from Taylor's code),
# which draws its own random total in [4000, 6200] and splits it evenly
# across the 5 attack categories. ATTACK_TARGET_TOTAL below is kept only
# for reference / for the LEGACY data_loader.py path (see below) and is
# NOT used by the current Taylor-pipeline-based run.
ATTACK_TARGET_TOTAL = 1_200

ATTACK_CATEGORIES = [
    "DDoS-HTTP_Flood",
    "DoS-HTTP_Flood",
    "DNS_Spoofing",
    "XSS",
    "Brute_Force",
]

# ============================================================
# LEGACY SETTINGS (pre-Taylor-integration pipeline)
# ============================================================
# Everything below this point (FILE_MAP, IDENTIFIER_COLS, LABEL_COL,
# PORT_COLS_ARE_FEATURES) belonged to our OWN preprocessing, built
# before we had access to Taylor's shared Phase 1 code. They are kept
# here for reference and for data_loader.py / the old preprocessing.py
# path, but the current pipeline (main.py) now uses taylor_pipeline.py
# instead, which supersedes this logic -- notably, Taylor's version
# ALSO excludes Src/Dst Port and Protocol as identifiers (via keyword
# matching), which our own IDENTIFIER_COLS below did NOT do. This was
# flagged as a leakage-adjacent risk in our own pipeline and is one
# reason the Taylor-ported version is preferred going forward.

# ============================================================
# FILE MAPPING
# ============================================================
# Each category can map to MULTIPLE files, which get POOLED (concatenated)
# together before sampling. This matters for two categories specifically:
#
#   - "Benign" has 4 separate files. We pool all of them into one large
#     benign pool, then sample BENIGN_TARGET rows from the pool -- this
#     gives us access to the full ~44,895 real benign rows rather than
#     being stuck with whichever single file we happened to pick.
#
#   - "DoS-HTTP_Flood" has 2 near-identical files ("flood" and "flood1").
#     Per the professor's Q&A, these represent the SAME attack type --
#     the duplication is just an artifact of how the data was captured,
#     not two different attacks. We pool them for the same reason as
#     Benign: more real rows to sample from.
FILE_MAP = {
    "Benign": [
        "BenignTraffic3_pcap_Flow.csv",
        "BenignTraffic_pcap_Flow_csv_for_claude.csv",
        "BenignTraffic1_pcap_Flow_csv_for_claude.csv",
        "BenignTraffic2_pcap_Flow_csv_for_claude.csv",
    ],
    "DDoS-HTTP_Flood": [
        "DDoS-HTTP_Flood-_pcap_Flow_for_claude.csv",
    ],
    "DoS-HTTP_Flood": [
        "DoS-HTTP_Flood_pcap_Flow_for_claude.csv",
        "DoS-HTTP_Flood1_pcap_Flow_for_claude.csv",
    ],
    "DNS_Spoofing": [
        "DNS_Spoofing_pcap_Flow_for_claude.csv",
    ],
    "XSS": [
        "XSS_pcap_Flow.csv",
    ],
    "Brute_Force": [
        "DictionaryBruteForce_pcap_Flow.csv",
    ],
}

# ============================================================
# COLUMN ROLES
# ============================================================
# Every column in a flow-level record falls into one of these roles.
# Getting this classification right is one of the most important
# modeling decisions in the whole pipeline -- get it wrong and you
# either leak information or throw away useful signal.

# IDENTIFIER columns describe WHO was involved in a connection, not
# WHAT the traffic behaved like. If we fed these into the model as
# numeric features, the model could "cheat" by memorizing specific
# IP addresses seen during training rather than learning generalizable
# behavioral patterns -- and it would fail completely on traffic from
# IPs it has never seen (i.e. it would not generalize to new attacks).
# We exclude these from the feature matrix entirely.
IDENTIFIER_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]

# Src Port / Dst Port are a genuinely gray area, flagged here on purpose:
# port numbers CAN carry real signal (e.g. many attacks target port 80
# or 443), but they can also just reflect quirks of how THIS dataset
# was captured (e.g. all benign traffic happens to use one test server's
# ports) rather than a pattern that generalizes to new deployments.
# We keep them IN the feature set by default (they are numeric columns,
# not excluded below), but flag this explicitly so you can discuss it
# as a limitation in your report, and can move them to IDENTIFIER_COLS
# with one line if you decide to exclude them.
PORT_COLS_ARE_FEATURES = True  # set False and add "Src Port","Dst Port"
                                # to IDENTIFIER_COLS to exclude them

# The Label column is the TARGET, not a feature -- handled separately,
# never passed into the model as an input.
LABEL_COL = "Label"

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
# class_weight="balanced" tells sklearn to automatically weight the
# minority class (attacks, ~3% of rows) more heavily during training.
# Without this, a model can achieve high accuracy by simply predicting
# "benign" for everything and never learning what an attack looks like
# -- this is the single most important setting for imbalanced data.
CLASS_WEIGHT = "balanced"

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,          # gradient-based solvers need enough
                                # iterations to converge on flow data's
                                # wide-ranging feature scales
    "class_weight": CLASS_WEIGHT,
    "random_state": SEED,
}

DECISION_TREE_PARAMS = {
    "max_depth": 8,            # caps tree depth to reduce overfitting;
                                # an unconstrained tree can grow one leaf
                                # per training row and memorize noise
    "min_samples_leaf": 50,    # requires each leaf to represent at least
                                # 50 rows, so splits reflect real patterns
                                # rather than a handful of coincidental rows
    "class_weight": CLASS_WEIGHT,
    "random_state": SEED,
}

# ============================================================
# ENSEMBLE MODELS (AdaBoost, Gradient Boosting)
# ============================================================
# Random Forest and XGBoost are off-limits (taken by teammates), but
# these two are genuinely different ensemble strategies worth comparing
# against a single Decision Tree and Logistic Regression:
#
# AdaBoost ("Adaptive Boosting") trains many very shallow, weak trees
# (here: depth-1 "stumps") in sequence. After each one, it increases the
# weight of the training rows the previous stump got wrong, so the next
# stump focuses harder on the mistakes. The final prediction is a
# weighted vote across all stumps.
#
# Gradient Boosting also trains trees in sequence, but instead of
# reweighting rows, each new tree is trained to predict the RESIDUAL
# ERROR (how wrong the current ensemble is) of the trees before it,
# gradually correcting the ensemble's mistakes. This is the same family
# of algorithm XGBoost belongs to, just without XGBoost's specific
# engineering optimizations (regularization, histogram-based splitting,
# etc.) -- so comparing it to a Decision Tree tests boosting itself,
# and comparing it conceptually to XGBoost's eventual result (via
# Vansh's numbers) shows how much those specific optimizations added.
#
# NEITHER estimator supports class_weight directly in sklearn:
# - AdaBoost: weighting is applied to its BASE estimator (a shallow
#   Decision Tree) instead -- see models.train_adaboost.
# - Gradient Boosting: has no class_weight parameter at all; we
#   approximate the same effect using per-row sample_weight computed
#   from the class distribution -- see models.train_gradient_boosting.
ADABOOST_PARAMS = {
    "n_estimators": 100,     # number of sequential weak stumps to combine
    "random_state": SEED,
}
ADABOOST_BASE_MAX_DEPTH = 1   # depth-1 trees ("stumps") are the classic,
                               # intentionally weak base learner for AdaBoost

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 100,     # number of sequential correction trees
    "max_depth": 3,          # each correction tree is still shallow --
                              # boosting gets its power from many trees
                              # working together, not from any one being deep
    "random_state": SEED,
}

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
TEST_SIZE = 0.2   # 20% of data held out for testing, never seen in training
STRATIFY = True   # preserve the same benign/attack ratio in both train
                   # and test sets -- critical with imbalanced data, since
                   # a random split could otherwise leave the test set
                   # with zero (or very few) attack examples by chance

# ============================================================
# MULTI-SEED ROBUSTNESS CHECK
# ============================================================
# Running the whole pipeline multiple times with different seeds tells
# you whether your result (e.g. "Decision Tree beats Logistic Regression")
# is a stable pattern or just an artifact of one particular random sample.
# This directly satisfies your professor's requirement: "Use a
# configurable seed so every run gives a different composition."
ROBUSTNESS_SEEDS = [42, 7, 123, 2024, 99]

# ============================================================
# TASK MODE (added after comparing notes with teammate's Phase 3 slide)
# ============================================================
# Vansh's results report MACRO-F1, which only makes sense for MULTI-CLASS
# prediction (predicting WHICH attack type, not just Attack-vs-Benign).
# Our original pipeline was BINARY (Attack vs. Benign), which is a
# different task and produces metrics that are not directly comparable
# to his. Rather than picking one, this pipeline now supports BOTH,
# controlled by this single flag -- run it twice (once per mode) so you
# can present binary results (matches the Phase 2->3 cascade design,
# where a single Attack/Benign alert feeds forward) AND multi-class
# results (directly comparable to Vansh's macro-F1 numbers).
#
# "binary"     -> target is 0=Benign, 1=Attack (any type)
# "multiclass" -> target is the actual category name (Benign, XSS, etc.)
TASK_MODE = "multiclass"   # change to "multiclass" and re-run for the other view

# Taylor's pipeline uses lowercase 'label' (not our earlier 'Label'), and
# lowercase 'benign' as the negative-class value (not 'Benign'). The new
# preprocessing adapter (preprocessing.py's Taylor-aware functions) reads
# these from here so the naming convention lives in one place.
TAYLOR_LABEL_COL = "label"
TAYLOR_BENIGN_VALUE = "benign"

# ============================================================
# GROUP-AWARE SPLITTING (fixes a leakage gap vs. teammate's pipeline)
# ============================================================
# Vansh explicitly checked for "group leakage" -- making sure flows from
# the same source IP don't end up split across both train and test. Our
# original pipeline didn't do this: a plain stratified random split can
# place some of an IP's flows in train and others in test, letting a
# model partially learn "this specific host's behavior" rather than a
# generalizable attack pattern. That inflates test metrics in a way that
# would NOT hold up on genuinely new traffic (different hosts).
#
# GROUP_COL defines what a "group" is -- all rows sharing the same value
# in this column are kept together, entirely in train OR entirely in
# test, never split across both.
GROUP_COL = "Src IP"

# ============================================================
# CLASS WEIGHT ABLATION
# ============================================================
# Vansh found "no weighting works better than weighting it" on his data
# -- a bit surprising, and worth testing rather than assuming on ours.
# When True, main.py will run each model BOTH with class_weight="balanced"
# and with class_weight=None, and report both, rather than assuming
# balanced weighting is always the right call.
RUN_CLASS_WEIGHT_ABLATION = True
