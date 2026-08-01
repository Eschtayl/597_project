"""
models.py
=========
Training logic for the two models being compared: Logistic Regression
(linear) and Decision Tree (non-linear).

WHY THESE TWO MODELS, SPECIFICALLY?
This pair was chosen to directly test one question: does separating
attack traffic from benign traffic require a non-linear decision
boundary, or is a simple linear one enough?

- Logistic Regression computes a weighted sum of all input features,
  passes it through a sigmoid function, and thresholds the result. Its
  decision boundary -- the surface separating "predict benign" from
  "predict attack" in feature space -- is a straight hyperplane. It can
  only combine features additively; it cannot natively express "attack
  if (A is high AND B is low) OR (C is very high)" -- interactions like
  that require either non-linear features or a non-linear model.

- A Decision Tree asks a sequence of yes/no questions ("is Flow Duration
  > 500?", then "is Packet Length Mean > 200?", etc.), producing a
  decision boundary made of axis-aligned rectangular regions. This lets
  it naturally express feature interactions and non-monotonic patterns
  that Logistic Regression cannot.

Both are also relatively INTERPRETABLE compared to, say, a neural
network or a large ensemble -- Logistic Regression via its coefficients
(each one tells you the direction and strength of a feature's effect),
and a Decision Tree via its explicit split rules (you can trace exactly
why any single prediction was made by following its path through the
tree). This interpretability is itself part of what we're comparing.
"""

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

import config

# Sentinel distinguishing "no override given, use config default" from
# "override explicitly set to None" (which means "no class weighting" --
# a real, meaningful setting used by the class-weight ablation, and thus
# not something we can safely conflate with "not provided").
_UNSET = object()


def train_logistic_regression(X_train, y_train, class_weight=_UNSET, already_scaled=False):
    """
    Fit a StandardScaler on the TRAINING data only, transform X_train,
    then fit a Logistic Regression model.

    Parameters
    ----------
    class_weight : str, dict, or _UNSET
        Overrides config.CLASS_WEIGHT for this call. Used by the
        class-weight ablation (see evaluate.py) to test both the
        weighted and unweighted case, rather than assuming balanced
        weighting is always the right call -- teammate Vansh found
        the OPPOSITE on his data ("no weighting works better"), which
        is a good reason to actually test this rather than assume it.
    already_scaled : bool
        Set True when X_train came from taylor_pipeline.py, which
        already applies log1p + RobustScaler to every numeric feature
        (see taylor_pipeline.log_and_scale). Re-scaling already-scaled
        data isn't wrong exactly, but it's redundant and makes the
        model's coefficients harder to interpret against Taylor's
        actual preprocessing -- so we skip our own scaler in that case
        and train directly on his output.

    WHY FIT THE SCALER ON TRAINING DATA ONLY (when NOT already_scaled)?
    This is the single most important anti-leakage rule in this whole
    pipeline. The scaler's parameters (mean and standard deviation per
    feature) are themselves a summary of the data they're fit on. If we
    fit the scaler on the FULL dataset (train + test combined), the
    scaler's parameters would already "know" something about the test
    set's distribution before the model ever sees it -- a subtle form
    of information leakage from test to train. The correct procedure:
        1. Fit the scaler using ONLY X_train  -> scaler.fit(X_train)
        2. Transform X_train with those params -> scaler.transform(X_train)
        3. Transform X_test with the SAME params (fit again) at
           evaluation time -- see evaluate.py.
    (Taylor's RobustScaler is fit on the full sampled dataset BEFORE the
    train/test split, in his current code -- see the README's note on
    this as a known discussion point for your report, since it's a
    minor leakage risk worth flagging even though it isn't ours to fix.)

    Returns
    -------
    model : fitted LogisticRegression
    scaler : fitted StandardScaler, or None if already_scaled=True
    train_time_sec : float
    """
    if already_scaled:
        X_train_scaled = X_train
        scaler = None
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

    params = dict(config.LOGISTIC_REGRESSION_PARAMS)
    if class_weight is not _UNSET:
        params["class_weight"] = class_weight

    t0 = time.time()
    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)
    train_time_sec = time.time() - t0

    return model, scaler, train_time_sec


def train_decision_tree(X_train, y_train, class_weight=_UNSET):
    """
    Fit a Decision Tree. No scaling needed -- a tree's splits are just
    threshold comparisons on raw feature values ("is column X > 12.5?"),
    which are invariant to the units or scale of the feature.

    Parameters
    ----------
    class_weight : str, dict, or None
        Overrides config.CLASS_WEIGHT for this call (see the ablation
        note in train_logistic_regression above).

    WHY max_depth AND min_samples_leaf?
    An unconstrained Decision Tree will keep splitting until every leaf
    is perfectly pure (often one training row per leaf), which means it
    has effectively memorized the training set, including its noise --
    this is a textbook case of OVERFITTING: excellent training accuracy,
    poor performance on new data. Capping max_depth limits how many
    sequential questions the tree can ask, and min_samples_leaf requires
    each final leaf to represent a meaningful number of rows rather than
    a handful of coincidental ones. Both act as regularization.

    Returns
    -------
    model : fitted DecisionTreeClassifier
    train_time_sec : float
    """
    params = dict(config.DECISION_TREE_PARAMS)
    if class_weight is not _UNSET:
        params["class_weight"] = class_weight

    t0 = time.time()
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    train_time_sec = time.time() - t0

    return model, train_time_sec


def train_dummy_classifier(X_train, y_train, strategy="most_frequent"):
    """
    Fit a DummyClassifier -- a trivial baseline that ignores the input
    features entirely and predicts based only on the label distribution
    seen during training (e.g. "most_frequent" always predicts whatever
    class appeared most often in training, which for us is Benign).

    WHY THIS MATTERS
    Teammate Vansh explicitly included a Dummy baseline in his Phase 3
    comparison. Without one, "our model got 88% accuracy" sounds
    impressive in isolation -- but on a ~97% benign dataset, a Dummy
    classifier that never even looks at the data would ALSO score ~97%
    accuracy. Including this baseline makes concrete exactly how much
    (or how little) real signal Logistic Regression and the Decision
    Tree are actually extracting, rather than leaving it implied.

    Returns
    -------
    model : fitted DummyClassifier
    train_time_sec : float
    """
    t0 = time.time()
    model = DummyClassifier(strategy=strategy, random_state=config.SEED)
    model.fit(X_train, y_train)
    train_time_sec = time.time() - t0

    return model, train_time_sec


def train_adaboost(X_train, y_train, class_weight=_UNSET):
    """
    Fit an AdaBoost ensemble of shallow decision-tree "stumps."

    HOW ADABOOST WORKS
    Trains many very weak learners (here: depth-1 trees, each barely
    better than a coin flip) one after another. After each stump, every
    training row that was misclassified gets a HIGHER weight for the
    next stump -- so the ensemble progressively focuses on whatever it's
    still getting wrong. The final prediction combines every stump's
    vote, weighted by how accurate that stump was.

    WHY class_weight IS NOT APPLIED HERE (a documented limitation, not
    a bug)
    In principle, class weighting could be passed to AdaBoost's base
    estimator (the depth-1 Decision Tree), the same way it's used in
    train_decision_tree. In practice, this combination is fragile:
    AdaBoost ALREADY reweights misclassified rows every round as its
    core mechanism. Stacking config.CLASS_WEIGHT="balanced" on TOP of
    that can push a stump's weighted training error above 50% at some
    round, at which point sklearn's SAMME algorithm cannot mathematically
    continue (a stump doing worse than a coin flip cannot be assigned a
    meaningful positive vote weight) and raises a hard error. This was
    confirmed empirically while building this pipeline -- the error
    reads: "BaseClassifier in AdaBoostClassifier ensemble is worse than
    random, ensemble can not be fit."

    Rather than silently retrying or masking this, AdaBoost is run
    WITHOUT class weighting here, relying entirely on its own built-in
    reweighting mechanism to attend to the rare (attack) class -- and
    this is stated explicitly rather than left implicit. State this
    exactly this way if asked: "AdaBoost's own boosting mechanism
    already reweights misclassified rows each round; combining this
    with explicit class_weight caused numerical failures, so AdaBoost
    is evaluated unweighted, and this is a known interaction rather
    than an oversight." The class_weight parameter is still accepted
    here (for a consistent function signature across all four models)
    but has no effect -- see the printed note below if a non-default
    value is passed.

    Returns
    -------
    model : fitted AdaBoostClassifier
    train_time_sec : float
    """
    if class_weight not in (_UNSET, None):
        print(f"  [NOTE] AdaBoost does not support class_weight={class_weight!r} "
              f"reliably (SAMME incompatibility -- see train_adaboost docstring). "
              f"Training unweighted instead.")

    base_estimator = DecisionTreeClassifier(
        max_depth=config.ADABOOST_BASE_MAX_DEPTH,
        random_state=config.SEED,
    )

    params = dict(config.ADABOOST_PARAMS)

    t0 = time.time()
    model = AdaBoostClassifier(estimator=base_estimator, **params)
    model.fit(X_train, y_train)
    train_time_sec = time.time() - t0

    return model, train_time_sec


def train_gradient_boosting(X_train, y_train, class_weight=_UNSET):
    """
    Fit a Gradient Boosting ensemble of shallow correction trees.

    HOW GRADIENT BOOSTING WORKS
    Also trains trees sequentially, but rather than reweighting
    misclassified ROWS like AdaBoost, each new tree is trained to
    predict the RESIDUAL ERROR of the ensemble built so far -- roughly,
    "here's how wrong the current combined prediction still is; learn
    to correct that." This is the same general family of algorithm
    XGBoost belongs to (XGBoost adds specific engineering optimizations
    on top of this same core idea).

    WHY CLASS WEIGHTING IS HANDLED DIFFERENTLY HERE
    sklearn's GradientBoostingClassifier has NO class_weight parameter
    at all (unlike Logistic Regression, Decision Tree, or even
    AdaBoost's base estimator). We approximate the same intent using
    per-row `sample_weight` at fit time: rows from the rare class get a
    proportionally higher weight, computed the same way sklearn's own
    "balanced" class_weight would compute it internally.

    Returns
    -------
    model : fitted GradientBoostingClassifier
    train_time_sec : float
    """
    if class_weight is _UNSET or class_weight == "balanced":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    else:
        sample_weight = None  # class_weight explicitly None -> no weighting

    params = dict(config.GRADIENT_BOOSTING_PARAMS)

    t0 = time.time()
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    train_time_sec = time.time() - t0

    return model, train_time_sec


def get_logistic_regression_coefficients(model, feature_names, top_n=15):
    """
    Extract and rank Logistic Regression's learned coefficients.

    HOW TO READ THESE (BINARY MODE)
    Each coefficient corresponds to one input feature (after scaling).
    - Sign: positive means higher values of this feature push the
      prediction TOWARD "attack"; negative pushes toward "benign".
    - Magnitude: larger absolute value means a stronger effect on the
      prediction, all else held equal.
    Because features were standardized before fitting, coefficients are
    directly comparable to each other in magnitude -- this comparability
    is exactly why we scaled in the first place.

    HOW TO READ THESE (MULTICLASS MODE)
    sklearn fits one set of coefficients PER CLASS (one-vs-rest style).
    `model.coef_` has shape (n_classes, n_features) instead of
    (1, n_features). There's no single "the" coefficient per feature
    anymore -- a feature can push strongly toward one attack type while
    being irrelevant to another. We summarize this by ranking features
    on their MEAN ABSOLUTE coefficient across all classes, which
    answers "which features matter most somewhere in the model overall,"
    while the full per-class matrix is still available on `model.coef_`
    for deeper inspection if you need to explain one specific class.
    """
    if model.coef_.shape[0] == 1:
        # Binary case: coef_ has shape (1, n_features)
        coefs = pd.Series(model.coef_[0], index=feature_names)
        return coefs.sort_values(key=abs, ascending=False).head(top_n)
    else:
        # Multiclass case: coef_ has shape (n_classes, n_features)
        mean_abs = pd.Series(
            np.abs(model.coef_).mean(axis=0), index=feature_names
        )
        return mean_abs.sort_values(ascending=False).head(top_n)


def get_feature_importances(model, feature_names, top_n=15):
    """
    Extract and rank feature_importances_ from ANY model that exposes
    them -- Decision Tree, AdaBoost, and Gradient Boosting all do.

    HOW TO READ THESE
    Each value represents how much that feature contributed to reducing
    "impurity" (roughly: class mixing) across all splits in the model,
    weighted by how many training rows passed through each split. For
    ensembles (AdaBoost, Gradient Boosting), this is further weighted by
    each individual tree's contribution to the final ensemble. Like a
    single Decision Tree's importances, these are always non-negative --
    there's no "direction" here, just "how useful was this feature
    somewhere in the model."
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(top_n)


def get_decision_tree_importances(model, feature_names, top_n=15):
    """Kept as a named alias for backward compatibility -- see
    get_feature_importances, which this now delegates to and which also
    works for AdaBoost and Gradient Boosting."""
    return get_feature_importances(model, feature_names, top_n=top_n)
