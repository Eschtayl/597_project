import time
from sklearn.ensemble import HistGradientBoostingClassifier

from config import RANDOM_SEED

# Histogram-binned boosting; scales to this row count far better than GradientBoostingClassifier.
GB_MAX_ITER = 300
GB_MAX_DEPTH = 8
GB_LEARNING_RATE = 0.1
GB_L2_REGULARIZATION = 0.0


def train_gradient_boosting(x_train, y_train, x_val=None, y_val=None):
    model = HistGradientBoostingClassifier(
        max_iter=GB_MAX_ITER,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        l2_regularization=GB_L2_REGULARIZATION,
        class_weight='balanced',
        random_state=RANDOM_SEED,
    )
    print(f'Fitting HistGradientBoosting on {len(x_train):,} rows, {x_train.shape[1]} features')
    t0 = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - t0
    print(f'Gradient Boosting fit in {train_time:.2f}s')
    if x_val is not None and y_val is not None:
        val_acc = model.score(x_val, y_val)
        print(f'Gradient Boosting val accuracy: {val_acc:.4f}')
    return model, train_time


def predict_gradient_boosting(model, x):
    t0 = time.time()
    y_pred = model.predict(x)
    scores = model.predict_proba(x)[:, 1]
    infer_time = time.time() - t0
    return y_pred, scores, infer_time
