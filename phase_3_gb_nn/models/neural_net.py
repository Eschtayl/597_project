import time
import numpy as np
from sklearn.neural_network import MLPClassifier

from config import RANDOM_SEED

# Kept shallow: ~76 features and a small attack class, so a deeper net overfits the majority.
NN_HIDDEN_LAYERS = (64, 32)
NN_ALPHA = 1e-3
NN_MAX_ITER = 200
NN_EARLY_STOPPING = True
NN_N_ITER_NO_CHANGE = 10


def train_neural_net(x_train, y_train, x_val=None, y_val=None):
    model = MLPClassifier(
        hidden_layer_sizes=NN_HIDDEN_LAYERS,
        alpha=NN_ALPHA,
        max_iter=NN_MAX_ITER,
        early_stopping=NN_EARLY_STOPPING,
        n_iter_no_change=NN_N_ITER_NO_CHANGE,
        random_state=RANDOM_SEED,
    )
    print(f'Fitting small MLP {NN_HIDDEN_LAYERS} on {len(x_train):,} rows, '
          f'{x_train.shape[1]} features')
    t0 = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - t0
    print(f'Neural net fit in {train_time:.2f}s ({model.n_iter_} iterations)')
    if x_val is not None and y_val is not None:
        val_acc = model.score(x_val, y_val)
        print(f'Neural net val accuracy: {val_acc:.4f}')
    return model, train_time


def predict_neural_net(model, x):
    t0 = time.time()
    y_pred = model.predict(x)
    scores = model.predict_proba(x)[:, 1]
    infer_time = time.time() - t0
    return y_pred, scores, infer_time
