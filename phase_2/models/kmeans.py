import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA

from config import (
    RANDOM_SEED,
    KM_N_CLUSTERS,
    KM_N_INIT,
    KM_PCA_COMPONENTS,
    KM_MINIBATCH,
)


# I kept the same interface as models/isolation_forest.py on purpose, so main.py
# does not need to change how it calls the model. Each model gives:
#   train_*(x_train, y_train)  -> a fitted model
#   get_anomaly_scores(model, x) -> a score where a higher value means more
#                                   anomalous.
# Once the score follows this "higher = more anomalous" rule, the existing
# choose_threshold / generate_alerts / evaluation code works for K-Means too.


class KMeansAnomalyModel:
    # I bundle three things together as one "model" object: the PCA projector (if
    # I use PCA), the fitted K-Means, and the attack purity of every cluster that
    # I measured on the training data. get_anomaly_scores() needs all three later.
    def __init__(self, pca, kmeans, cluster_purity):
        self.pca = pca
        self.kmeans = kmeans
        self.cluster_purity = cluster_purity  # purity[c] = share of attacks in cluster c (train)


def _to_array(x):
    # data_prep sometimes passes a DataFrame and sometimes a numpy array, so I
    # just normalise it to an array here to avoid surprises.
    return x.to_numpy() if hasattr(x, 'to_numpy') else np.asarray(x)


def train_kmeans(x_train, y_train, seed=RANDOM_SEED, n_clusters=KM_N_CLUSTERS,
                 n_init=KM_N_INIT, pca_components=KM_PCA_COMPONENTS, minibatch=KM_MINIBATCH):
    # The training itself is unsupervised: K-Means only sees x_train and never the
    # labels. I only bring y_train in AFTER clustering, to measure how many attacks
    # fell into each cluster. This is the same idea the IF and AE models use, where
    # the labels are for interpreting the result, not for fitting it.
    x_arr = _to_array(x_train)
    y = np.asarray(y_train)

    # Reason for PCA first: after preprocessing we have around 100 features, and in
    # that many dimensions the Euclidean distances all look similar (the usual
    # curse of dimensionality), so K-Means struggles to form clean groups. Reducing
    # to a few components first makes the clusters much sharper. Set the config
    # value to 0 if you want to switch PCA off and compare.
    pca = None
    if pca_components and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=seed).fit(x_arr)
        x_arr = pca.transform(x_arr)

    # Why I use a large k: attacks are only about 2% of the data, so if I ask for
    # k=2 the algorithm just cuts the big benign cloud into two halves and finds
    # nothing useful. With many small clusters the rare attacks can gather into
    # their own almost-pure clusters instead. MiniBatch is here because full
    # K-Means gets slow once k is in the hundreds.
    if minibatch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init,
                                 random_state=seed, batch_size=4096)
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    clusters = kmeans.fit_predict(x_arr)

    # For each cluster I compute its "attack purity": among the training points that
    # landed in this cluster, what fraction were attacks. A cluster full of attacks
    # gets a purity near 1, a benign cluster gets near 0. Later I reuse this number
    # straight away as the anomaly score.
    purity = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = clusters == c
        purity[c] = y[mask].mean() if mask.any() else 0.0

    n_attack_clusters = int((purity >= 0.5).sum())
    print(f'K-Means fitted: k={n_clusters}, pca={pca_components}, '
          f'{n_attack_clusters} clusters are majority-attack')
    return KMeansAnomalyModel(pca, kmeans, purity)


def get_anomaly_scores(model, x_data):
    # For a new point I find which cluster it belongs to, then return that cluster's
    # attack purity as the score. So a point that lands in a mostly-attack cluster
    # gets a high score. One thing to keep in mind: because a point can only take
    # the purity of its own cluster, the score has at most k different values, which
    # makes the ROC curve look step-like. That is just how a hard clustering behaves,
    # it is not a bug in the code.
    x_arr = _to_array(x_data)
    if model.pca is not None:
        x_arr = model.pca.transform(x_arr)
    clusters = model.kmeans.predict(x_arr)
    return model.cluster_purity[clusters]
