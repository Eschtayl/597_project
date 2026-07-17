import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

from config import SAVED_FIGS_DIR, BENIGN_LABEL, RANDOM_SEED, RESULTS_FILE


def _benign_cmap():
    # Skip pale end of Blues so bins stay readable on white background
    colors = plt.cm.Blues(np.linspace(0.55, 1.0, 256))
    return ListedColormap(colors)


def _benign_legend_handle():
    # Proxy artist so hexbin density still appears in the legend
    return Patch(facecolor=plt.cm.Blues(0.75), edgecolor='none', label='benign')


def ensure_figs_dir(path=SAVED_FIGS_DIR):
    os.makedirs(path, exist_ok=True)
    return path


def _add_count_pct_labels(ax, bars, counts, total):
    # Label each bar with count and percentage
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total if total else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{int(count)}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=9
        )


def visualise_class_counts(labels,
                           save_name_with_attack='class_counts_with_attack.png',
                           save_name_without_attack='class_counts_benign_vs_attack.png'):
    # Class counts with attack types, and benign vs attack (attacks grouped)
    ensure_figs_dir()
    counts_with_attack = labels.value_counts()
    binary_labels = labels.where(labels == BENIGN_LABEL, other='attack')
    counts_without_attack = binary_labels.value_counts().reindex(['benign', 'attack'])

    # With attack types
    fig0, ax0 = plt.subplots(figsize=(9, 7))
    bars0 = ax0.bar(counts_with_attack.index.astype(str), counts_with_attack.values)
    _add_count_pct_labels(ax0, bars0, counts_with_attack.values, counts_with_attack.sum())
    ax0.set_title('Class counts (with attack types)')
    ax0.set_xlabel('Label')
    ax0.set_ylabel('Count')
    ax0.tick_params(axis='x', rotation=30)
    ax0.set_ylim(0, counts_with_attack.max() * 1.18)
    fig0.savefig(os.path.join(SAVED_FIGS_DIR, save_name_with_attack), bbox_inches='tight')
    plt.close(fig0)

    # Without attack types (benign vs attack)
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    bars1 = ax1.bar(counts_without_attack.index.astype(str), counts_without_attack.values)
    _add_count_pct_labels(ax1, bars1, counts_without_attack.values, counts_without_attack.sum())
    ax1.set_title('Class counts (benign vs attack)')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    ax1.set_ylim(0, counts_without_attack.max() * 1.18)
    fig1.savefig(os.path.join(SAVED_FIGS_DIR, save_name_without_attack), bbox_inches='tight')
    plt.close(fig1)

    print('--- Class counts (with attack types) ---')
    print(counts_with_attack.to_string())
    print()
    print('--- Class counts (benign vs attack) ---')
    print(counts_without_attack.to_string())
    print()
    return counts_with_attack


def compute_pca_2d(x_features, seed=RANDOM_SEED):
    # Project behaviour features to 2D for separation plots
    pca = PCA(n_components=2, random_state=seed)
    x_pca = pca.fit_transform(x_features)
    explained = pca.explained_variance_ratio_
    print('--- PCA explained variance ---')
    print(f'PC1: {explained[0]:.4f}, PC2: {explained[1]:.4f}, total: {explained.sum():.4f}')
    print()
    return x_pca, pca


def visualise_pca_benign_vs_attack(x_pca, labels, save_name='pca_benign_vs_attack.png'):
    ensure_figs_dir()
    is_attack = labels.to_numpy() != BENIGN_LABEL
    fig, ax = plt.subplots(figsize=(9, 7))
    # Benign density via hexbin
    hb = ax.hexbin(
        x_pca[~is_attack, 0], x_pca[~is_attack, 1],
        gridsize=60, cmap=_benign_cmap(), mincnt=1, linewidths=0.2,
        norm=LogNorm(), alpha=1.0, zorder=1
    )
    fig.colorbar(hb, ax=ax, label='benign count')
    ax.scatter(
        x_pca[is_attack, 0], x_pca[is_attack, 1],
        s=18, alpha=0.85, color='#d62728', edgecolors='k', linewidths=0.2, label='attack', zorder=2
    )
    ax.set_title('PCA: benign density vs attack')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles=[_benign_legend_handle()] + handles, labels=['benign'] + legend_labels, markerscale=2)
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name), bbox_inches='tight')
    plt.close(fig)
    return


def visualise_pca_by_attack_type(x_pca, labels, save_name='pca_by_attack_type.png'):
    ensure_figs_dir()
    fig, ax = plt.subplots(figsize=(10, 7))
    benign_mask = labels.to_numpy() == BENIGN_LABEL
    # Benign density via hexbin
    hb = ax.hexbin(
        x_pca[benign_mask, 0], x_pca[benign_mask, 1],
        gridsize=60, cmap=_benign_cmap(), mincnt=1, linewidths=0.2,
        norm=LogNorm(), alpha=1.0, zorder=1
    )
    fig.colorbar(hb, ax=ax, label='benign count')
    for attack_label in sorted(labels[labels != BENIGN_LABEL].unique()):
        mask = labels.to_numpy() == attack_label
        ax.scatter(
            x_pca[mask, 0], x_pca[mask, 1],
            s=20, alpha=0.9, label=str(attack_label), edgecolors='k', linewidths=0.15, zorder=2
        )
    ax.set_title('PCA: benign density + attack types')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=[_benign_legend_handle()] + handles,
        labels=['benign'] + legend_labels,
        markerscale=2,
        fontsize=8
    )
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name), bbox_inches='tight')
    plt.close(fig)
    return


def compute_benign_centroid_distances(x_features, labels):
    # Euclidean distance of each row to the benign mean (scaled feature space)
    benign_mask = labels == BENIGN_LABEL
    centroid = x_features.loc[benign_mask].mean().values
    distances = np.linalg.norm(x_features.values - centroid, axis=1)
    return distances


def visualise_benign_centroid_distances(distances, labels, save_name='benign_centroid_distance.png'):
    ensure_figs_dir()
    is_attack = labels != BENIGN_LABEL
    fig = plt.figure(figsize=(9, 7))
    plt.hist(distances[~is_attack], bins=80, alpha=0.5, density=True, label='benign')
    plt.hist(distances[is_attack], bins=80, alpha=0.5, density=True, label='attack')
    plt.title('Distance to benign centroid')
    plt.xlabel('Euclidean distance')
    plt.ylabel('Density')
    plt.legend()
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name), bbox_inches='tight')
    plt.close(fig)

    print('--- Distance to benign centroid ---')
    print(f'benign median: {np.median(distances[~is_attack]):.4f}')
    print(f'attack median: {np.median(distances[is_attack]):.4f}')
    print(f'benign 95th pct: {np.percentile(distances[~is_attack], 95):.4f}')
    print(f'attack 5th pct:  {np.percentile(distances[is_attack], 5):.4f}')
    print()
    return


def visualise_correlation_heatmap(x_features, top_n=25, sample_rows=5000, seed=RANDOM_SEED,
                                  save_name='correlation_heatmap.png'):
    # Use highest-variance features so the heatmap stays readable
    ensure_figs_dir()
    variances = x_features.var().sort_values(ascending=False)
    top_cols = variances.head(top_n).index.tolist()
    sample_df = x_features[top_cols].sample(n=min(sample_rows, len(x_features)), random_state=seed)
    corr = sample_df.corr()

    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True)
    plt.title(f'Correlation heatmap (top {top_n} variance features)')
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name), bbox_inches='tight')
    plt.close(fig)

    abs_corr = corr.abs().where(~np.eye(corr.shape[0], dtype=bool))
    max_pair = abs_corr.max().max()
    print('--- Correlation (top variance features) ---')
    print(f'max absolute pairwise correlation: {max_pair:.4f}')
    print(f'feature dim used in models: {x_features.shape[1]}')
    print()
    return corr


def visualise_top_mean_diff_features(x_features, labels, top_n=15, save_name='top_mean_diff_features.png'):
    # Features with largest |mean_attack - mean_benign|
    ensure_figs_dir()
    benign_mask = labels == BENIGN_LABEL
    mean_benign = x_features.loc[benign_mask].mean()
    mean_attack = x_features.loc[~benign_mask].mean()
    mean_diff = (mean_attack - mean_benign).abs().sort_values(ascending=False)
    top = mean_diff.head(top_n)

    fig = plt.figure(figsize=(9, 7))
    plt.barh(top.index.astype(str)[::-1], top.values[::-1])
    plt.title(f'Top {top_n} features by |mean attack - mean benign|')
    plt.xlabel('Absolute mean difference')
    fig.savefig(os.path.join(SAVED_FIGS_DIR, save_name), bbox_inches='tight')
    plt.close(fig)

    print('--- Top mean-difference features ---')
    print(top.to_string())
    print()
    return top


def write_eda_summary(counts, pca, distances, labels, top_features, path=RESULTS_FILE):
    is_attack = labels != BENIGN_LABEL
    explained = pca.explained_variance_ratio_
    with open(path, 'w') as f:
        f.write('### PHASE 2 EDA SUMMARY ###\n\n')
        f.write('Class counts:\n')
        f.write(counts.to_string())
        f.write('\n\n')
        f.write(f'PCA variance PC1={explained[0]:.4f}, PC2={explained[1]:.4f}, total={explained.sum():.4f}\n')
        f.write(f'Benign centroid distance — benign median={np.median(distances[~is_attack]):.4f}, '
                f'attack median={np.median(distances[is_attack]):.4f}\n')
        f.write('\nTop mean-diff features:\n')
        f.write(top_features.to_string())
        f.write('\n')
    print(f'Wrote EDA summary to {path}')
    return


def run_eda(x_features, labels):
    print('Running EDA')
    counts = visualise_class_counts(labels)
    x_pca, pca = compute_pca_2d(x_features)
    visualise_pca_benign_vs_attack(x_pca, labels)
    visualise_pca_by_attack_type(x_pca, labels)
    distances = compute_benign_centroid_distances(x_features, labels)
    visualise_benign_centroid_distances(distances, labels)
    visualise_correlation_heatmap(x_features)
    top_features = visualise_top_mean_diff_features(x_features, labels)
    write_eda_summary(counts, pca, distances, labels, top_features)
    print(f'EDA figures saved under {SAVED_FIGS_DIR}')
    return
