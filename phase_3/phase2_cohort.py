"""Phase 2 packet cohort for the cascade.

Produces the existing-style Phase 2 packet predictions with a train/val/test split
and the raw 5-tuple identifiers needed to map alerts to flows. ŷ₂ is packet-level.

Phase 2 model = Isolation Forest with phase_2's configuration (fast, deterministic;
the autoencoder is a later robustness swap). Packets are drawn with a memory-bounded
min-key sample across the full packet captures so we never load ~2.9 GB at once.

Outputs `phase_3/data/phase2_cohort.parquet`: one row per val/test packet with
identifiers, label, split, anomaly score, and ŷ₂ alert flag.

Usage: python phase_3/phase2_cohort.py
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from config import PACKET_DIR, DATA_DIR, RANDOM_SEED, BENIGN_LABEL
from flow_aggregation import label_from_filename
from helpers import feature_cleaner, log_and_scale

SEED = RANDOM_SEED
ID_COLS = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'l4_tcp', 'l4_udp']
IDENTIFIER_KEYWORDS = ['ip', 'port', 'mac', 'timestamp', 'flow_id', 'protocol',
                       'server', 'host', 'user_agent', 'oui', 'uri', 'content_type']
BENIGN_TARGET = 200_000
ATTACK_TARGET_PER_TYPE = 1_200
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def min_key_sample(files, n_target, seed, chunksize=100_000):
    """Uniform without-replacement sample of n_target rows streamed across files
    (keep the n_target rows with the smallest random keys). Memory-bounded."""
    rng = np.random.default_rng(seed)
    kept = None
    for f in files:
        for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):
            chunk = chunk.copy()
            chunk['__k'] = rng.random(len(chunk))
            chunk['__src'] = os.path.basename(f)
            kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
            if len(kept) > n_target:
                kept = kept.nsmallest(n_target, '__k')
    return kept


def get_identifier_columns(df):
    cols = []
    for c in df.columns:
        words = c.lower().replace(' ', '_').replace('-', '_').split('_')
        if any(k in words for k in IDENTIFIER_KEYWORDS):
            cols.append(c)
    return cols


def load_packet_cohort(seed=SEED):
    all_csvs = sorted(glob.glob(os.path.join(PACKET_DIR, '*.csv')))
    benign_files = [f for f in all_csvs if os.path.basename(f).startswith('Benign')]
    attack_files = [f for f in all_csvs if not os.path.basename(f).startswith('Benign')]
    print(f'Packet captures: {len(benign_files)} benign, {len(attack_files)} attack')

    print('Sampling benign packets (memory-bounded)...')
    benign = min_key_sample(benign_files, BENIGN_TARGET, seed)
    benign['label'] = BENIGN_LABEL

    print('Sampling attack packets per type...')
    by_label = {}
    for f in attack_files:
        by_label.setdefault(label_from_filename(f), []).append(f)
    attack_parts = []
    for lab, files in by_label.items():
        s = min_key_sample(files, ATTACK_TARGET_PER_TYPE, seed + hash(lab) % 1000)
        s['label'] = lab
        attack_parts.append(s)
    attacks = pd.concat(attack_parts, ignore_index=True)

    df = pd.concat([benign, attacks], ignore_index=True).drop(columns=['__k', '__src'], errors='ignore')
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f'  packet cohort: {len(df):,} ({(df.label==BENIGN_LABEL).sum():,} benign, '
          f'{(df.label!=BENIGN_LABEL).sum():,} attack)')
    return df


def main():
    df = load_packet_cohort()
    identity = df[ID_COLS].reset_index(drop=True)
    labels = df['label'].reset_index(drop=True)

    print('Preprocessing packet features (Phase 1 pipeline)...')
    df_numeric, df_identifiers = feature_cleaner(df.drop(columns=['label']))
    df_final, _ = log_and_scale(df_numeric, df_identifiers, labels)

    id_cols = get_identifier_columns(df_final)
    x = df_final.drop(columns=id_cols + ['label'], errors='ignore').select_dtypes(include=[np.number])
    y_true = (labels != BENIGN_LABEL).astype(int).to_numpy()
    print(f'  feature matrix: {x.shape}, identifiers held out: {len(id_cols)}')

    # train / val / test split (stratified), identity kept aligned
    idx = np.arange(len(x))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, stratify=y_true)
    val_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=SEED, stratify=y_true[tmp_idx])
    split = np.array(['train'] * len(x), dtype=object)
    split[val_idx] = 'val'; split[te_idx] = 'test'

    # Phase 2 = Isolation Forest on benign train (phase_2 config)
    print('Training Phase 2 Isolation Forest on benign train packets...')
    benign_train = tr_idx[y_true[tr_idx] == 0]
    iso = IsolationForest(n_estimators=200, max_samples=256, contamination='auto',
                          random_state=SEED, n_jobs=-1)
    iso.fit(x.iloc[benign_train])
    scores = -iso.decision_function(x)     # higher = more anomalous

    # threshold by max-F1 on train (phase_2 convention)
    st = scores[tr_idx]; yt = y_true[tr_idx]
    lo, hi = np.percentile(st, 1), np.percentile(st, 99)
    cands = np.linspace(lo, hi, 200)
    best_t, best_f1 = cands[0], -1
    for t in cands:
        f = f1_score(yt, (st >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    alert = (scores >= best_t).astype(int)
    print(f'  Phase 2 threshold={best_t:.6f} (train F1={best_f1:.4f})')

    cohort = identity.copy()
    cohort['label'] = labels.values
    cohort['y_true'] = y_true
    cohort['phase2_split'] = split
    cohort['anomaly_score'] = scores
    cohort['y2_alert'] = alert

    # report Phase 2 quality on val/test
    print('\nPhase 2 packet performance (alerts):')
    for part in ['val', 'test']:
        m = cohort['phase2_split'] == part
        sub = cohort[m]
        tp = int(((sub.y2_alert == 1) & (sub.y_true == 1)).sum())
        fp = int(((sub.y2_alert == 1) & (sub.y_true == 0)).sum())
        fn = int(((sub.y2_alert == 0) & (sub.y_true == 1)).sum())
        n_alert = int((sub.y2_alert == 1).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        print(f'  {part}: packets={len(sub):,} alerts={n_alert:,} TP={tp:,} FP={fp:,} '
              f'recall={rec:.3f} precision={prec:.3f}')

    os.makedirs(DATA_DIR, exist_ok=True)
    cohort.to_parquet(os.path.join(DATA_DIR, 'phase2_cohort.parquet'))
    print(f'\nSaved Phase 2 cohort -> {os.path.join(DATA_DIR, "phase2_cohort.parquet")}')
    print('Next: cascade_alignment.py (maps val/test alerts to the unified-flow index '
          'and runs the leakage guard).')


if __name__ == '__main__':
    main()
