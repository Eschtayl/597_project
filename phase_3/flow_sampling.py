"""Sampling over UNIFIED logical flows.

Sampling happens after flow unification: we build one record per LogicalFlowID,
then draw 200,000 benign flows and 4,000-6,200 attack flows balanced across the
five attack types. Sampling raw segments would over-weight multi-segment flows
and collapse under grouping, so it is done on unified flows only.

If a class has fewer unique flows than requested it is sampled in full (no
silent duplication) and a warning is emitted.
"""
import os

import numpy as np
import pandas as pd

from config import (
    FLOW_DIR, RANDOM_SEED, DATA_DIR, UNIFIED_SAMPLE_PATH,
    LABEL_COL, BENIGN_LABEL, ATTACK_LABELS, N_ATTACK_TYPES,
    BENIGN_SAMPLE_N, ATTACK_SAMPLE_MIN, ATTACK_SAMPLE_MAX,
)
from flow_aggregation import build_unified_flows


def sample_flows(unified, seed=RANDOM_SEED):
    """Draw the Phase 3 sample from a unified-flow frame. Returns the sampled
    subframe (rows are whole logical flows)."""
    rng = np.random.default_rng(seed)

    benign = unified[unified[LABEL_COL] == BENIGN_LABEL]
    if len(benign) < BENIGN_SAMPLE_N:
        print(f'  WARNING: only {len(benign):,} benign flows (< {BENIGN_SAMPLE_N:,}); taking all')
    n_benign = min(BENIGN_SAMPLE_N, len(benign))
    benign_s = benign.sample(n=n_benign, random_state=seed)

    attacks = unified[unified[LABEL_COL] != BENIGN_LABEL]
    present = sorted(attacks[LABEL_COL].unique())
    if len(present) != N_ATTACK_TYPES:
        raise ValueError(f'Expected {N_ATTACK_TYPES} attack types, found {len(present)}: {present}')

    total_attack = int(rng.integers(ATTACK_SAMPLE_MIN, ATTACK_SAMPLE_MAX, endpoint=True))
    per_type = total_attack // N_ATTACK_TYPES

    parts = []
    for lab in present:
        pool = attacks[attacks[LABEL_COL] == lab]
        take = min(per_type, len(pool))
        if take < per_type:
            print(f'  WARNING: attack "{lab}" has only {len(pool):,} flows (< {per_type:,}); taking all')
        parts.append(pool.sample(n=take, random_state=seed))
    attack_s = pd.concat(parts, ignore_index=False)

    sampled = pd.concat([benign_s, attack_s], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    return sampled


def build_sampled_dataset(files=None, path=FLOW_DIR, seed=RANDOM_SEED, cache=True):
    """Full path: build unified flows for every capture, sample, optionally cache."""
    print('Building unified logical flows across all captures...')
    unified = build_unified_flows(path=path, files=files)
    print(f'  unified flows: {len(unified):,}')
    by_class = unified[LABEL_COL].value_counts()
    for lab, c in by_class.items():
        print(f'    {lab:<18} {c:>9,}')

    print('Sampling 200k benign + balanced attacks over unified flows...')
    sampled = sample_flows(unified, seed=seed)
    print(f'  sampled flows: {len(sampled):,}')

    if cache:
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            sampled.to_parquet(UNIFIED_SAMPLE_PATH)
            print(f'  cached -> {UNIFIED_SAMPLE_PATH}')
        except Exception as e:   # pyarrow/fastparquet not available -> pickle fallback
            alt = UNIFIED_SAMPLE_PATH.replace('.parquet', '.pkl')
            sampled.to_pickle(alt)
            print(f'  parquet unavailable ({e.__class__.__name__}); cached -> {alt}')
    return sampled


def load_sampled_dataset(path=UNIFIED_SAMPLE_PATH):
    if os.path.exists(path):
        return pd.read_parquet(path)
    alt = path.replace('.parquet', '.pkl')
    if os.path.exists(alt):
        return pd.read_pickle(alt)
    raise FileNotFoundError('No cached sample; run build_sampled_dataset() first.')


if __name__ == '__main__':
    build_sampled_dataset()
