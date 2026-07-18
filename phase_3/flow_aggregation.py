"""LogicalFlowID construction and mathematically-defensible segment aggregation.

CICFlowMeter splits flows longer than 2 minutes into successive segments that
share a directional 5-field Flow ID. ~45% of rows belong to such multi-segment
flows, and the same Flow ID is also *reused* by unrelated conversations hours
apart. This module:

  1. groups segments into logical flows (Flow ID + temporal cluster), and
  2. aggregates each logical flow into one record using per-feature rules that
     follow the meaning of each column (sum / min / max / weighted-mean /
     pooled-variance / recomputed-rate), rather than a blind mean/sum/max.

The aggregation rule for every feature is declared in AGG_RULES so the
"feature -> rule -> justification" dictionary is report material (see
feature_dictionary()).
"""
import os
import glob

import numpy as np
import pandas as pd

from config import (
    FLOW_DIR,
    FLOW_ID_COL,
    TIMESTAMP_COL,
    TIMESTAMP_FORMAT,
    GAP_CUTOFF_SECONDS,
    DURATION_COL,
    MICROSECONDS_PER_SECOND,
    W_FWD,
    W_BWD,
    BENIGN_LABEL,
    ATTACK_SUBSTRING_MAP,
    LABEL_COL,
    NATIVE_LABEL_COL,
)

# ---------------------------------------------------------------------------
# Per-feature aggregation rules.
#   sum     : additive counts / totals / active durations
#   max/min : extrema of extrema
#   first   : connection-setup properties (init window) — earliest segment
#   wmean:W : packet-count-weighted mean (weight column W); exact for 1 segment
#   pooled:(mean_col, n_col) : pooled standard deviation across segments
#   rate:N  : recomputed rate = sum(N) / sum(duration_us) * 1e6
#   drop    : Tier-3 std whose pooled sufficient statistics are unavailable
# Anything not listed is treated as metadata and carried via `first`.
# ---------------------------------------------------------------------------
SUM_COLS = [
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd IAT Total', 'Bwd IAT Total',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
    'Subflow Bwd Bytes', 'Fwd Act Data Pkts',
]
MAX_COLS = [
    'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Flow IAT Max',
    'Fwd IAT Max', 'Bwd IAT Max', 'Packet Length Max', 'Active Max', 'Idle Max',
]
MIN_COLS = [
    'Fwd Packet Length Min', 'Bwd Packet Length Min', 'Flow IAT Min',
    'Fwd IAT Min', 'Bwd IAT Min', 'Packet Length Min', 'Fwd Seg Size Min',
    'Active Min', 'Idle Min',
]
FIRST_COLS = ['FWD Init Win Bytes', 'Bwd Init Win Bytes']

# weighted-mean column -> weight column
WMEAN_COLS = {
    'Fwd Packet Length Mean': W_FWD,
    'Bwd Packet Length Mean': W_BWD,
    'Fwd Segment Size Avg': W_FWD,
    'Bwd Segment Size Avg': W_BWD,
    'Fwd IAT Mean': W_FWD,
    'Bwd IAT Mean': W_BWD,
    'Packet Length Mean': '__total_pkts',
    'Average Packet Size': '__total_pkts',
    'Flow IAT Mean': '__total_pkts',
    'Down/Up Ratio': '__total_pkts',
    'Active Mean': '__total_pkts',
    'Idle Mean': '__total_pkts',
    'Fwd Bytes/Bulk Avg': '__total_pkts',
    'Fwd Packet/Bulk Avg': '__total_pkts',
    'Fwd Bulk Rate Avg': '__total_pkts',
    'Bwd Bytes/Bulk Avg': '__total_pkts',
    'Bwd Packet/Bulk Avg': '__total_pkts',
    'Bwd Bulk Rate Avg': '__total_pkts',
}

# pooled-std column -> (segment mean column, segment count column)
POOLED_STD_COLS = {
    'Fwd Packet Length Std': ('Fwd Packet Length Mean', W_FWD),
    'Bwd Packet Length Std': ('Bwd Packet Length Mean', W_BWD),
    'Packet Length Std': ('Packet Length Mean', '__total_pkts'),
}
# variance columns computed as (pooled std)**2 -> source std column
POOLED_VAR_COLS = {'Packet Length Variance': 'Packet Length Std'}

# rate column -> numerator (summed) column
RATE_COLS = {
    'Flow Bytes/s': '__total_bytes',
    'Flow Packets/s': '__total_pkts',
    'Fwd Packets/s': 'Total Fwd Packet',
    'Bwd Packets/s': 'Total Bwd packets',
}
# Tier-3: pooled sufficient statistics unavailable (no inter-segment packet
# gaps / active-period counts). Dropped now; drop-vs-approx revisited at modelling.
DROP_COLS = ['Flow IAT Std', 'Fwd IAT Std', 'Bwd IAT Std', 'Active Std', 'Idle Std']

ENGINEERED_COLS = ['SegmentCount', 'ts_first', 'ts_last', 'LogicalFlowID', 'source_file', LABEL_COL]


# ---------------------------------------------------------------------------
# Loading + labelling
# ---------------------------------------------------------------------------
def label_from_filename(filename):
    base = os.path.basename(filename)
    if base.startswith('Benign'):
        return BENIGN_LABEL
    for substring, label in ATTACK_SUBSTRING_MAP:
        if substring in base:
            return label
    return 'Unknown Attack'


def load_flow_file(path):
    """Load one flow CSV, attach filename-derived label and source_file."""
    df = pd.read_csv(path, low_memory=False)
    df[LABEL_COL] = label_from_filename(path)
    df['source_file'] = os.path.basename(path)
    return df


# ---------------------------------------------------------------------------
# LogicalFlowID
# ---------------------------------------------------------------------------
def assign_logical_flow_id(df, cutoff=GAP_CUTOFF_SECONDS):
    """Add a LogicalFlowID column: same (source_file, Flow ID) segments are one
    logical flow while consecutive timestamp gaps stay <= cutoff seconds; a
    larger gap starts a new cluster (ID reuse). Rows are returned sorted by
    (source_file, Flow ID, timestamp)."""
    df = df.copy()
    df['ts'] = pd.to_datetime(df[TIMESTAMP_COL], format=TIMESTAMP_FORMAT, errors='coerce')
    key = ['source_file', FLOW_ID_COL]
    df = df.sort_values(key + ['ts']).reset_index(drop=True)

    gap = df.groupby(key, sort=False)['ts'].diff().dt.total_seconds()
    # A new cluster starts on the first segment of a key (gap is NaN) or when the
    # gap exceeds the cutoff.
    new_cluster = gap.isna() | (gap > cutoff)
    df['__cluster'] = new_cluster.astype(int).groupby([df['source_file'], df[FLOW_ID_COL]]).cumsum()

    df['LogicalFlowID'] = (
        df['source_file'].astype(str) + '|' + df[FLOW_ID_COL].astype(str)
        + '#' + df['__cluster'].astype(str)
    )
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _weighted_mean(df, g, col, weight_col):
    w = df[weight_col].where(df[col].notna(), 0.0)
    num = (df[col].fillna(0.0) * w).groupby(df['LogicalFlowID'], sort=False).sum()
    den = w.groupby(df['LogicalFlowID'], sort=False).sum()
    out = num / den.replace(0.0, np.nan)
    return out.reindex(g).fillna(0.0)


def _pooled_std(df, g, std_col, mean_col, n_col):
    n = df[n_col].fillna(0.0).clip(lower=0.0)
    s2 = df[std_col].fillna(0.0) ** 2
    m = df[mean_col].fillna(0.0)
    parts = pd.DataFrame({
        'A': (n - 1).clip(lower=0.0) * s2,   # within-segment SS
        'B': n * m * m,                       # for between-segment SS
        'C': n * m,
        'N': n,
    }).groupby(df['LogicalFlowID'], sort=False).sum()
    N = parts['N']
    between = parts['B'] - (parts['C'] ** 2) / N.replace(0.0, np.nan)
    var = (parts['A'] + between) / (N - 1).where(N > 1, np.nan)
    var = var.clip(lower=0.0).fillna(0.0)   # single-packet / empty groups -> 0
    return np.sqrt(var).reindex(g).fillna(0.0)


def aggregate_logical_flows(df):
    """Collapse segment rows (with LogicalFlowID assigned) into one row per
    logical flow using the per-feature AGG_RULES."""
    df = df.copy()
    df['__total_pkts'] = df[W_FWD].fillna(0) + df[W_BWD].fillna(0)
    df['__total_bytes'] = (df['Total Length of Fwd Packet'].fillna(0)
                           + df['Total Length of Bwd Packet'].fillna(0))

    grp = df.groupby('LogicalFlowID', sort=False)
    g = grp.size().index  # canonical group order

    out = {}
    # metadata + engineered
    for c in ['source_file', LABEL_COL, FLOW_ID_COL, 'Src IP', 'Src Port',
              'Dst IP', 'Dst Port', 'Protocol']:
        out[c] = grp[c].first()
    out['SegmentCount'] = grp.size()
    out['ts_first'] = grp['ts'].min()
    out['ts_last'] = grp['ts'].max()

    for c in SUM_COLS:
        out[c] = grp[c].sum(min_count=1)
    for c in MAX_COLS:
        out[c] = grp[c].max()
    for c in MIN_COLS:
        out[c] = grp[c].min()
    for c in FIRST_COLS:
        out[c] = grp[c].first()
    for c, w in WMEAN_COLS.items():
        out[c] = _weighted_mean(df, g, c, w)
    for c, (mean_col, n_col) in POOLED_STD_COLS.items():
        out[c] = _pooled_std(df, g, c, mean_col, n_col)

    result = pd.DataFrame(out)

    # variance = pooled std ** 2
    for var_col, std_col in POOLED_VAR_COLS.items():
        result[var_col] = result[std_col] ** 2

    # recomputed rates (microsecond durations)
    dur = grp[DURATION_COL].sum(min_count=1).replace(0.0, np.nan)
    for c, num_col in RATE_COLS.items():
        num = grp[num_col].sum(min_count=1)
        result[c] = (num / dur * MICROSECONDS_PER_SECOND).fillna(0.0)

    result = result.reset_index(drop=True)
    result['LogicalFlowID'] = list(g)
    return result


def build_unified_flows(path=FLOW_DIR, files=None, cutoff=GAP_CUTOFF_SECONDS):
    """Full pipeline: load every flow CSV, assign LogicalFlowID per file, and
    aggregate to one record per logical flow. Returns the concatenated frame."""
    if files is None:
        files = sorted(glob.glob(os.path.join(path, '*.csv')))
    frames = []
    for f in files:
        raw = load_flow_file(f)
        raw = assign_logical_flow_id(raw, cutoff=cutoff)
        frames.append(aggregate_logical_flows(raw))
    return pd.concat(frames, ignore_index=True)


def feature_dictionary():
    """feature -> (rule, justification) — report material."""
    d = {}
    for c in SUM_COLS:
        d[c] = ('sum', 'additive count/total/active-duration across segments')
    for c in MAX_COLS:
        d[c] = ('max', 'extreme of per-segment maxima')
    for c in MIN_COLS:
        d[c] = ('min', 'extreme of per-segment minima')
    for c in FIRST_COLS:
        d[c] = ('first', 'connection-setup property fixed at handshake')
    for c, w in WMEAN_COLS.items():
        d[c] = (f'weighted mean (w={w})', 'mean weighted by packet count; exact for single-segment flows')
    for c, (m, n) in POOLED_STD_COLS.items():
        d[c] = ('pooled std', f'pooled variance from per-segment mean={m}, n={n}')
    for c, s in POOLED_VAR_COLS.items():
        d[c] = ('pooled var', f'(pooled {s})**2')
    for c, num in RATE_COLS.items():
        d[c] = ('recomputed rate', f'sum({num}) / sum(duration_us) * 1e6')
    for c in DROP_COLS:
        d[c] = ('dropped (Tier 3)', 'pooled sufficient statistics unavailable (no inter-segment gaps)')
    return d
