"""Feature builders: behaviour-only and service-aware model matrices, plus the
frozen diagnostic partitions. Both model matrices are built
from the same sampled rows and the same split/fold assignments.

Key leakage rules:
  * behaviour-only excludes identity, ports, protocol, timestamps, labels, and all
    diagnostic buckets.
  * service-aware adds only *universal* transformed service context (protocol,
    service categories, port-range categories, HTTP/DNS/well-known indicators) —
    never exact ports, never `service_aligned` (which is capture-label-derived and
    would leak the target), never the Brute Force diagnostic bucket.
  * diagnostics (`service_aligned`, Brute Force buckets) are computed and saved for
    later sensitivity analysis but are never model features.
"""
import numpy as np
import pandas as pd

from config import LABEL_COL, BENIGN_LABEL
from flow_aggregation import (
    SUM_COLS, MAX_COLS, MIN_COLS, FIRST_COLS, WMEAN_COLS,
    POOLED_STD_COLS, POOLED_VAR_COLS, RATE_COLS,
)
from splits import grouped_stratified_split, blocked_temporal_folds, SPLIT_COL, FOLD_COL

EPS = 1e-9
_HTTP_PORTS = {80, 8080, 443}
_DNS_PORTS = {53}
_KNOWN_SERVICE = {80: 'http', 8080: 'http', 443: 'https', 53: 'dns'}

ENGINEERED_NUMERIC = [
    'BytesPerPacket', 'FwdBwdPacketRatio', 'FwdBwdByteRatio', 'BackwardFraction',
    'SynAckRatio', 'LogDuration', 'LogTotalPackets', 'LogTotalBytes',
]
# behaviour numeric = every aggregated CICFlowMeter statistic + SegmentCount + engineered
BEHAVIOUR_NUMERIC = (
    SUM_COLS + MAX_COLS + MIN_COLS + FIRST_COLS + list(WMEAN_COLS)
    + list(POOLED_STD_COLS) + list(POOLED_VAR_COLS) + list(RATE_COLS)
    + ['SegmentCount'] + ENGINEERED_NUMERIC
)
SERVICE_CATEGORICAL = ['Protocol', 'KnownServiceCategory', 'SourcePortRange', 'DestinationPortRange']
SERVICE_BINARY = ['IsHTTPService', 'IsDNSService', 'IsWellKnownDestinationPort']

# Columns that must never appear in any model matrix
FORBIDDEN_IN_MATRIX = [
    'Flow ID', 'LogicalFlowID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port',
    'Timestamp', 'ts_first', 'ts_last', 'source_file', LABEL_COL, 'Label',
    'service_aligned', 'behaviorally_attack_like', 'behavioral_bucket',
    'attempts_same_target_60s', 'attempts_same_target_300s',
    'short_flow', 'low_response', 'syn_dominant', 'repetition_extreme',
    SPLIT_COL, FOLD_COL,
]


# ---------------------------------------------------------------------------
# Engineered behavioural features (per-flow, no identity)
# ---------------------------------------------------------------------------
def add_engineered_behaviour(df):
    df = df.copy()
    fwd_p = df['Total Fwd Packet'].fillna(0)
    bwd_p = df['Total Bwd packets'].fillna(0)
    fwd_b = df['Total Length of Fwd Packet'].fillna(0)
    bwd_b = df['Total Length of Bwd Packet'].fillna(0)
    tot_p = fwd_p + bwd_p
    tot_b = fwd_b + bwd_b
    df['BytesPerPacket'] = tot_b / (tot_p + EPS)
    df['FwdBwdPacketRatio'] = (fwd_p + EPS) / (bwd_p + EPS)
    df['FwdBwdByteRatio'] = (fwd_b + EPS) / (bwd_b + EPS)
    df['BackwardFraction'] = bwd_p / (tot_p + EPS)
    df['SynAckRatio'] = (df['SYN Flag Count'].fillna(0) + 1) / (df['ACK Flag Count'].fillna(0) + 1)
    df['LogDuration'] = np.log1p(df['Flow Duration'].clip(lower=0).fillna(0))
    df['LogTotalPackets'] = np.log1p(tot_p)
    df['LogTotalBytes'] = np.log1p(tot_b)
    return df


# ---------------------------------------------------------------------------
# Service context (universal port semantics — not label-derived)
# ---------------------------------------------------------------------------
def _port_range(p):
    p = pd.to_numeric(p, errors='coerce')
    out = np.select(
        [p.isna() | (p <= 0), p <= 1023, p <= 49151],
        ['none', 'well_known', 'registered'],
        default='ephemeral',
    )
    return pd.Series(out, index=p.index)


def add_service_fields(df):
    df = df.copy()
    src = pd.to_numeric(df['Src Port'], errors='coerce')
    dst = pd.to_numeric(df['Dst Port'], errors='coerce')
    df['Protocol'] = pd.to_numeric(df['Protocol'], errors='coerce').fillna(0).astype(int).astype(str)
    svc = dst.map(_KNOWN_SERVICE)
    svc = svc.fillna(src.map(_KNOWN_SERVICE)).fillna('other')
    df['KnownServiceCategory'] = svc.values
    df['SourcePortRange'] = _port_range(src)
    df['DestinationPortRange'] = _port_range(dst)
    df['IsHTTPService'] = (src.isin(_HTTP_PORTS) | dst.isin(_HTTP_PORTS)).astype(int)
    df['IsDNSService'] = (src.isin(_DNS_PORTS) | dst.isin(_DNS_PORTS)).astype(int)
    df['IsWellKnownDestinationPort'] = ((dst > 0) & (dst <= 1023)).astype(int)
    return df


# ---------------------------------------------------------------------------
# Diagnostics (analysis only — never features)
# ---------------------------------------------------------------------------
def compute_service_aligned(df):
    src = pd.to_numeric(df['Src Port'], errors='coerce')
    dst = pd.to_numeric(df['Dst Port'], errors='coerce')
    proto = pd.to_numeric(df['Protocol'], errors='coerce')
    tcp, udp = proto == 6, proto == 17
    http = src.isin({80, 8080}) | dst.isin({80, 8080})
    xss = src.isin({80, 443}) | dst.isin({80, 443})
    dns = (src == 53) | (dst == 53)
    lab = df[LABEL_COL]
    out = pd.Series('unaligned', index=df.index)
    out[lab == BENIGN_LABEL] = 'benign'
    out[lab == 'brute_force'] = 'service_indeterminate'
    out[(lab.isin(['DoS-HTTP', 'DDOS-HTTP_flood'])) & tcp & http] = 'aligned'
    out[(lab == 'XSS') & tcp & xss] = 'aligned'
    out[(lab == 'DNS_spoofing') & (tcp | udp) & dns] = 'aligned'
    return out


def _rolling_attempts(df, window_s):
    d = df[['Src IP', 'Dst IP']].copy()
    d['t'] = df['ts_first'].astype('int64')  # ns since epoch
    d = d.sort_values(['Src IP', 'Dst IP', 't'])
    w = int(window_s) * 1_000_000_000
    counts = np.empty(len(d), dtype=np.int64)
    pos = 0
    for _, g in d.groupby(['Src IP', 'Dst IP'], sort=False):
        t = g['t'].to_numpy()
        left = np.searchsorted(t, t - w, side='left')
        counts[pos:pos + len(t)] = np.arange(1, len(t) + 1) - left
        pos += len(t)
    return pd.Series(counts, index=d.index).reindex(df.index)


def compute_bruteforce_diagnostics(df, train_mask=None):
    """Add Brute Force diagnostic fields. Thresholds are benign quantiles computed
    from the benign flows inside `train_mask` only (default: `split == 'train'`).
    For temporal folds, pass that fold's benign training-block mask so thresholds
    are recomputed per fold rather than reused from the primary split."""
    df = df.copy()
    df['attempts_same_target_60s'] = _rolling_attempts(df, 60)
    df['attempts_same_target_300s'] = _rolling_attempts(df, 300)

    if train_mask is None:
        train_mask = df[SPLIT_COL] == 'train'
    benign_train = df[(df[LABEL_COL] == BENIGN_LABEL) & train_mask]
    thr = {
        'short_flow_p25_duration': float(benign_train['Flow Duration'].quantile(0.25)),
        'low_response_p10_bwdfrac': float(benign_train['BackwardFraction'].quantile(0.10)),
        'syn_dominant_p90_synack': float(benign_train['SynAckRatio'].quantile(0.90)),
        'repetition_extreme_p99_attempts60': float(benign_train['attempts_same_target_60s'].quantile(0.99)),
    }
    df['short_flow'] = df['Flow Duration'] <= thr['short_flow_p25_duration']
    df['low_response'] = df['BackwardFraction'] <= thr['low_response_p10_bwdfrac']
    df['syn_dominant'] = df['SynAckRatio'] >= thr['syn_dominant_p90_synack']
    df['repetition_extreme'] = df['attempts_same_target_60s'] >= thr['repetition_extreme_p99_attempts60']
    combo = df[['short_flow', 'low_response', 'syn_dominant']].sum(axis=1)
    df['behaviorally_attack_like'] = df['repetition_extreme'] & (combo >= 2)
    df['behavioral_bucket'] = np.where(df['behaviorally_attack_like'],
                                       'behaviorally_attack_like', 'behaviorally_ambiguous')
    return df, thr


# ---------------------------------------------------------------------------
# Assemble everything
# ---------------------------------------------------------------------------
def build_datasets(sampled, seed=None, split_by_lfid=None):
    """Return a dict with X_behaviour, X_service, y_binary, y_multiclass, split,
    fold, and the diagnostics frame — all row-aligned.

    `split_by_lfid`: optional {LogicalFlowID: 'train'|'val'|'test'} mapping. When
    given, split assignments are taken from it instead of re-running the grouped
    stratified split (used by the cascade leakage-guard retrain, which must
    preserve the original partition for retained flows)."""
    df = sampled.reset_index(drop=True).copy()
    df = add_engineered_behaviour(df)
    df = add_service_fields(df)

    if split_by_lfid is not None:
        df[SPLIT_COL] = df['LogicalFlowID'].map(split_by_lfid)
        if df[SPLIT_COL].isna().any():
            missing = df.loc[df[SPLIT_COL].isna(), 'LogicalFlowID'].head(3).tolist()
            raise ValueError(f'split_by_lfid missing assignments, e.g. {missing}')
    else:
        df = grouped_stratified_split(df)            # adds SPLIT_COL
    df[FOLD_COL] = blocked_temporal_folds(df)[FOLD_COL].values

    df, bf_thresholds = compute_bruteforce_diagnostics(df)
    df['service_aligned'] = compute_service_aligned(df)

    y_multi = df[LABEL_COL]
    y_binary = (df[LABEL_COL] != BENIGN_LABEL).astype(int)

    X_behaviour = df[BEHAVIOUR_NUMERIC].copy()
    X_service = df[BEHAVIOUR_NUMERIC + SERVICE_CATEGORICAL + SERVICE_BINARY].copy()

    diagnostics = df[[
        'LogicalFlowID', LABEL_COL, SPLIT_COL, FOLD_COL, 'service_aligned',
        'behavioral_bucket', 'attempts_same_target_60s', 'attempts_same_target_300s',
        'short_flow', 'low_response', 'syn_dominant', 'repetition_extreme',
    ]].copy()

    return {
        'X_behaviour': X_behaviour,
        'X_service': X_service,
        'y_binary': y_binary,
        'y_multiclass': y_multi,
        'split': df[SPLIT_COL],
        'fold': df[FOLD_COL],
        'diagnostics': diagnostics,
        'bf_thresholds': bf_thresholds,
    }


# ---------------------------------------------------------------------------
# Train-only preprocessing
# ---------------------------------------------------------------------------
def clean_frame(X):
    """Stateless: map +/-inf and -1 sentinels to NaN (consistent with Phase 1)."""
    return X.replace([np.inf, -np.inf], np.nan).replace(-1, np.nan)


def build_preprocessor(numeric_cols, categorical_cols, model_kind):
    """ColumnTransformer (unfitted). model_kind: 'linear' scales numeric; 'tree'
    leaves numeric unscaled. Categoricals one-hot with handle_unknown='ignore'.
    Median imputation + missing indicators are fitted on train (by the caller)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    num_steps = [('impute', SimpleImputer(strategy='median', add_indicator=True))]
    if model_kind == 'linear':
        num_steps.append(('scale', StandardScaler()))
    transformers = [('num', Pipeline(num_steps), numeric_cols)]
    if categorical_cols:
        cat = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat', cat, categorical_cols))
    return ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)


def numeric_categorical_cols(variant):
    if variant == 'behaviour':
        return BEHAVIOUR_NUMERIC, []
    if variant == 'service':
        return BEHAVIOUR_NUMERIC + SERVICE_BINARY, SERVICE_CATEGORICAL
    raise ValueError(variant)


def feature_reason_table():
    """Feature | behaviour-only | service-aware | reason (report material)."""
    rows = []
    for c in BEHAVIOUR_NUMERIC:
        rows.append((c, 'Yes', 'Yes', 'behaviour'))
    for c in SERVICE_CATEGORICAL + SERVICE_BINARY:
        rows.append((c, 'No', 'Yes', 'controlled service context (universal port semantics)'))
    rows.append(('Src IP / Dst IP', 'No', 'No', 'identity leakage'))
    rows.append(('Src Port / Dst Port (exact)', 'No', 'No', 'shortcut risk; ephemeral'))
    rows.append(('Timestamp', 'No', 'No', 'identity / temporal leakage'))
    rows.append(('service_aligned', 'No', 'No', 'capture-label-derived diagnostic'))
    rows.append(('behavioral_bucket', 'No', 'No', 'diagnostic partition'))
    rows.append(('label / NeedManualLabel', 'No', 'No', 'target'))
    return pd.DataFrame(rows, columns=['Feature', 'BehaviourOnly', 'ServiceAware', 'Reason'])
