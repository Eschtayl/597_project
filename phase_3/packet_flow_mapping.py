"""Packet -> logical-flow mapping via a canonical bidirectional 5-field key.

Packets carry no absolute timestamp, so they cannot be resolved to a
specific temporal cluster. We therefore map a packet's connection 5-tuple to the
*set* of logical flows sharing that canonical key and classify the outcome; the
cascade only trusts an unambiguous (unique) match and otherwise retains the
Phase 2 alert.

The canonical key sorts the two endpoints deterministically so that reversing
source/destination yields the same key:

    endpoint_1 = (ip_1, port_1);  endpoint_2 = (ip_2, port_2)
    canonical_key = (min(e1, e2), max(e1, e2), protocol)

Labels are used only to *categorise* ambiguity (same vs conflicting proxy
labels) for the quality report — never to choose among candidate flows.
"""
import os
import glob

import numpy as np
import pandas as pd

from config import LABEL_COL
from flow_aggregation import assign_logical_flow_id, label_from_filename

# Mapping status categories
UNIQUE_MATCH = 'unique_match'
MULTI_SAME_LABEL = 'multiple_same_label'
MULTI_SAME_CAPTURE = 'multiple_same_capture'
MULTI_CONFLICTING = 'multiple_conflicting_labels'
NO_MATCH = 'no_match'
INVALID_KEY = 'invalid_key'

# Statuses whose flow prediction the strict cascade (Policy A) may act on; every
# other status falls back to retaining the Phase 2 alert. Policy B additionally
# evaluates the whole candidate set for MULTI_SAME_CAPTURE — see cascade_eval.py.
TRUSTED_STATUSES = {UNIQUE_MATCH}

_FLOW_KEY_COLS = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp']


# ---------------------------------------------------------------------------
# Canonical key
# ---------------------------------------------------------------------------
def _valid(ip_a, port_a, ip_b, port_b, proto):
    def ok_ip(s):
        return s.notna() & (s.astype(str).str.len() > 0) & (s.astype(str) != 'nan')
    return (
        ok_ip(ip_a) & ok_ip(ip_b)
        & (port_a > 0) & (port_b > 0)
        & proto.isin([6, 17])
    )


def canonical_key(ip_a, port_a, ip_b, port_b, proto):
    """Vectorised canonical key (Series of strings) + validity mask (Series of
    bool). Endpoints are sorted lexicographically as 'ip:port' tokens so the key
    is invariant to source/destination direction."""
    ip_a, ip_b = ip_a.astype(str), ip_b.astype(str)
    pa = pd.to_numeric(port_a, errors='coerce')
    pb = pd.to_numeric(port_b, errors='coerce')
    pr = pd.to_numeric(proto, errors='coerce')
    valid = _valid(ip_a, pa, ip_b, pb, pr)

    tok_a = ip_a + ':' + pa.astype('Int64').astype(str)
    tok_b = ip_b + ':' + pb.astype('Int64').astype(str)
    lo = np.where(tok_a.values <= tok_b.values, tok_a.values, tok_b.values)
    hi = np.where(tok_a.values <= tok_b.values, tok_b.values, tok_a.values)
    key = pd.Series(lo, index=ip_a.index) + '||' + hi + '||' + pr.astype('Int64').astype(str)
    key = key.where(valid, other=np.nan)
    return key, valid


def packet_protocol(l4_tcp, l4_udp):
    """Derive IANA protocol number from packet TCP/UDP flags (6 / 17 / 0)."""
    proto = np.where(pd.to_numeric(l4_tcp, errors='coerce') == 1, 6,
                     np.where(pd.to_numeric(l4_udp, errors='coerce') == 1, 17, 0))
    return pd.Series(proto, index=l4_tcp.index)


# ---------------------------------------------------------------------------
# Flow-side key index (lightweight — no feature aggregation needed, since every
# segment of a logical flow shares the 5-tuple)
# ---------------------------------------------------------------------------
def build_flow_key_index(flow_files):
    """From flow CSVs, assign LogicalFlowID and build a canonical_key -> candidate
    index: one row per key with candidate count, label/capture diversity, and a
    representative LogicalFlowID (used only for unique matches)."""
    per_flow = []
    for f in flow_files:
        df = pd.read_csv(f, usecols=_FLOW_KEY_COLS, low_memory=False)
        df[LABEL_COL] = label_from_filename(f)
        df['source_file'] = os.path.basename(f)
        df = assign_logical_flow_id(df)
        one = df.groupby('LogicalFlowID', sort=False).first().reset_index()
        key, valid = canonical_key(one['Src IP'], one['Src Port'],
                                   one['Dst IP'], one['Dst Port'], one['Protocol'])
        one['canonical_key'] = key
        per_flow.append(one.loc[valid, ['canonical_key', 'LogicalFlowID', LABEL_COL, 'source_file']])
    flows = pd.concat(per_flow, ignore_index=True)

    grp = flows.groupby('canonical_key', sort=False)
    index = pd.DataFrame({
        'n_candidates': grp['LogicalFlowID'].size(),
        'n_labels': grp[LABEL_COL].nunique(),
        'n_captures': grp['source_file'].nunique(),
        'rep_label': grp[LABEL_COL].first(),
        'rep_lfid': grp['LogicalFlowID'].first(),
    })
    return index


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _status(row):
    n = row['n_candidates']
    if pd.isna(n):
        return NO_MATCH
    if n == 1:
        return UNIQUE_MATCH
    if row['n_labels'] > 1:
        return MULTI_CONFLICTING
    if row['n_captures'] > 1:
        return MULTI_SAME_LABEL      # same proxy label, different captures
    return MULTI_SAME_CAPTURE        # same label, one capture -> temporal reuse


def classify_packets(packet_df, flow_index):
    """Return packet_df with columns: canonical_key, packet label, mapping
    `status`, `n_candidates`, and `matched_lfid` (only for unique matches)."""
    p = packet_df.copy()
    proto = packet_protocol(p['l4_tcp'], p['l4_udp'])
    key, valid = canonical_key(p['src_ip'], p['src_port'], p['dst_ip'], p['dst_port'], proto)
    p['canonical_key'] = key

    joined = p.join(flow_index, on='canonical_key')
    status = joined.apply(_status, axis=1)
    status = status.where(valid, other=INVALID_KEY)   # invalid keys override

    p['status'] = status
    p['n_candidates'] = joined['n_candidates']
    p['matched_lfid'] = np.where(status == UNIQUE_MATCH, joined['rep_lfid'], None)
    return p


def cascade_flow_for_packet(status, matched_lfid):
    """Production cascade rule: trust only a unique match; otherwise retain the
    Phase 2 alert. Returns (logical_flow_id_or_None, retain_flag)."""
    if status in TRUSTED_STATUSES:
        return matched_lfid, False
    return None, True     # no reliable match -> retain Phase 2 alert


# ---------------------------------------------------------------------------
# Verification helpers (used by the quality report)
# ---------------------------------------------------------------------------
def verify_direction_symmetry(sample_df):
    """Reversing src/dst must produce the same canonical key."""
    proto = packet_protocol(sample_df['l4_tcp'], sample_df['l4_udp'])
    k1, _ = canonical_key(sample_df['src_ip'], sample_df['src_port'],
                          sample_df['dst_ip'], sample_df['dst_port'], proto)
    k2, _ = canonical_key(sample_df['dst_ip'], sample_df['dst_port'],
                          sample_df['src_ip'], sample_df['src_port'], proto)
    both_valid = k1.notna() & k2.notna()
    return bool((k1[both_valid] == k2[both_valid]).all()) and both_valid.any()


def verify_protocol_in_key():
    """Same endpoints, different protocol -> different key."""
    df = pd.DataFrame({'ip_a': ['10.0.0.1'], 'pa': [1234], 'ip_b': ['10.0.0.2'], 'pb': [80]})
    k_tcp, _ = canonical_key(df['ip_a'], df['pa'], df['ip_b'], df['pb'], pd.Series([6]))
    k_udp, _ = canonical_key(df['ip_a'], df['pa'], df['ip_b'], df['pb'], pd.Series([17]))
    return bool(k_tcp.iloc[0] != k_udp.iloc[0])


def default_capture_pairs(packet_dir, flow_dir):
    """Pair packet CSVs with their flow CSV (<stem>.csv <-> <stem>.pcap_Flow.csv)."""
    pairs = []
    for pf in sorted(glob.glob(os.path.join(packet_dir, '*.csv'))):
        stem = os.path.basename(pf)[:-4]
        ff = os.path.join(flow_dir, stem + '.pcap_Flow.csv')
        if os.path.exists(ff):
            pairs.append((pf, ff))
    return pairs
