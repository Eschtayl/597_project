"""Cascade alert->flow alignment + mandatory leakage guard.

Implements the LOCKED packet-centric protocol:

  Phase 2 val/test packets -> keep flagged (alert) packets
    -> map each alert against the FULL unified-flow index (NOT the 204k sample)
    -> preserve the full candidate LogicalFlowID list per packet (Policy B needs it)
    -> classify mapping status, with invalid_key split into subcategories
    -> leakage guard: candidate flows reachable from val AND test alerts must not
       intersect Phase 3 train (and test candidates must not intersect Phase 3 val)

Outputs:
  phase_3/data/unified_flows_full.parquet   (cached full unified-flow frame)
  phase_3/data/cascade_alignment.parquet    (one row per val/test alert packet)
  phase_3/data/cascade_leakage.json         (guard verdict + overlap LFID lists)

Usage: python phase_3/cascade_alignment.py
"""
import os
import sys
import json

import numpy as np
import pandas as pd

from config import DATA_DIR, PHASE3_DIR, LABEL_COL, BENIGN_LABEL
from flow_aggregation import build_unified_flows
from flow_sampling import load_sampled_dataset
from features import build_datasets
from packet_flow_mapping import (
    canonical_key, packet_protocol,
    UNIQUE_MATCH, MULTI_SAME_LABEL, MULTI_SAME_CAPTURE, MULTI_CONFLICTING,
    NO_MATCH, INVALID_KEY,
)

UNIFIED_FULL_PATH = os.path.join(DATA_DIR, 'unified_flows_full.parquet')
ALIGNMENT_PATH = os.path.join(DATA_DIR, 'cascade_alignment.parquet')
LEAKAGE_PATH = os.path.join(DATA_DIR, 'cascade_leakage.json')
COHORT_PATH = os.path.join(DATA_DIR, 'phase2_cohort.parquet')
CAND_SEP = ';;'   # LogicalFlowID contains '|' and '#', so use a distinct separator

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
_report = []
def emit(s=''):
    print(s)
    _report.append(s)


def load_or_build_unified():
    if os.path.exists(UNIFIED_FULL_PATH):
        emit(f'Loading cached full unified flows: {UNIFIED_FULL_PATH}')
        return pd.read_parquet(UNIFIED_FULL_PATH)
    emit('Building FULL unified-flow frame (all captures — several minutes)...')
    unified = build_unified_flows()
    os.makedirs(DATA_DIR, exist_ok=True)
    unified.to_parquet(UNIFIED_FULL_PATH)
    emit(f'  built {len(unified):,} unified flows -> cached {UNIFIED_FULL_PATH}')
    return unified


def build_candidate_index(unified):
    """canonical_key -> candidate stats + full LogicalFlowID list (CAND_SEP-joined)."""
    key, valid = canonical_key(unified['Src IP'], unified['Src Port'],
                               unified['Dst IP'], unified['Dst Port'], unified['Protocol'])
    flows = unified.loc[valid, ['LogicalFlowID', LABEL_COL, 'source_file']].copy()
    flows['canonical_key'] = key[valid]
    grp = flows.groupby('canonical_key', sort=False)
    index = pd.DataFrame({
        'n_candidates': grp['LogicalFlowID'].size(),
        'n_labels': grp[LABEL_COL].nunique(),
        'n_captures': grp['source_file'].nunique(),
        'candidates': grp['LogicalFlowID'].agg(CAND_SEP.join),
    })
    emit(f'  flow key index: {len(flows):,} keyable flows -> {len(index):,} canonical keys '
         f'({(index.n_candidates > 1).sum():,} keys with >1 candidate)')
    return index


def classify_alert_packets(alerts, index):
    """Attach canonical key, mapping status (+ invalid subcategory), candidate
    stats and the full candidate list to each alert packet row."""
    p = alerts.copy().reset_index(drop=True)
    proto = packet_protocol(p['l4_tcp'], p['l4_udp'])

    def bad_ip(col):
        s = col.astype(str)
        return col.isna() | (s.str.len() == 0) | (s == 'nan')
    missing_endpoint = bad_ip(p['src_ip']) | bad_ip(p['dst_ip'])
    zero_port = (pd.to_numeric(p['src_port'], errors='coerce').fillna(0) <= 0) | \
                (pd.to_numeric(p['dst_port'], errors='coerce').fillna(0) <= 0)
    unsupported_protocol = ~proto.isin([6, 17])

    key, valid = canonical_key(p['src_ip'], p['src_port'], p['dst_ip'], p['dst_port'], proto)
    p['canonical_key'] = key
    joined = p.join(index, on='canonical_key')

    n = joined['n_candidates']
    status = pd.Series(NO_MATCH, index=p.index, dtype=object)
    status[n == 1] = UNIQUE_MATCH
    multi = n > 1
    status[multi & (joined['n_labels'] > 1)] = MULTI_CONFLICTING
    status[multi & (joined['n_labels'] == 1) & (joined['n_captures'] > 1)] = MULTI_SAME_LABEL
    status[multi & (joined['n_labels'] == 1) & (joined['n_captures'] == 1)] = MULTI_SAME_CAPTURE
    status[~valid] = INVALID_KEY

    invalid_reason = pd.Series('', index=p.index, dtype=object)
    invalid_reason[~valid & unsupported_protocol] = 'unsupported_protocol'
    invalid_reason[~valid & zero_port] = 'zero_port'
    invalid_reason[~valid & missing_endpoint] = 'missing_endpoint'

    p['status'] = status
    p['invalid_reason'] = invalid_reason
    p['n_candidates'] = joined['n_candidates'].fillna(0).astype(int)
    p['n_captures'] = joined['n_captures'].fillna(0).astype(int)
    p['candidates'] = joined['candidates'].where(status != INVALID_KEY, other=None)
    return p


def coverage_report(aligned):
    for part in ['val', 'test']:
        sub = aligned[aligned['phase2_split'] == part]
        emit(f'\nMapping coverage — Phase 2 {part} alerts (n={len(sub):,}):')
        counts = sub['status'].value_counts()
        for st, c in counts.items():
            emit(f'  {st:<28} {c:>6,}  ({100*c/len(sub):5.2f}%)')
        inv = sub[sub['status'] == INVALID_KEY]['invalid_reason'].value_counts()
        for r, c in inv.items():
            emit(f'    invalid_key/{r:<22} {c:>6,}')
        emit(f'  per-label unique/no_match rates:')
        for lab, g in sub.groupby('label'):
            u = (g['status'] == UNIQUE_MATCH).mean()
            nm = (g['status'] == NO_MATCH).mean()
            usable = g['status'].isin([UNIQUE_MATCH, MULTI_SAME_CAPTURE]).mean()
            emit(f'    {lab:<18} n={len(g):>5,}  unique={100*u:5.1f}%  '
                 f'no_match={100*nm:5.1f}%  usable(A|B)={100*usable:5.1f}%')


def candidate_lfid_set(aligned_part):
    """Union of all candidate LogicalFlowIDs reachable from these alerts
    (ALL mapping statuses, per the mandatory guard)."""
    out = set()
    for s in aligned_part['candidates'].dropna():
        out.update(s.split(CAND_SEP))
    return out


def main():
    unified = load_or_build_unified()
    index = build_candidate_index(unified)

    cohort = pd.read_parquet(COHORT_PATH)
    alerts = cohort[(cohort['phase2_split'].isin(['val', 'test'])) & (cohort['y2_alert'] == 1)]
    emit(f'\nPhase 2 alert packets: val={int((alerts.phase2_split=="val").sum()):,} '
         f'test={int((alerts.phase2_split=="test").sum()):,}')

    aligned = classify_alert_packets(alerts, index)
    coverage_report(aligned)
    aligned.to_parquet(ALIGNMENT_PATH)
    emit(f'\nSaved alignment -> {ALIGNMENT_PATH}')

    # ------------------------------------------------------------------
    # MANDATORY leakage guard
    # ------------------------------------------------------------------
    emit('\n=== Leakage guard (cascade candidates vs Phase 3 sample) ===')
    sampled = load_sampled_dataset()
    data = build_datasets(sampled)
    diag = data['diagnostics']
    part_lfids = {p: set(diag.loc[diag['split'] == p, 'LogicalFlowID'])
                  for p in ['train', 'val', 'test']}

    c_val = candidate_lfid_set(aligned[aligned['phase2_split'] == 'val'])
    c_test = candidate_lfid_set(aligned[aligned['phase2_split'] == 'test'])
    emit(f'candidate flows reachable from val alerts:  {len(c_val):,}')
    emit(f'candidate flows reachable from test alerts: {len(c_test):,}')

    checks = {
        'cascade_val_x_train': sorted(c_val & part_lfids['train']),
        'cascade_test_x_train': sorted(c_test & part_lfids['train']),
        'cascade_test_x_val': sorted(c_test & part_lfids['val']),
    }
    clean = all(len(v) == 0 for v in checks.values())
    for name, ov in checks.items():
        emit(f'  [{"PASS" if not ov else "FAIL"}] {name} = {len(ov):,} overlapping flows')
    if clean:
        emit('Guard PASSED — heads may be used as-is (preprocessors must still be refit for scoring).')
    else:
        emit('Guard FAILED (expected — samples were independent). Next: remove candidate flows '
             'from Phase 3 training, resample replacements, retrain heads with locked params '
             '(cascade_heads.py).')

    with open(LEAKAGE_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'clean': clean,
            'n_candidates_val': len(c_val),
            'n_candidates_test': len(c_test),
            'overlaps': {k: v for k, v in checks.items()},
        }, f, indent=1)
    emit(f'Saved guard verdict -> {LEAKAGE_PATH}')

    with open(os.path.join(PHASE3_DIR, 'cascade_alignment_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report) + '\n')


if __name__ == '__main__':
    main()
