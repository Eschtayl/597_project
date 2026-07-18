"""Integrity + leakage tests for LogicalFlowID construction and aggregation. 
Runs on real flow captures and prints PASS/FAIL.

Usage:  python phase_3/check_aggregation.py [file1.csv file2.csv ...]
Defaults to the benign + DNS captures (benign holds the 278-segment flow).
"""
import os
import sys
import glob

import numpy as np
import pandas as pd

from config import FLOW_DIR, W_FWD, W_BWD, GAP_CUTOFF_SECONDS
from flow_aggregation import (
    load_flow_file,
    assign_logical_flow_id,
    aggregate_logical_flows,
    SUM_COLS,
    MAX_COLS,
    MIN_COLS,
    WMEAN_COLS,
)

PASS, FAIL = 'PASS', 'FAIL'
results = []


def check(name, ok, detail=''):
    tag = PASS if ok else FAIL
    results.append(ok)
    print(f'  [{tag}] {name}' + (f'  -- {detail}' if detail else ''))


def default_files():
    cands = [
        os.path.join(FLOW_DIR, 'BenignTraffic.pcap_Flow.csv'),
        os.path.join(FLOW_DIR, 'DNS_Spoofing.pcap_Flow.csv'),
    ]
    have = [c for c in cands if os.path.exists(c)]
    return have or sorted(glob.glob(os.path.join(FLOW_DIR, '*.csv')))[:2]


def main():
    files = sys.argv[1:] or default_files()
    print(f'Cutoff = {GAP_CUTOFF_SECONDS}s. Files:')
    for f in files:
        print(f'  - {os.path.basename(f)}')

    raw_frames, agg_frames = [], []
    for f in files:
        raw = assign_logical_flow_id(load_flow_file(f))
        agg = aggregate_logical_flows(raw)
        raw_frames.append(raw)
        agg_frames.append(agg)
    raw = pd.concat(raw_frames, ignore_index=True)
    agg = pd.concat(agg_frames, ignore_index=True)
    print(f'\nRaw segment rows: {len(raw):,}   Unified logical flows: {len(agg):,}\n')

    # 1. every raw row -> exactly one LogicalFlowID
    check('every raw row has a LogicalFlowID', raw['LogicalFlowID'].notna().all())

    # 2. segment counts reconcile
    check('sum(SegmentCount) == raw rows',
          int(agg['SegmentCount'].sum()) == len(raw),
          f"{int(agg['SegmentCount'].sum()):,} vs {len(raw):,}")

    # 3. LogicalFlowID is unique in the aggregate
    check('LogicalFlowID unique after aggregation',
          agg['LogicalFlowID'].is_unique)

    # 4. each LogicalFlowID maps to a single (source_file, Flow ID)
    per = raw.groupby('LogicalFlowID').agg(nfile=('source_file', 'nunique'),
                                           nfid=('Flow ID', 'nunique'))
    check('each logical flow has one source_file and one Flow ID',
          (per['nfile'] == 1).all() and (per['nfid'] == 1).all())

    # 5. a genuinely contiguous long flow is preserved intact: the largest
    #    LOGICAL flow must have every internal gap within the cutoff (i.e. we
    #    kept it together for the right reason, not by accident).
    biggest_lfid = agg.loc[agg['SegmentCount'].idxmax(), 'LogicalFlowID']
    seg_rows = raw[raw['LogicalFlowID'] == biggest_lfid].sort_values('ts')
    internal_gaps = seg_rows['ts'].diff().dt.total_seconds().dropna()
    check('largest logical flow is internally contiguous (all gaps <= cutoff)',
          len(internal_gaps) > 0 and internal_gaps.max() <= GAP_CUTOFF_SECONDS,
          f'{len(seg_rows)} segments, max internal gap={internal_gaps.max():.0f}s')

    # 6. large-gap reuse is separated: some Flow ID yields >1 LogicalFlowID
    fids_split = (raw.groupby('Flow ID')['LogicalFlowID'].nunique() > 1).sum()
    check('ID reuse is separated (some Flow IDs -> multiple logical flows)',
          fids_split > 0, f'{fids_split:,} Flow IDs split into multiple logical flows')

    # 7. single-segment identity: agg feature == raw value for sum/min/max/wmean
    single_ids = agg.loc[agg['SegmentCount'] == 1, 'LogicalFlowID']
    r1 = raw[raw['LogicalFlowID'].isin(single_ids)].set_index('LogicalFlowID').sort_index()
    a1 = agg[agg['LogicalFlowID'].isin(single_ids)].set_index('LogicalFlowID').sort_index()
    idcheck = list(SUM_COLS) + list(MAX_COLS) + list(MIN_COLS) + list(WMEAN_COLS.keys())
    max_abs = 0.0
    for c in idcheck:
        rv = pd.to_numeric(r1[c], errors='coerce').fillna(0.0).to_numpy()
        av = pd.to_numeric(a1[c], errors='coerce').fillna(0.0).to_numpy()
        d = np.nanmax(np.abs(rv - av)) if len(rv) else 0.0
        max_abs = max(max_abs, d)
    check('single-segment aggregation is identity (sum/min/max/wmean)',
          max_abs < 1e-6, f'max abs diff = {max_abs:.2e} over {len(a1):,} flows')

    # 8. recomputed rate matches raw for single-segment flows
    rate_ok = True
    detail8 = ''
    for c in ['Flow Bytes/s', 'Flow Packets/s']:
        rv = pd.to_numeric(r1[c], errors='coerce').to_numpy()
        av = pd.to_numeric(a1[c], errors='coerce').to_numpy()
        m = np.isfinite(rv) & np.isfinite(av) & (np.abs(rv) > 1e-9)
        rel = np.abs(rv[m] - av[m]) / np.abs(rv[m])
        p = float(np.nanpercentile(rel, 99)) if m.any() else 0.0
        detail8 += f'{c} p99 rel-err={p:.1e}  '
        rate_ok = rate_ok and p < 1e-3
    check('recomputed rates match raw for single-segment flows', rate_ok, detail8.strip())

    # 9. active duration = sum of segment durations (by construction; verify a sample)
    sample = agg.sample(min(500, len(agg)), random_state=0)['LogicalFlowID']
    man = raw[raw['LogicalFlowID'].isin(sample)].groupby('LogicalFlowID')['Flow Duration'].sum()
    got = agg.set_index('LogicalFlowID').loc[sample, 'Flow Duration']
    check('Flow Duration == sum of segment durations',
          np.allclose(man.sort_index().to_numpy(), got.sort_index().to_numpy(), rtol=1e-6))

    # 10. pooled multi-segment std is finite and non-negative
    multi = agg[agg['SegmentCount'] > 1]
    stds = multi[['Fwd Packet Length Std', 'Bwd Packet Length Std', 'Packet Length Std']]
    check('pooled multi-segment stds are finite and >= 0',
          np.isfinite(stds.to_numpy()).all() and (stds.to_numpy() >= 0).all(),
          f'{len(multi):,} multi-segment flows checked')

    print('\n' + ('ALL CHECKS PASSED' if all(results) else 'SOME CHECKS FAILED')
          + f'  ({sum(results)}/{len(results)})')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
