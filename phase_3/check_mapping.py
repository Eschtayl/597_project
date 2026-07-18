"""Packet -> logical-flow mapping-quality report.

Builds a canonical-key index from a representative set of flow captures, maps a
sample of packets from the corresponding packet captures, and reports match /
ambiguity / no-match / invalid rates plus the required verifications. Writes to
phase_3/mapping_quality_report.txt and stdout, then STOPS for review.

Usage: python phase_3/check_mapping.py [n_packets_per_capture]
"""
import os
import sys

import numpy as np
import pandas as pd

from config import PACKET_DIR, FLOW_DIR, PHASE3_DIR, BENIGN_LABEL, LABEL_COL
from flow_aggregation import label_from_filename
from packet_flow_mapping import (
    build_flow_key_index, classify_packets, canonical_key, packet_protocol,
    verify_direction_symmetry, verify_protocol_in_key,
    UNIQUE_MATCH, MULTI_SAME_LABEL, MULTI_SAME_CAPTURE, MULTI_CONFLICTING,
    NO_MATCH, INVALID_KEY,
)

# Representative subset: 2 benign captures (enables cross-capture same-label) + all 5 attacks
SUBSET_STEMS = [
    'BenignTraffic', 'BenignTraffic1',
    'DNS_Spoofing', 'XSS', 'DDoS-HTTP_Flood-', 'DictionaryBruteForce', 'DoS-HTTP_Flood',
]
PACKET_COLS = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'l4_tcp', 'l4_udp']
STATUS_ORDER = [UNIQUE_MATCH, MULTI_SAME_LABEL, MULTI_SAME_CAPTURE,
                MULTI_CONFLICTING, NO_MATCH, INVALID_KEY]
SHORT = {UNIQUE_MATCH: 'unique', MULTI_SAME_LABEL: 'multi_lbl', MULTI_SAME_CAPTURE: 'multi_cap',
         MULTI_CONFLICTING: 'conflict', NO_MATCH: 'no_match', INVALID_KEY: 'invalid'}

_out_lines = []
def emit(s=''):
    print(s)
    _out_lines.append(s)


def main():
    n_pkt = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    flow_files = [os.path.join(FLOW_DIR, s + '.pcap_Flow.csv') for s in SUBSET_STEMS]
    flow_files = [f for f in flow_files if os.path.exists(f)]

    emit('=== Packet -> Logical-Flow Mapping Quality Report ===')
    emit(f'Captures ({len(flow_files)}): ' + ', '.join(os.path.basename(f) for f in flow_files))
    emit(f'Packets sampled per capture: {n_pkt:,} (head sample — representative diagnostic)\n')

    emit('Building flow-side canonical-key index (full flow files)...')
    flow_index = build_flow_key_index(flow_files)
    emit(f'  logical flows indexed -> unique canonical keys: {int(flow_index["n_candidates"].sum()):,} '
         f'-> {len(flow_index):,}')
    emit(f'  keys with >1 candidate: {(flow_index["n_candidates"] > 1).sum():,} '
         f'({100*(flow_index["n_candidates"]>1).mean():.1f}%)\n')

    # map packets per capture
    parts = []
    for s in SUBSET_STEMS:
        pf = os.path.join(PACKET_DIR, s + '.csv')
        if not os.path.exists(pf):
            continue
        pk = pd.read_csv(pf, usecols=PACKET_COLS, nrows=n_pkt, low_memory=False)
        pk[LABEL_COL] = label_from_filename(pf)
        parts.append(classify_packets(pk, flow_index))
    pkts = pd.concat(parts, ignore_index=True)
    total = len(pkts)

    # 1. overall status rates
    emit(f'Total packets considered: {total:,}\n')
    emit('Mapping status breakdown:')
    vc = pkts['status'].value_counts()
    for st in STATUS_ORDER:
        c = int(vc.get(st, 0))
        emit(f'  {st:<28} {c:>9,}  ({100*c/total:5.2f}%)')

    # 2. rates by benign vs attack
    emit('\nMatch status by class group:')
    grpcol = np.where(pkts[LABEL_COL] == BENIGN_LABEL, 'benign', 'attack')
    tab = pd.crosstab(grpcol, pkts['status'], normalize='index').reindex(columns=STATUS_ORDER, fill_value=0)
    for g in tab.index:
        emit(f'  {g:<8} ' + '  '.join(f'{SHORT[st]}:{tab.loc[g, st]*100:4.1f}%' for st in STATUS_ORDER))

    # 3. rates by attack type
    emit('\nExact unique-match rate by label (attack type):')
    for lab in sorted(pkts[LABEL_COL].unique()):
        sub = pkts[pkts[LABEL_COL] == lab]
        u = (sub['status'] == UNIQUE_MATCH).mean()
        nm = (sub['status'] == NO_MATCH).mean()
        iv = (sub['status'] == INVALID_KEY).mean()
        emit(f'  {lab:<18} n={len(sub):>7,}  unique={u*100:5.1f}%  no_match={nm*100:5.1f}%  invalid={iv*100:5.1f}%')

    # 4. unique packets per logical flow
    um = pkts[pkts['status'] == UNIQUE_MATCH]
    if len(um):
        per = um.groupby('matched_lfid').size()
        emit('\nUnique packets mapping to each logical flow (unique matches):')
        emit(f'  logical flows hit: {len(per):,}   packets: {len(um):,}')
        emit(f'  per-flow packet count  min={per.min()}  median={int(per.median())}  '
             f'mean={per.mean():.1f}  p95={int(per.quantile(.95))}  max={per.max()}')

    # 5. candidate-count distribution per packet
    emit('\nCandidate-flow count per packet (0 = no_match / invalid):')
    cc = pkts['n_candidates'].fillna(0).astype(int).clip(upper=5)
    for k, c in cc.value_counts().sort_index().items():
        lbl = f'{k}' if k < 5 else '5+'
        emit(f'  {lbl:>3} candidates: {int(c):>9,}  ({100*c/total:5.2f}%)')

    # 6. examples of each ambiguity category
    emit('\nExamples per category (canonical_key | #cand | #labels | #captures):')
    for st in [UNIQUE_MATCH, MULTI_SAME_LABEL, MULTI_SAME_CAPTURE, MULTI_CONFLICTING, NO_MATCH, INVALID_KEY]:
        ex = pkts[pkts['status'] == st].head(2)
        if ex.empty:
            emit(f'  {st}: (none in sample)')
            continue
        emit(f'  {st}:')
        joined = ex.join(flow_index[['n_labels', 'n_captures']], on='canonical_key')
        for _, r in joined.iterrows():
            key = str(r['canonical_key'])[:52]
            emit(f'    {key:<54} cand={r.get("n_candidates","-")} '
                 f'labels={r.get("n_labels","-")} captures={r.get("n_captures","-")}')

    # 7. verifications
    emit('\nVerifications:')
    sym = verify_direction_symmetry(pkts.sample(min(5000, total), random_state=0))
    emit(f'  [{"PASS" if sym else "FAIL"}] reversing src/dst yields identical canonical key')
    emit(f'  [{"PASS" if verify_protocol_in_key() else "FAIL"}] protocol is part of the key '
         '(same endpoints, TCP vs UDP -> different key)')
    # resolution independence: shuffle labels in a copy of the index, re-classify,
    # confirm unique matches + matched flow are unchanged (labels never resolve).
    shuf = flow_index.copy()
    rng = np.random.default_rng(0)
    shuf['rep_label'] = rng.permutation(shuf['rep_label'].values)
    shuf['n_labels'] = rng.permutation(shuf['n_labels'].values)
    re = classify_packets(pkts[PACKET_COLS + [LABEL_COL]].copy(), shuf)
    unique_same = (re['status'] == UNIQUE_MATCH).equals(pkts['status'] == UNIQUE_MATCH)
    lfid_same = pd.Series(re['matched_lfid']).fillna('_').equals(pd.Series(pkts['matched_lfid']).fillna('_'))
    emit(f'  [{"PASS" if unique_same and lfid_same else "FAIL"}] label/model never resolves '
         'ambiguity (unique matches + chosen flow invariant to shuffled labels)')

    report_path = os.path.join(PHASE3_DIR, 'mapping_quality_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(_out_lines) + '\n')
    emit(f'\nReport written to {report_path}')
    emit('STOP — review the mapping report before proceeding to the sampler.')


if __name__ == '__main__':
    main()
