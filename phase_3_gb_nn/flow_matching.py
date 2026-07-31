import numpy as np
import pandas as pd

from config import (
    LABEL_COL,
    FLOW_IDENTIFIER_COLS,
    PACKET_SRC_IP,
    PACKET_DST_IP,
    PACKET_SRC_PORT,
    PACKET_DST_PORT,
)


PROTO_TCP = 6
PROTO_UDP = 17
PROTO_ICMP = 1


def _infer_packet_protocol(df_packet_preprocessed, packet_indices):
    # Packet CSVs expose l4_tcp/l4_udp/icmp_* flags rather than a protocol number.
    sub = df_packet_preprocessed.loc[packet_indices]
    proto = pd.Series(np.nan, index=sub.index, dtype=float)

    if 'l4_tcp' in sub.columns:
        proto = proto.where(~(pd.to_numeric(sub['l4_tcp'], errors='coerce') > 0), PROTO_TCP)
    if 'l4_udp' in sub.columns:
        proto = proto.where(~(pd.to_numeric(sub['l4_udp'], errors='coerce') > 0), PROTO_UDP)
    if 'icmp_type' in sub.columns:
        has_icmp = sub['icmp_type'].notna()
        proto = proto.where(~has_icmp, PROTO_ICMP)

    return proto


def build_packet_flow_keys(df_packet_preprocessed, packet_indices):
    # Both directions are built since either endpoint may be logged as the source.
    needed = [PACKET_SRC_IP, PACKET_DST_IP, PACKET_SRC_PORT, PACKET_DST_PORT]
    missing = [c for c in needed if c not in df_packet_preprocessed.columns]
    if missing:
        raise ValueError(f'Packet dataframe missing identifier columns: {missing}')

    sub = df_packet_preprocessed.loc[packet_indices, needed].copy()
    for col in (PACKET_SRC_PORT, PACKET_DST_PORT):
        sub[col] = pd.to_numeric(sub[col], errors='coerce').fillna(-1).astype(int)
    sub[PACKET_SRC_IP] = sub[PACKET_SRC_IP].astype(str).str.strip()
    sub[PACKET_DST_IP] = sub[PACKET_DST_IP].astype(str).str.strip()

    proto = _infer_packet_protocol(df_packet_preprocessed, packet_indices).values

    forward_keys = list(zip(sub[PACKET_SRC_IP], sub[PACKET_DST_IP],
                            sub[PACKET_SRC_PORT], sub[PACKET_DST_PORT], proto))
    reverse_keys = list(zip(sub[PACKET_DST_IP], sub[PACKET_SRC_IP],
                            sub[PACKET_DST_PORT], sub[PACKET_SRC_PORT], proto))
    return forward_keys, reverse_keys


def _build_flow_lookup(df_flow_agg):
    needed = ['Src IP', 'Dst IP', 'Src Port', 'Dst Port']
    missing = [c for c in needed if c not in df_flow_agg.columns]
    if missing:
        raise ValueError(f'Flow dataframe missing identifier columns: {missing}')

    src_ip = df_flow_agg['Src IP'].astype(str).str.strip().values
    dst_ip = df_flow_agg['Dst IP'].astype(str).str.strip().values
    src_port = pd.to_numeric(df_flow_agg['Src Port'], errors='coerce').fillna(-1).astype(int).values
    dst_port = pd.to_numeric(df_flow_agg['Dst Port'], errors='coerce').fillna(-1).astype(int).values

    proto_full = None
    if 'Protocol' in df_flow_agg.columns:
        proto_full = pd.to_numeric(df_flow_agg['Protocol'], errors='coerce').values

    lookup_with_proto = {}
    lookup_no_proto = {}
    for i in range(len(df_flow_agg)):
        key4 = (src_ip[i], dst_ip[i], int(src_port[i]), int(dst_port[i]))
        lookup_no_proto.setdefault(key4, i)
        if proto_full is not None and not np.isnan(proto_full[i]):
            key5 = key4 + (float(proto_full[i]),)
            lookup_with_proto.setdefault(key5, i)

    return lookup_with_proto, lookup_no_proto


def match_packets_to_flows(df_packet_preprocessed, packet_indices, df_flow_agg, scaler):
    # Protocol-aware match first, then a 4-tuple fallback.
    from flow_prep import preprocess_flow_features

    fwd_keys, rev_keys = build_packet_flow_keys(df_packet_preprocessed, packet_indices)
    lookup_with_proto, lookup_no_proto = _build_flow_lookup(df_flow_agg)

    matched_rows = np.full(len(fwd_keys), -1, dtype=int)
    for i in range(len(fwd_keys)):
        fk, rk = fwd_keys[i], rev_keys[i]
        idx = lookup_with_proto.get(fk, -1)
        if idx == -1:
            idx = lookup_with_proto.get(rk, -1)
        if idx == -1:
            idx = lookup_no_proto.get(fk[:4], -1)
        if idx == -1:
            idx = lookup_no_proto.get(rk[:4], -1)
        matched_rows[i] = idx

    matched_mask = matched_rows >= 0
    match_rate = matched_mask.mean() if len(matched_mask) else 0.0
    print(f'Matched {int(matched_mask.sum()):,}/{len(matched_mask):,} '
          f'packets to flows ({match_rate:.1%})')

    if matched_mask.sum() == 0:
        empty = pd.DataFrame(columns=df_flow_agg.columns)
        return empty.drop(columns=[c for c in FLOW_IDENTIFIER_COLS + [LABEL_COL] if c in empty.columns],
                          errors='ignore'), matched_mask, empty

    matched_flow_rows = df_flow_agg.iloc[matched_rows[matched_mask]].copy().reset_index(drop=True)
    df_pre, _ = preprocess_flow_features(matched_flow_rows, fit_scaler=False, scaler=scaler)

    identifier_cols_present = [c for c in FLOW_IDENTIFIER_COLS if c in df_pre.columns]
    x_matched = df_pre.drop(columns=identifier_cols_present + [LABEL_COL], errors='ignore')

    return x_matched, matched_mask, matched_flow_rows
