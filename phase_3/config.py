"""Central configuration for Phase 3 (paths, labels, frozen constants, sampling
and split sizes). Imported first by every phase_3 module; also puts the project
root on sys.path so `from helpers import ...` works from any launch directory.
"""
import os
import sys

# phase_3/ is this file's directory; project root holds helpers.py and data folders
PHASE3_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PHASE3_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Data paths
# The real flow data lives under flow_and_packet/ .
# Fall back to a root-level flow_based/ if someone mirrors the Phase 2 layout.
# ---------------------------------------------------------------------------
_FLOW_NESTED = os.path.join(PROJECT_ROOT, 'flow_and_packet', 'flow_based')
_FLOW_ROOT = os.path.join(PROJECT_ROOT, 'flow_based')
FLOW_DIR = _FLOW_NESTED if os.path.isdir(_FLOW_NESTED) else _FLOW_ROOT

_PACKET_NESTED = os.path.join(PROJECT_ROOT, 'flow_and_packet', 'packet_based')
_PACKET_ROOT = os.path.join(PROJECT_ROOT, 'packet_based')
PACKET_DIR = _PACKET_NESTED if os.path.isdir(_PACKET_NESTED) else _PACKET_ROOT

# Phase 3 outputs live inside phase_3/
RESULTS_FILE = os.path.join(PHASE3_DIR, 'phase_3_results.txt')
SAVED_FIGS_DIR = os.path.join(PHASE3_DIR, 'saved_figs')

RANDOM_SEED = 23

# ---------------------------------------------------------------------------
# Labelling (filename-derived proxy labels — the flow Label column is
# 'NeedManualLabel' and is never used). Label strings match the Phase 2
# convention in helpers.py. Order matters: 'DDoS-HTTP' must be checked before
# 'DoS-HTTP' because 'DoS-HTTP' is a substring of 'DDoS-HTTP'.
# ---------------------------------------------------------------------------
BENIGN_LABEL = 'benign'
ATTACK_SUBSTRING_MAP = [
    ('DDoS-HTTP', 'DDOS-HTTP_flood'),
    ('DoS-HTTP', 'DoS-HTTP'),
    ('Spoofing', 'DNS_spoofing'),
    ('XSS', 'XSS'),
    ('BruteForce', 'brute_force'),
]
LABEL_COL = 'label'
NATIVE_LABEL_COL = 'Label'  # kept for audit only, dropped from features

# ---------------------------------------------------------------------------
# LogicalFlowID construction
# Consecutive segments of the same directional Flow ID are one logical flow
# while their timestamp gap stays within the 2-minute cadence; a gap beyond the
# cutoff (density valley between the ~120-150s contiguous mode and the >600s
# reuse tail) starts a new logical flow.
# ---------------------------------------------------------------------------
FLOW_ID_COL = 'Flow ID'
TIMESTAMP_COL = 'Timestamp'
TIMESTAMP_FORMAT = '%d/%m/%Y %I:%M:%S %p'   # DD/MM/YYYY hh:MM:SS AM/PM, second granularity
GAP_CUTOFF_SECONDS = 300

# Identifier / metadata columns — carried through aggregation but excluded from
# model features (identity, not behaviour).
FLOW_META_COLS = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp']

# Rate recomputation: CICFlowMeter durations are in microseconds, so
# rate = count / sum(duration_us) * 1e6 (verified against raw rows).
DURATION_COL = 'Flow Duration'
MICROSECONDS_PER_SECOND = 1_000_000

# Weight columns for weighted means
W_FWD = 'Total Fwd Packet'
W_BWD = 'Total Bwd packets'

# ---------------------------------------------------------------------------
# Sampling (over unified logical flows) and splits
# ---------------------------------------------------------------------------
ATTACK_LABELS = ['DDOS-HTTP_flood', 'DoS-HTTP', 'DNS_spoofing', 'XSS', 'brute_force']
N_ATTACK_TYPES = 5
BENIGN_SAMPLE_N = 200_000
ATTACK_SAMPLE_MIN = 4_000
ATTACK_SAMPLE_MAX = 6_200

# Grouped stratified split (group unit = LogicalFlowID, one row per flow)
TEST_SIZE = 0.15
VAL_SIZE = 0.15   # fraction of the whole; train gets the remainder

# Blocked temporal robustness folds (expanding window within each class)
N_TEMPORAL_FOLDS = 4

# Cached sampled Phase 3 dataset
DATA_DIR = os.path.join(PHASE3_DIR, 'data')
UNIFIED_SAMPLE_PATH = os.path.join(DATA_DIR, 'sampled_unified_flows.parquet')
