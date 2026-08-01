"""Central configuration for Phase 3.

Imported first by every ``phase_3`` module. Two jobs:

1. Hold every tunable / frozen constant (paths, labels, gap cutoff, sample sizes,
   split fractions, seed) in one place so scripts do not hard-code magic numbers.
2. Put the project root on ``sys.path`` so ``from helpers import ...`` works
   whether you launch from the repo root or from inside ``phase_3/``.

If you change a constant here, re-run the downstream scripts that depend on it
(sampling, cascade alignment, heads). See ``phase_3/README.md`` for the full
pipeline story.
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
# Prefer the nested CIC layout (flow_and_packet/{flow,packet}_based/).
# Fall back to root-level folders so a Phase-2-style mirror still works.
# ---------------------------------------------------------------------------
_FLOW_NESTED = os.path.join(PROJECT_ROOT, 'flow_and_packet', 'flow_based')
_FLOW_ROOT = os.path.join(PROJECT_ROOT, 'flow_based')
FLOW_DIR = _FLOW_NESTED if os.path.isdir(_FLOW_NESTED) else _FLOW_ROOT

_PACKET_NESTED = os.path.join(PROJECT_ROOT, 'flow_and_packet', 'packet_based')
_PACKET_ROOT = os.path.join(PROJECT_ROOT, 'packet_based')
PACKET_DIR = _PACKET_NESTED if os.path.isdir(_PACKET_NESTED) else _PACKET_ROOT

# Phase 3 outputs live inside phase_3/ (reports committed; data/ is gitignored)
RESULTS_FILE = os.path.join(PHASE3_DIR, 'phase_3_results.txt')
SAVED_FIGS_DIR = os.path.join(PHASE3_DIR, 'saved_figs')

RANDOM_SEED = 23

# ---------------------------------------------------------------------------
# Labelling
# Filename-derived PROXY labels. The flow CSV `Label` column is always
# 'NeedManualLabel' in this dataset and must never be used as a target or
# feature (kept only for audit under NATIVE_LABEL_COL).
#
# Order in ATTACK_SUBSTRING_MAP matters: 'DDoS-HTTP' must be checked before
# 'DoS-HTTP' because the latter is a substring of the former.
# Label strings match the Phase 2 convention in helpers.py.
# ---------------------------------------------------------------------------
BENIGN_LABEL = 'benign'
ATTACK_SUBSTRING_MAP = [
    ('DDoS-HTTP', 'DDOS-HTTP_flood'),
    ('DoS-HTTP', 'DoS-HTTP'),
    ('Spoofing', 'DNS_spoofing'),
    ('XSS', 'XSS'),
    ('BruteForce', 'brute_force'),
]
LABEL_COL = 'label'              # proxy label column we create
NATIVE_LABEL_COL = 'Label'       # original CSV column — audit only, never a feature

# ---------------------------------------------------------------------------
# LogicalFlowID construction
# CICFlowMeter splits long conversations into ~2-minute segments that share a
# directional 5-field Flow ID. The same Flow ID is ALSO reused by unrelated
# conversations hours later. We therefore cluster consecutive segments of the
# same (source_file, Flow ID) while gaps stay within the contiguous mode, and
# start a new logical flow when the gap crosses the density valley (~300 s)
# between the ~120–150 s contiguous mode and the >600 s reuse tail.
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

# Weight columns for packet-count-weighted means during aggregation
W_FWD = 'Total Fwd Packet'
W_BWD = 'Total Bwd packets'

# ---------------------------------------------------------------------------
# Sampling (over unified logical flows) and splits
# Sample sizes reuse Task 1.1 / Phase 2 semantics so phase comparisons are fair:
# 200k benign + 4k–6.2k attack, balanced across the five attack types.
# ---------------------------------------------------------------------------
ATTACK_LABELS = ['DDOS-HTTP_flood', 'DoS-HTTP', 'DNS_spoofing', 'XSS', 'brute_force']
N_ATTACK_TYPES = 5
BENIGN_SAMPLE_N = 200_000
ATTACK_SAMPLE_MIN = 4_000
ATTACK_SAMPLE_MAX = 6_200

# Grouped stratified split (group unit = LogicalFlowID, one row per flow).
# TEST_SIZE / VAL_SIZE are fractions of the whole sample; train gets the rest.
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Blocked temporal robustness folds (expanding window within each class)
N_TEMPORAL_FOLDS = 4

# Cached sampled Phase 3 dataset (built by flow_sampling / check_sampling)
DATA_DIR = os.path.join(PHASE3_DIR, 'data')
UNIFIED_SAMPLE_PATH = os.path.join(DATA_DIR, 'sampled_unified_flows.parquet')
