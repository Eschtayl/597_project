import os
import sys

PHASE3_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PHASE3_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Only used by the isolated import in main.py; not added to sys.path here.
PHASE2_DIR = os.path.join(PROJECT_ROOT, 'phase_2')

PACKET_PATH = os.path.join(PROJECT_ROOT, 'packet_based')
FLOW_PATH = os.path.join(PROJECT_ROOT, 'flow_based')

RANDOM_SEED = 23

RESULTS_FILE = os.path.join(PHASE3_DIR, 'phase_3_results.txt')
SAVED_FIGS_DIR = os.path.join(PHASE3_DIR, 'saved_figs')

LABEL_COL = 'label'
BENIGN_LABEL = 'benign'
ORIG_LABEL_COL = 'Label'
FLOW_ID_COL = 'Flow ID'

# Held out of training; used for matching only
FLOW_IDENTIFIER_COLS = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol', 'Timestamp']

PACKET_SRC_IP = 'src_ip'
PACKET_DST_IP = 'dst_ip'
PACKET_SRC_PORT = 'src_port'
PACKET_DST_PORT = 'dst_port'

TEST_SIZE = 0.20
VAL_SIZE = 0.15
