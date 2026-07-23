import os
import sys

# phase_2/ is this file's directory; project root holds helpers.py and data folders
PHASE2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PHASE2_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Data folders live at project root (next to helpers.py)
PACKET_PATH = os.path.join(PROJECT_ROOT, 'packet_based')
FLOW_PATH = os.path.join(PROJECT_ROOT, 'flow_based')

# Random seed
RANDOM_SEED = 23

# Phase 2 output paths (inside phase_2/)
RESULTS_FILE = os.path.join(PHASE2_DIR, 'phase_2_results.txt')
SAVED_FIGS_DIR = os.path.join(PHASE2_DIR, 'saved_figs')

# Label column name after preprocessing
LABEL_COL = 'label'
BENIGN_LABEL = 'benign'

# Columns that identify a machine/user — do not train the anomaly model on these
IDENTIFIER_KEYWORDS = [
    'ip', 'port', 'mac', 'timestamp', 'flow_id', 'protocol',
    'server', 'host', 'user_agent', 'oui', 'uri', 'content_type'
]

# Train / test split for Phase 2 evaluation
TEST_SIZE = 0.25

# Isolation Forest hyperparameters
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 256
IF_CONTAMINATION = 'auto'
IF_N_JOBS = -1

# Autoencoder hyperparameters
AE_HIDDEN_DIMS = (64, 32, 16)
AE_EPOCHS = 20
AE_BATCH_SIZE = 256
AE_LEARNING_RATE = 1e-3
AE_WEIGHT_DECAY = 1e-5
AE_VAL_FRACTION = 0.1

# Candidate architectures for systematic AE tuning (selected by benign val MSE)
AE_HIDDEN_DIMS_GRID = [
    (128, 64, 32),
    (64, 32, 16),
    (32, 16, 8),
    (64, 16, 4),
]

# Ensemble (score fusion of IF + AE)
# method: 'mean' | 'max' | 'weighted'
ENSEMBLE_METHOD = 'mean'
ENSEMBLE_WEIGHT_AE = 0.6

# K-Means anomaly detector (packet level)
# k is large on purpose: at ~2% attack prevalence, k=2 just splits the benign
# cloud; many small clusters let attacks form high-purity clusters instead.
KM_N_CLUSTERS = 300
KM_N_INIT = 10
KM_PCA_COMPONENTS = 30     # PCA before clustering (0 disables); tamed distance concentration
KM_MINIBATCH = True        # MiniBatchKMeans keeps large-k fitting fast
