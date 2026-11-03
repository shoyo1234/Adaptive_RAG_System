"""Central configuration. Change experiment settings here only."""

SEED = 42

# Dataset
DATASET_NAME = "scifact"
DEV_FRACTION = 0.5          # dev = threshold tuning, test = final report

# Retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_CANDIDATES = 50           # candidates pulled from EACH stream
RRF_K = 60                  # standard RRF constant
FINAL_TOP_K = 10            # chunks passed on / evaluated

# Reranking (used from Day 2)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MAX_LENGTH = 256   # tokens per (query + passage) pair

# Adaptive gate
GATE_K_RERANK = 5     # rerank depth when the gate does not skip

# Paths
INDEX_DIR = "indexes"
RESULTS_DIR = "results"