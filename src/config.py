# src/config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Config:
    # Data paths
    RAW_LOG_DIR = PROJECT_ROOT / "data" / "raw_logs"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    LABEL_CSV = PROJECT_ROOT / "data" / "labels.csv"

    # Vocabulary / tokenization
    MIN_TOKEN_FREQ = 2
    MAX_SEQ_LEN = 512

    # Training parameters
    BATCH_SIZE = 64
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    DROPOUT = 0.1
    NUM_EPOCHS = 20
    LR = 1e-3

    # Deep SVDD
    SVDD_CONTAMINATION = 0.1  # assumed fraction of anomalies

    # Device
    DEVICE = "cuda"  # or "cpu"
