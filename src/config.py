from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Config:
    # 数据路径
    RAW_LOG_DIR = PROJECT_ROOT / "data" / "raw_logs"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    LABEL_CSV = PROJECT_ROOT / "data" / "labels.csv"

    # 词表 & 序列长度
    MIN_TOKEN_FREQ = 2
    MAX_SEQ_LEN = 512

    # 训练参数
    BATCH_SIZE = 32
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    NUM_LAYERS = 1
    DROPOUT = 0.1
    NUM_EPOCHS = 10
    LR = 1e-3

    # Deep SVDD 假定异常比例
    SVDD_CONTAMINATION = 0.25  # 你的数据里大概 50/200 是 flaky，可稍微调高点

    # 设备
    DEVICE = "cpu"  # 如果你之后想用 GPU，改成 "cuda"
