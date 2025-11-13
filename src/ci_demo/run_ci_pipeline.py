# src/ci_demo/run_ci_pipeline.py

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import joblib

from src.config import Config
from src.preprocessing import tokenize_log_file, encode_tokens
from src.models.encoder import LogEncoder
from src.models.deep_svdd import SimpleSVDD


def load_vocab():
    vocab_path = Config.PROCESSED_DIR / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    return vocab


def load_encoder(vocab_size: int, device: torch.device) -> LogEncoder:
    encoder = LogEncoder(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ).to(device)

    state_dict = torch.load(Config.PROCESSED_DIR / "encoder.pt", map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def load_svdd() -> SimpleSVDD:
    svdd: SimpleSVDD = joblib.load(Config.PROCESSED_DIR / "deep_svdd.pkl")
    return svdd


def score_single_log(log_path: Path):
    device = torch.device(Config.DEVICE)

    # 1. 加载 vocab / encoder / svdd
    vocab = load_vocab()
    encoder = load_encoder(len(vocab), device)
    svdd = load_svdd()

    # 2. 读取并 token 化日志
    tokens = tokenize_log_file(log_path)
    input_ids = encode_tokens(tokens, vocab, max_len=Config.MAX_SEQ_LEN)
    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

    # 3. 生成 embedding
    with torch.no_grad():
        emb = encoder(x)               # (1, D)
        emb_np = emb.cpu().numpy()     # (1, D)

    # 4. 计算 anomaly score + 预测
    score = float(svdd.decision_scores(emb_np)[0])
    label = int(svdd.predict(emb_np)[0])  # 1 = anomaly (flaky-like), 0 = normal

    result = {
        "log_file": str(log_path),
        "anomaly_score": score,
        "flagged_as_flaky_like": bool(label),
    }

    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Score a single test log with SimpleSVDD.")
    parser.add_argument("--log", type=str, required=True, help="Path to a .log file")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    score_single_log(log_path)


if __name__ == "__main__":
    main()
