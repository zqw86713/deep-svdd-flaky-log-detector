# src/ci_demo/run_ci_pipeline.py

import json
from pathlib import Path

import numpy as np
import torch
import joblib

from ..config import Config, PROJECT_ROOT
from ..preprocessing import tokenize_line, encode_tokens
from ..models.encoder import LogEncoder

THRESHOLD = 0.5  # Example threshold after normalization or use model's built-in

def load_vocab() -> dict:
    vocab_path = PROJECT_ROOT / "data" / "processed" / "vocab.json"
    return json.loads(vocab_path.read_text(encoding="utf-8"))

def load_models(vocab_size: int):
    device = torch.device(Config.DEVICE)
    encoder = LogEncoder(vocab_size=vocab_size).to(device)
    encoder.load_state_dict(torch.load(PROJECT_ROOT / "data" / "processed" / "encoder.pt", map_location=device))
    encoder.eval()

    svdd = joblib.load(PROJECT_ROOT / "data" / "processed" / "deep_svdd.pkl")
    return encoder, svdd, device

def score_log(log_path: Path):
    vocab = load_vocab()
    encoder, svdd, device = load_models(len(vocab))

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    tokens = tokenize_line(text)
    input_ids = encode_tokens(tokens, vocab, max_len=Config.MAX_SEQ_LEN)

    x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = encoder(x)
        score = svdd.decision_scores(emb.cpu().numpy())[0]

    result = {
        "log_file": log_path.name,
        "anomaly_score": float(score),
        "flagged_as_flaky_like": bool(score > THRESHOLD),
    }

    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    # Example usage: python -m src.ci_demo.run_ci_pipeline --log data/raw_logs/sample.log
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to test log file")
    args = parser.parse_args()

    score_log(Path(args.log))
