# src/evaluate.py

import json
from pathlib import Path

import numpy as np
import torch
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import Config, PROJECT_ROOT
from .data_loader import create_dataloader
from .models.encoder import LogEncoder

def load_vocab() -> dict:
    vocab_path = PROJECT_ROOT / "data" / "processed" / "vocab.json"
    return json.loads(vocab_path.read_text(encoding="utf-8"))

def evaluate():
    vocab = load_vocab()
    labels_df = pd.read_csv(Config.LABEL_CSV)  # columns: filename, label(0/1)
    label_map = dict(zip(labels_df["filename"], labels_df["label"]))

    dataloader = create_dataloader(Config.RAW_LOG_DIR, vocab, labels=label_map, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Load encoder and Deep SVDD
    device = torch.device(Config.DEVICE)
    encoder = LogEncoder(vocab_size=len(vocab)).to(device)
    encoder.load_state_dict(torch.load(PROJECT_ROOT / "data" / "processed" / "encoder.pt", map_location=device))
    encoder.eval()

    svdd: "DeepSVDDWrapper" = joblib.load(PROJECT_ROOT / "data" / "processed" / "deep_svdd.pkl")

    y_true = []
    y_score = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels, fnames = batch
            mask = labels >= 0
            if not mask.any():
                continue

            input_ids = input_ids[mask].to(device)
            labels = labels[mask]

            emb = encoder(input_ids)
            scores = svdd.decision_scores(emb.cpu().numpy())

            y_true.extend(labels.numpy().tolist())
            y_score.extend(scores.tolist())

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision (PR AUC): {ap:.4f}")

if __name__ == "__main__":
    evaluate()
