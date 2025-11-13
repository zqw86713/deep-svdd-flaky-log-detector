# src/train.py

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib
import json

from .config import Config, PROJECT_ROOT
from .preprocessing import build_vocabulary
from .data_loader import create_dataloader
from .models.encoder import LogEncoder
from .models.deep_svdd import DeepSVDDWrapper

def train_encoder_and_svdd():
    # 1. Collect log files
    log_dir = Config.RAW_LOG_DIR
    log_files = list(log_dir.glob("*.log"))
    assert log_files, f"No log files found in {log_dir}"

    # 2. Build vocabulary
    vocab = build_vocabulary(log_files, Config.MIN_TOKEN_FREQ)
    vocab_path = PROJECT_ROOT / "data" / "processed" / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")

    # 3. Create DataLoader (no labels needed for unsupervised training)
    dataloader = create_dataloader(log_dir, vocab, labels=None, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 4. Initialize encoder
    device = torch.device(Config.DEVICE)
    encoder = LogEncoder(vocab_size=len(vocab)).to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=Config.LR)
    criterion = nn.MSELoss()

    # Simple self-supervised objective: try to reconstruct a projection (e.g., identity)
    # For Deep SVDD you could also skip this and train only Deep SVDD on raw encoder outputs.

    encoder.train()
    for epoch in range(Config.NUM_EPOCHS):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}"):
            input_ids, _, _ = batch
            input_ids = input_ids.to(device)
            embeddings = encoder(input_ids)

            # Dummy target: zero vector (just force compact representation)
            target = torch.zeros_like(embeddings)
            loss = criterion(embeddings, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * input_ids.size(0)

        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} - avg loss: {avg_loss:.4f}")

    # Save encoder
    encoder_path = PROJECT_ROOT / "data" / "processed" / "encoder.pt"
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Saved encoder to {encoder_path}")

    # 5. Extract embeddings for Deep SVDD
    encoder.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extract embeddings"):
            input_ids, _, _ = batch
            input_ids = input_ids.to(device)
            emb = encoder(input_ids)
            all_embeddings.append(emb.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # 6. Fit Deep SVDD
    svdd = DeepSVDDWrapper(
        n_features=all_embeddings.shape[1],
        contamination=Config.SVDD_CONTAMINATION,
    )
    svdd.fit(all_embeddings)

    svdd_path = PROJECT_ROOT / "data" / "processed" / "deep_svdd.pkl"
    joblib.dump(svdd, svdd_path)
    print(f"Saved Deep SVDD model to {svdd_path}")

if __name__ == "__main__":
    train_encoder_and_svdd()
