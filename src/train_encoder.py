import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.config import Config
from src.preprocessing import build_vocabulary
from src.data_loader import create_dataloader, load_labels
from src.models.encoder import LogEncoder


def train_encoder():
    device = torch.device(Config.DEVICE)
    log_dir = Config.RAW_LOG_DIR
    log_files = list(log_dir.glob("*.log"))

    print(f"Found {len(log_files)} logs")

    # --- load vocab ---
    vocab_path = Config.PROCESSED_DIR / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    # --- load labels (optional) ---
    labels = load_labels(Config.LABEL_CSV)

    # --- dataloader ---
    loader = create_dataloader(log_dir, vocab, labels=labels, batch_size=Config.BATCH_SIZE, shuffle=True)

    # --- init model ---
    model = LogEncoder(
        vocab_size=len(vocab),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
    ).to(device)

    # simple self-supervised loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    print("Training encoder...")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}"):
            input_ids, y, fnames = batch
            input_ids = input_ids.to(device)

            emb = model(input_ids)                 # shape (batch, 256)
            target = torch.zeros_like(emb)         # simple SVDD-like compactness

            loss = criterion(emb, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

    # --- save encoder ---
    out_path = Config.PROCESSED_DIR / "encoder.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved encoder to {out_path}")

    # --- extract embeddings ---
    print("Extracting embeddings...")
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in create_dataloader(log_dir, vocab, labels=labels, batch_size=Config.BATCH_SIZE, shuffle=False):
            input_ids, y, fnames = batch
            input_ids = input_ids.to(device)

            emb = model(input_ids)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(y.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    np.save(Config.PROCESSED_DIR / "embeddings.npy", all_embeddings)
    np.save(Config.PROCESSED_DIR / "labels.npy", all_labels)

    print("Saved embeddings.npy and labels.npy")
    print("Phase 4 encoder training complete.")


if __name__ == "__main__":
    train_encoder()
