# src/data_loader.py

from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import tokenize_line, encode_tokens
from .config import Config

class LogDataset(Dataset):
    """Dataset for log files. Each sample is a single test log file."""

    def __init__(
        self,
        log_paths: List[Path],
        vocab: Dict[str, int],
        labels: Optional[Dict[str, int]] = None,
        max_len: int = Config.MAX_SEQ_LEN,
    ):
        """
        Args:
            log_paths: list of paths to log files.
            vocab: token vocabulary mapping.
            labels: optional mapping from log filename → label (0 normal, 1 anomaly).
        """
        self.log_paths = log_paths
        self.vocab = vocab
        self.labels = labels or {}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.log_paths)

    def _read_log(self, path: Path) -> str:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def __getitem__(self, idx: int):
        path = self.log_paths[idx]
        text = self._read_log(path)
        tokens = tokenize_line(text)
        input_ids = encode_tokens(tokens, self.vocab, self.max_len)

        x = torch.tensor(input_ids, dtype=torch.long)

        # Optional label (for evaluation)
        fname = path.name
        y = self.labels.get(fname, -1)  # -1 means "unlabeled"
        y = torch.tensor(y, dtype=torch.long)

        return x, y, fname

def create_dataloader(
    log_dir: Path,
    vocab: Dict[str, int],
    labels: Optional[Dict[str, int]] = None,
    batch_size: int = Config.BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    paths = sorted(list(log_dir.glob("*.log")))
    dataset = LogDataset(paths, vocab, labels=labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
