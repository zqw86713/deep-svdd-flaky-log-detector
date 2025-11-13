# src/data_loader.py

from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from .preprocessing import tokenize_log_file, encode_tokens
from .config import Config

class LogDataset(Dataset):
    """
    每个样本 = 一个 log 文件。
    x = token id 序列
    y = label (0=normal, 1=flaky)，如果没有就 -1
    """

    def __init__(
        self,
        log_paths: List[Path],
        vocab: Dict[str, int],
        labels: Optional[Dict[str, int]] = None,
        max_len: int = Config.MAX_SEQ_LEN,
    ):
        self.log_paths = log_paths
        self.vocab = vocab
        self.labels = labels or {}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.log_paths)

    def __getitem__(self, idx: int):
        path = self.log_paths[idx]
        tokens = tokenize_log_file(path)
        input_ids = encode_tokens(tokens, self.vocab, self.max_len)
        x = torch.tensor(input_ids, dtype=torch.long)

        fname = path.name
        y = self.labels.get(fname, -1)
        y = torch.tensor(y, dtype=torch.long)
        return x, y, fname

def load_labels(label_csv: Path) -> Dict[str, int]:
    """从 labels.csv 读取 filename → label 映射。"""
    df = pd.read_csv(label_csv)
    mapping = dict(zip(df["filename"], df["label"]))
    return mapping

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
