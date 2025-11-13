# src/preprocessing.py

import re
from collections import Counter
from pathlib import Path
from typing import List, Dict

from .config import Config

TOKEN_PATTERN = re.compile(r"\w+|\S")

def tokenize_line(line: str) -> List[str]:
    """Tokenize a single log line using a simple regex-based tokenizer."""
    return TOKEN_PATTERN.findall(line.lower())

def build_vocabulary(log_files: List[Path], min_freq: int) -> Dict[str, int]:
    """Build a token vocabulary from multiple log files."""
    counter = Counter()
    for path in log_files:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tokens = tokenize_line(line)
                counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab

def encode_tokens(tokens: List[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    """Convert tokens to indices and pad/truncate to fixed length."""
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids
