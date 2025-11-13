# src/models/encoder.py

import torch
import torch.nn as nn

from ..config import Config

class LogEncoder(nn.Module):
    """Encode a log (sequence of tokens) into a fixed-size vector."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = Config.EMBEDDING_DIM,
        hidden_dim: int = Config.HIDDEN_DIM,
        num_layers: int = Config.NUM_LAYERS,
        dropout: float = Config.DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: shape (batch_size, seq_len)
        Returns:
            embeddings: shape (batch_size, hidden_dim * 2)
        """
        x = self.embedding(input_ids)
        output, (h_n, c_n) = self.lstm(x)
        # Use mean pooling over time dimension
        pooled = output.mean(dim=1)
        pooled = self.dropout(pooled)
        return pooled
