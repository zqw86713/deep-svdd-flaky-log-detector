# src/models/encoder.py

import torch
import torch.nn as nn

class LogEncoder(nn.Module):
    """LSTM log encoder that outputs a fixed-size vector for each log."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        output, (h_n, c_n) = self.lstm(x)

        # mean pooling
        pooled = output.mean(dim=1)

        return self.dropout(pooled)
