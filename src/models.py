import torch
import torch.nn as nn

from .data import VOCAB_SIZE


class ProteinCNN(nn.Module):
    """1D CNN: embedding -> parallel convolutions -> masked global max pooling -> linear."""

    def __init__(self, n_classes, emb_dim=64, n_filters=128,
                 kernel_sizes=(3, 9, 15), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x):                              # x: (batch, length)
        mask = (x != 0).unsqueeze(1)                   # (batch, 1, length), True on real residues
        h = self.embedding(x).transpose(1, 2)          # (batch, emb_dim, length)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(h))                    # (batch, n_filters, length)
            c = c.masked_fill(~mask, -1e9)             # ignore padding when taking the max
            pooled.append(c.max(dim=2).values)         # (batch, n_filters)
        z = self.dropout(torch.cat(pooled, dim=1))
        return self.classifier(z)                      # (batch, n_classes) logits