import torch
import torch.nn as nn


class IntentHead(nn.Module):
    """
    Binary intent classifier: TEXT (1) vs DRAWING (0)

    Input:
        h : Tensor of shape (B, T, D)
    Output:
        logits : Tensor of shape (B, 1)
    """

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        """
        Args:
            h (Tensor): shape (B, T, D)

        Returns:
            logits (Tensor): shape (B, 1)
        """
        # Temporal mean pooling
        h_mean = h.mean(dim=1)   # (B, D)

        logits = self.fc(h_mean) # (B, 1)
        return logits
