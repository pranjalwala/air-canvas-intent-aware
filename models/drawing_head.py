import torch
import torch.nn as nn
import torch.nn.functional as F


class DrawingHead(nn.Module):
    """
    Maps a gesture sequence embedding into a semantic embedding space.

    Input:
        h : Tensor of shape (B, T, D)
    Output:
        z : Tensor of shape (B, E)
    """

    def __init__(self, hidden_dim=128, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, h):
        """
        Args:
            h (Tensor): shape (B, T, D)

        Returns:
            z (Tensor): shape (B, E), L2-normalized
        """
        # Temporal average pooling
        h_mean = h.mean(dim=1)          # (B, D)

        z = self.proj(h_mean)            # (B, E)
        z = F.normalize(z, dim=-1)       # unit-norm embedding

        return z
