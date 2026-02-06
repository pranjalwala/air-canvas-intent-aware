import torch
import torch.nn as nn
import torch.nn.functional as F


class TextHead(nn.Module):
    """
    Frame-wise character prediction head for air-writing.

    Input:
        h : Tensor of shape (B, T, D)
    Output:
        log_probs : Tensor of shape (T, B, V)
    """

    def __init__(self, hidden_dim=128, vocab_size=30):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, h):
        """
        Args:
            h (Tensor): shape (B, T, D)

        Returns:
            log_probs (Tensor): shape (T, B, V)
        """
        logits = self.fc(h)                # (B, T, V)
        logits = logits.permute(1, 0, 2)   # (T, B, V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
