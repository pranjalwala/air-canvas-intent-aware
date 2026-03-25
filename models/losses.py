import torch
import torch.nn as nn
import torch.nn.functional as F

# Intent loss
intent_criterion = nn.BCEWithLogitsLoss()

# Text recognition loss (CTC)
ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)


def contrastive_loss(gesture_emb, text_emb, temperature=0.07):
    """
    CLIP-style symmetric contrastive loss.

    Args:
        gesture_emb : (B, E)
        text_emb    : (B, E)
    """
    logits = gesture_emb @ text_emb.T / temperature   # (B, B)

    labels = torch.arange(logits.size(0), device=logits.device)

    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_g2t + loss_t2g)
def motion_smoothness_loss(x):
    """
    Second-order temporal smoothness loss.

    Args:
        x (Tensor): (B, T, N, d) raw motion input

    Returns:
        scalar Tensor
    """
    # x_{t+1} - 2 x_t + x_{t-1}
    accel = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]   # (B, T-2, N, d)

    loss = (accel ** 2).mean()
    return loss
