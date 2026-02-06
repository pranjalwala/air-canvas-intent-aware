import torch

from models.motion_encoder import MotionEncoder
from models.text_head import TextHead
from models.losses import ctc_criterion


def main():
    # Dummy dimensions
    B, T, N, d = 2, 80, 21, 3
    vocab_size = 30  # includes blank at index 0

    # Dummy input
    x = torch.randn(B, T, N, d)

    # Dummy target sequences (characters)
    # Values must be in [1, vocab_size-1]
    targets = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)

    # Target lengths (sum must equal len(targets))
    target_lengths = torch.tensor([3, 3], dtype=torch.long)

    # Input lengths (CTC time steps per batch)
    input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long)

    # Models
    encoder = MotionEncoder(num_joints=N, joint_dim=d, hidden_dim=128)
    text_head = TextHead(hidden_dim=128, vocab_size=vocab_size)

    # Forward pass
    h = encoder(x)                 # (B, T, D)
    log_probs = text_head(h)       # (T, B, V)

    # CTC loss
    loss = ctc_criterion(
        log_probs,
        targets,
        input_lengths,
        target_lengths
    )

    print("Log_probs shape :", log_probs.shape)
    print("CTC loss        :", loss.item())

    # Assertions
    assert log_probs.shape == (T, B, vocab_size)
    assert loss.item() >= 0

    print(" TextHead + CTC loss forward pass works!")


if __name__ == "__main__":
    main()
