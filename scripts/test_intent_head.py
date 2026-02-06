import torch

from models.motion_encoder import MotionEncoder
from models.intent_head import IntentHead
from models.losses import intent_criterion


def main():
    # Dummy batch
    B, T, N, d = 4, 60, 21, 3

    x = torch.randn(B, T, N, d)

    # Ground-truth intent labels
    # 1 = TEXT, 0 = DRAW
    y_intent = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    # Models
    encoder = MotionEncoder(num_joints=N, joint_dim=d, hidden_dim=128)
    intent_head = IntentHead(hidden_dim=128)

    # Forward pass
    h = encoder(x)              # (B, T, D)
    logits = intent_head(h)     # (B, 1)

    # Loss
    loss = intent_criterion(logits, y_intent)

    print("Latent shape :", h.shape)
    print("Logits shape :", logits.shape)
    print("Intent loss  :", loss.item())

    # Assertions
    assert logits.shape == (B, 1)
    assert loss.item() > 0

    print(" IntentHead + loss forward pass works!")


if __name__ == "__main__":
    main()
