import torch

from models.motion_encoder import MotionEncoder
from models.drawing_head import DrawingHead
from models.losses import contrastive_loss


def main():
    # Dummy dimensions
    B, T, N, d = 4, 60, 21, 3
    hidden_dim = 128
    embed_dim = 256

    # Dummy gesture input
    x = torch.randn(B, T, N, d)

    # Dummy text embeddings (stand-in for language model)
    text_emb = torch.randn(B, embed_dim)
    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

    # Models
    encoder = MotionEncoder(num_joints=N, joint_dim=d, hidden_dim=hidden_dim)
    drawing_head = DrawingHead(hidden_dim=hidden_dim, embed_dim=embed_dim)

    # Forward
    h = encoder(x)              # (B, T, D)
    gesture_emb = drawing_head(h)  # (B, E)

    # Loss
    loss = contrastive_loss(gesture_emb, text_emb)

    print("Gesture embedding shape :", gesture_emb.shape)
    print("Contrastive loss        :", loss.item())

    # Assertions
    assert gesture_emb.shape == (B, embed_dim)
    assert torch.allclose(
        gesture_emb.norm(dim=-1),
        torch.ones(B),
        atol=1e-4
    )
    assert loss.item() > 0

    print(" DrawingHead + contrastive loss works!")


if __name__ == "__main__":
    main()
