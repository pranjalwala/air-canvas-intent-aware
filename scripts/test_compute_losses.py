import torch

from models.air_canvas_model import AirCanvasModel
from models.compute_losses import compute_losses


def main():
    B, T, N, d = 4, 40, 21, 3
    vocab_size = 30
    embed_dim = 256

    # Dummy model
    model = AirCanvasModel(
        num_joints=N,
        joint_dim=d,
        hidden_dim=128,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )

    # Dummy input
    motion = torch.randn(B, T, N, d)
    outputs = model(motion)

    # Dummy batch
    batch = {
        "motion": motion,
        "intent": torch.tensor([1, 1, 0, 0]),  # first two TEXT, last two DRAW
        "text_targets": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "text_target_lengths": torch.tensor([1, 1, 1, 1]),
        "text_input_lengths": torch.full((B,), T, dtype=torch.long),
        "semantic_text_emb": torch.randn(B, embed_dim),
    }
    batch["semantic_text_emb"] = torch.nn.functional.normalize(
        batch["semantic_text_emb"], dim=-1
    )

    losses = compute_losses(outputs, batch)

    for k, v in losses.items():
        print(f"{k:10s} : {v.item():.4f}")

    assert losses["total"].item() > 0
    print(" Intent-gated loss computation works!")


if __name__ == "__main__":
    main()
