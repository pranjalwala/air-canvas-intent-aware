import torch

from models.air_canvas_model import AirCanvasModel
from models.compute_losses import compute_losses


def main():
    # -------------------------
    # Hyperparameters (fixed)
    # -------------------------
    B, T, N, d = 4, 40, 21, 3
    vocab_size = 30
    embed_dim = 256
    lr = 1e-3

    # -------------------------
    # Model
    # -------------------------
    model = AirCanvasModel(
        num_joints=N,
        joint_dim=d,
        hidden_dim=128,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Dummy batch (same as tests)
    # -------------------------
    motion = torch.randn(B, T, N, d)

    batch = {
        "motion": motion,
        "intent": torch.tensor([1, 1, 0, 0]),  # TEXT, TEXT, DRAW, DRAW
        "text_targets": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "text_target_lengths": torch.tensor([1, 1, 1, 1]),
        "text_input_lengths": torch.full((B,), T, dtype=torch.long),
        "semantic_text_emb": torch.randn(B, embed_dim),
    }
    batch["semantic_text_emb"] = torch.nn.functional.normalize(
        batch["semantic_text_emb"], dim=-1
    )

    # -------------------------
    # Forward
    # -------------------------
    outputs = model(motion)

    losses = compute_losses(outputs, batch)
    total_loss = losses["total"]

    print("Total loss before backward:", total_loss.item())

    # -------------------------
    # Backward
    # -------------------------
    optimizer.zero_grad()
    total_loss.backward()

    # -------------------------
    # Gradient sanity checks
    # -------------------------
    grad_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()

    print("Total grad norm:", grad_norm)
    assert grad_norm > 0, "No gradients flowing!"

    # -------------------------
    # Optimizer step
    # -------------------------
    optimizer.step()

    print(" One training step completed successfully!")


if __name__ == "__main__":
    main()
