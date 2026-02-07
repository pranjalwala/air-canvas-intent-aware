import torch

from models.air_canvas_model import AirCanvasModel


def main():
    # Dummy dimensions
    B, T, N, d = 3, 50, 21, 3
    vocab_size = 30
    embed_dim = 256

    # Dummy input
    x = torch.randn(B, T, N, d)

    # Model
    model = AirCanvasModel(
        num_joints=N,
        joint_dim=d,
        hidden_dim=128,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
    )

    # Forward
    out = model(x)

    # Extract
    h = out["h"]
    intent_logits = out["intent_logits"]
    text_log_probs = out["text_log_probs"]
    drawing_emb = out["drawing_emb"]

    # Prints
    print("h shape              :", h.shape)
    print("intent_logits shape  :", intent_logits.shape)
    print("text_log_probs shape :", text_log_probs.shape)
    print("drawing_emb shape    :", drawing_emb.shape)

    # Assertions
    assert h.shape == (B, T, 128)
    assert intent_logits.shape == (B, 1)
    assert text_log_probs.shape == (T, B, vocab_size)
    assert drawing_emb.shape == (B, embed_dim)

    print(" Unified AirCanvasModel forward pass works!")


if __name__ == "__main__":
    main()
