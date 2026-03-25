import torch
from torch.utils.data import DataLoader

from models.air_canvas_model import AirCanvasModel
from models.compute_losses import compute_losses
from data.quickdraw_ndjson_dataset import QuickDrawNDJSONDataset


def collate_fn(batch):
    motion = torch.stack([b["motion"] for b in batch])
    intent = torch.tensor([b["intent"] for b in batch])

    text_targets = torch.cat([b["text_targets"] for b in batch])
    text_target_lengths = torch.tensor(
        [b["text_target_lengths"] for b in batch]
    )
    text_input_lengths = torch.tensor(
        [b["text_input_lengths"] for b in batch]
    )

    semantic_text_emb = torch.stack(
        [b["semantic_text_emb"] for b in batch]
    )

    return {
        "motion": motion,
        "intent": intent,
        "text_targets": text_targets,
        "text_target_lengths": text_target_lengths,
        "text_input_lengths": text_input_lengths,
        "semantic_text_emb": semantic_text_emb,
    }


def main():
    BATCH_SIZE = 4
    EPOCHS = 2
    LR = 1e-3

    dataset = QuickDrawNDJSONDataset(
        file_path=r"C:\Users\croma\air-canvas-intent-aware\data\quickdraw\hexagon.ndjson"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = AirCanvasModel()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}")

        for step, batch in enumerate(loader):

            motion = batch["motion"]

            outputs = model(motion)

            losses = compute_losses(outputs, batch)
            total_loss = losses["total"]

            optimizer.zero_grad()
            total_loss.backward()

            # gradient check
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item()

            assert grad_norm > 0, "No gradients!"

            optimizer.step()

            print(
                f"Step {step} | "
                f"Total: {total_loss.item():.2f} | "
                f"Text: {losses['text'].item():.2f} | "
                f"Sem: {losses['semantic'].item():.2f} | "
                f"Smooth: {losses['smooth'].item():.2f}"
            )

    print("\n Training loop ran successfully!")


if __name__ == "__main__":
    main()