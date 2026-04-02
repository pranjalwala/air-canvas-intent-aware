import torch
from torch.utils.data import DataLoader

from models.air_canvas_model import AirCanvasModel
from models.compute_losses import compute_losses
from data.quickdraw_multiclass_dataset import QuickDrawMultiClassDataset

import open_clip


# 🔥 DEVICE SETUP
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")


# 🔥 Load CLIP
print("⏳ Loading CLIP...")
clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

clip_model = clip_model.to(device)
clip_model.eval()
print("✅ CLIP loaded")


# 🔥 Projection: 512 → 256
projection = torch.nn.Linear(512, 256).to(device)


# 🔥 CLIP cache (VERY IMPORTANT)
label_cache = {}


def text_to_embedding(label):
    if label in label_cache:
        return label_cache[label]

    tokens = tokenizer([label]).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)

    emb = torch.nn.functional.normalize(text_features[0], dim=-1)

    label_cache[label] = emb
    return emb


def collate_fn(batch):
    motion = torch.stack([b["motion"] for b in batch])
    intent = torch.tensor([b["intent"] for b in batch])

    text_targets = torch.cat([b["text_targets"] for b in batch])
    text_target_lengths = torch.tensor([b["text_target_lengths"] for b in batch])
    text_input_lengths = torch.tensor([b["text_input_lengths"] for b in batch])

    labels = [b["label"] for b in batch]

    return {
        "motion": motion,
        "intent": intent,
        "text_targets": text_targets,
        "text_target_lengths": text_target_lengths,
        "text_input_lengths": text_input_lengths,
        "label": labels,
    }


def main():
    print("🔥 Starting training script...")

    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 5e-4

    print("⏳ Loading dataset...")

    dataset = QuickDrawMultiClassDataset(
        folder_path=r"C:\Users\croma\air-canvas-intent-aware\data\quickdraw"
    )

    print(f"✅ Dataset loaded | Samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,   # 🔥 IMPORTANT: avoid Windows multiprocessing freeze
    )

    model = AirCanvasModel().to(device)
    model.train()

    projection.train()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projection.parameters()),
        lr=LR
    )

    print("🚀 Starting training loop...")

    for epoch in range(EPOCHS):
        print(f"\n🔥 Epoch {epoch+1}/{EPOCHS}")

        for step, batch in enumerate(loader):

            motion = batch["motion"].to(device)

            # 🔥 CLIP embeddings (cached)
            emb_list = [text_to_embedding(lbl) for lbl in batch["label"]]
            semantic_text_emb = torch.stack(emb_list)

            # 🔥 project 512 → 256
            semantic_text_emb = projection(semantic_text_emb)

            # 🔥 normalize
            semantic_text_emb = torch.nn.functional.normalize(
                semantic_text_emb, dim=-1
            )

            outputs = model(motion)

            batch["semantic_text_emb"] = semantic_text_emb

            losses = compute_losses(outputs, batch)
            total_loss = losses["total"]

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 20 == 0:
                print(
                    f"Step {step} | "
                    f"Total: {total_loss.item():.3f} | "
                    f"Text: {losses['text'].item():.3f} | "
                    f"Sem: {losses['semantic'].item():.3f}"
                )

    print("\n✅ Training COMPLETE!")


if __name__ == "__main__":
    main()