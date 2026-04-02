import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from models.air_canvas_model import AirCanvasModel
from data.quickdraw_multiclass_dataset import QuickDrawMultiClassDataset

from torch.utils.data import DataLoader


def collate_fn(batch):
    motion = torch.stack([b["motion"] for b in batch])
    labels = [b["label"] for b in batch]

    return {
        "motion": motion,
        "label": labels
    }


def main():
    dataset = QuickDrawMultiClassDataset(
        folder_path=r"C:\Users\croma\air-canvas-intent-aware\data\quickdraw"
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = AirCanvasModel()
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i > 50:   # 🔥 MORE DATA
                break

            motion = batch["motion"]

            output = model(motion)

            # 🔥 correct key
            gesture_emb = output["drawing_emb"]

            # 🔥 normalize
            gesture_emb = torch.nn.functional.normalize(
                gesture_emb, dim=-1
            )

            embeddings.append(gesture_emb)
            labels.extend(batch["label"])

    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()

    # 🔥 t-SNE (BETTER THAN PCA)
    tsne = TSNE(n_components=2, perplexity=30)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    unique_labels = list(set(labels))

    for lab in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=lab, alpha=0.7)

    plt.legend()
    plt.title("t-SNE Embedding Visualization")
    plt.xlabel("Dim-1")
    plt.ylabel("Dim-2")

    plt.show()


if __name__ == "__main__":
    main()