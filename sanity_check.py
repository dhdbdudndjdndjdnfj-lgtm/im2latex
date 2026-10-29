"""Тест на переобучение: модель должна выучить 16 картинок наизусть."""

import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from MultyDataset import MathWritting, Base_dataset
from Model import Model, Trainer, device
from metrics import edit_distance

N, EPOCHS = 16, 300


def main():
    ds = MathWritting(mode="train", aug=False)
    subset = Subset(ds, range(N))

    os.makedirs("debug_images", exist_ok=True)
    for i in range(N):
        img, f = subset[i]
        save_image(img, f"debug_images/{i}.png")
        print(i, " ".join(ds.digit2tok[t] for t in f[1:-1]))

    loader = DataLoader(subset, batch_size=8, shuffle=True, collate_fn=Base_dataset.collate_fn)
    model = Model(vocab_size=len(ds.vocab), enc_dim=256, dropout=0.0)
    trainer = Trainer(model, loader, None, epochs=EPOCHS, lr=3e-4, weight_decay=0.0)

    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    for epoch in range(EPOCHS):
        loss = trainer.train_epoch(scaler)
        if (epoch + 1) % 25 == 0:
            print(f"epoch {epoch + 1}: train loss = {loss:.4f}")

    images, formulas = next(iter(DataLoader(subset, batch_size=N, collate_fn=Base_dataset.collate_fn)))
    max_len = formulas.size(1) + 5
    preds = model.predict(images.to(device), max_len=max_len)

    exact = 0
    for i, (pred, row) in enumerate(zip(preds, formulas.tolist())):
        ref = row[1:]
        ref = ref[:ref.index(2)]
        exact += int(pred == ref)
        if pred != ref:
            print(f"[{i}] ED={edit_distance(pred, ref)}\n  pred: {' '.join(ds.digit2tok[t] for t in pred)}"
                  f"\n  true: {' '.join(ds.digit2tok[t] for t in ref)}")
    print(f"Exact match на обучающих 16: {exact}/{N}")


if __name__ == "__main__":
    main()
