"""Обучение на MathWriting. Все настройки меняются в блоке ниже, аргументы не нужны."""
import os
import time
 
import torch
from torch.utils.data import DataLoader, Subset
 
from MultyDataset import MathWritting, Base_dataset
from Model import Model, Trainer
 
# ===================== НАСТРОЙКИ =====================
N_TRAIN = 200     # сколько обучающих картинок брать (None = весь train)
N_VAL = 5         # сколько картинок брать для валидации
EPOCHS = 30
BATCH = 32           
LR = 1e-4
WORKERS = 0         # если зависнет, ставьте 0
PATIENCE = 8        # эпох без улучшения до early stopping
RESUME = r"checkpoints\20261017_214901\best.pth"      # например "checkpoints/<папка>/last.pth"
# =====================================================
 
 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    train_ds = MathWritting(mode="train", aug=True)
    val_ds = MathWritting(mode="val", aug=False)
 
    n_train = min(N_TRAIN or len(train_ds), len(train_ds))
    n_val = min(N_VAL, len(val_ds))
 
    common = dict(collate_fn=Base_dataset.collate_fn, num_workers=WORKERS,
                  pin_memory=device.type == "cuda", persistent_workers=WORKERS > 0)
    train_loader = DataLoader(Subset(train_ds, range(n_train)), batch_size=BATCH,
                              shuffle=True, **common)
    val_loader = DataLoader(Subset(val_ds, range(n_val)), batch_size=BATCH,
                            shuffle=False, **common)
 
    print(f"Train: {n_train}, Val: {n_val}, Словарь: {len(train_ds.vocab)}, "
          f"шагов за эпоху: {len(train_loader)}")
 
    model = Model(vocab_size=len(train_ds.vocab), enc_dim=256)
 
    ckpt_dir = (os.path.dirname(RESUME) if RESUME
                else f"checkpoints/{time.strftime('%Y%m%d_%H%M%S')}_full")
    print(f"Чекпоинты: {ckpt_dir}")
 
    trainer = Trainer(model, train_loader, val_loader, epochs=EPOCHS, lr=LR,
                      patience=PATIENCE, ckpt_dir=ckpt_dir, vocab=train_ds.vocab,
                      eval_every=1, gen_eval_batches=16, gen_max_len=150)
 
    print("\n--- Обучение ---")
    trainer.train(resume=RESUME)
 
    best = os.path.join(ckpt_dir, "best.pth")
    if os.path.exists(best):
        ckpt = torch.load(best, map_location="cpu")
        print(f"\nbest.pth: epoch={ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.4f}")
        print(f"Путь для проверок: {best}")
 
 
if __name__ == "__main__":
    main()
