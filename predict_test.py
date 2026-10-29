import argparse
import glob
import os

import torch
from torch.utils.data import Subset, DataLoader

from MultyDataset import MathWritting, Base_dataset
from Model import Model
from metrics import edit_distance

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=None, help="путь к best.pth (по умолчанию самый новый в checkpoints/)")
parser.add_argument("--n", type=int, default=5)
parser.add_argument("--max_len", type=int, default=200)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt_path = args.ckpt
if ckpt_path is None:
    found = sorted(glob.glob("checkpoints/*/best.pth"), key=os.path.getmtime)
    if not found:
        raise SystemExit("Чекпоинт не найден: укажите --ckpt")
    ckpt_path = found[-1]

ckpt = torch.load(ckpt_path, map_location=device)
model = Model(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
print(f"Загружено {ckpt_path}: epoch={ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.4f}")

dataset = MathWritting(mode="val", aug=False)
if ckpt.get("vocab") is not None and ckpt["vocab"] != dataset.vocab:
    print("! Словарь датасета не совпадает со словарём чекпоинта, декодирую по словарю чекпоинта")
vocab = ckpt.get("vocab") or dataset.vocab

subset = Subset(dataset, range(args.n))
loader = DataLoader(subset, batch_size=args.n, collate_fn=Base_dataset.collate_fn)
images, formulas = next(iter(loader))
preds = model.predict(images.to(device), max_len=args.max_len)

exact, dist = 0, 0.0
for i, (pred, row) in enumerate(zip(preds, formulas.tolist())):
    ref = row[1:]
    ref = ref[:ref.index(2)] if 2 in ref else [t for t in ref if t != 0]
    ok = pred == ref
    exact += int(ok)
    dist += edit_distance(pred, ref) / max(len(ref), 1)
    print(f"\n--- Картинка {i} ---")
    print("Предсказано:", " ".join(vocab[t] for t in pred))
    print("Правильно:  ", " ".join(vocab[t] for t in ref))
    print("Совпадает:  ", ok)

print(f"\nExact match: {exact}/{len(preds)}, средний норм. edit distance: {dist / len(preds):.3f}")
