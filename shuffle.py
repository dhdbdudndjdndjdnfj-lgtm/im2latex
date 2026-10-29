"""Проверка: использует ли модель изображение.
 
Запуск: python shuffle_test.py [--ckpt checkpoints/<папка>/best.pth]
"""
import argparse
import glob
import os
 
import torch
from torch.utils.data import DataLoader, Subset
 
from MultyDataset import MathWritting, Base_dataset
from Model import Model
from validate import validate
 
 
class Shuffled:
    """Обёртка над loader: перемешивает картинки внутри батча."""
    def __init__(self, loader):
        self.loader = loader
 
    def __len__(self):
        return len(self.loader)
 
    def __iter__(self):
        for img, f in self.loader:
            yield img[torch.randperm(len(img))], f
 
 
class Blank:
    """Обёртка над loader: заменяет все картинки нулями."""
    def __init__(self, loader):
        self.loader = loader
 
    def __len__(self):
        return len(self.loader)
 
    def __iter__(self):
        for img, f in self.loader:
            yield torch.zeros_like(img), f
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5, help="сколько раз повторить перемешивание")
    args = parser.parse_args()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
 
    ckpt_path = args.ckpt
    if ckpt_path is None:
        found = sorted(glob.glob("checkpoints/*/best.pth"), key=os.path.getmtime)
        if not found:
            raise SystemExit("Чекпоинт не найден: укажите --ckpt")
        ckpt_path = found[-1]
 
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Model(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Загружено {ckpt_path} (epoch {ckpt['epoch']})")
 
    val_ds = MathWritting(mode="val", aug=False)
    val_loader = DataLoader(Subset(val_ds, range(min(args.n_val, len(val_ds)))),
                            batch_size=args.batch, shuffle=False,
                            collate_fn=Base_dataset.collate_fn)
 
    normal = validate(model, val_loader, device)
    shuffled = [validate(model, Shuffled(val_loader), device) for _ in range(args.repeats)]
    shuffled_mean = sum(shuffled) / len(shuffled)
    blank = validate(model, Blank(val_loader), device)
 
    print(f"\nval loss, обычные картинки:       {normal:.4f}")
    print(f"val loss, картинки перемешаны:    {shuffled_mean:.4f}  (по {args.repeats} прогонам: "
          f"{min(shuffled):.4f}..{max(shuffled):.4f})")
    print(f"val loss, пустые картинки:        {blank:.4f}")
    print(f"\nРазница (перемешано - обычные):   {shuffled_mean - normal:+.4f}")
    print(f"Разница (пустые - обычные):       {blank - normal:+.4f}")
    gap = shuffled_mean - normal
    # Пороги эвристические: ориентир, а не строгий критерий
    if gap < 0.05:
        verdict = "модель почти не использует изображение"
    elif gap < 0.2:
        verdict = "изображение используется слабо"
    else:
        verdict = "изображение заметно влияет на предсказание"
    print(f"\nВывод (по перемешанным картинкам, эвристика): {verdict}.")
 
 
if __name__ == "__main__":
    main()