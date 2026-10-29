"""Просмотр обработанных картинок MathWriting.
Сохраняет debug_images/preview.png: для каждой случайной формулы строка из
  [оригинал после to_gray | обработка с MAX_STRETCH = 1.0 | 1.5 | 100]
Так видно, как растяжение влияет на вид формул. Заодно печатает статистику по размерам.
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw

import MultyDataset as md
from MultyDataset import MathWritting, Base_dataset, IMG_H, IMG_W

N = 8                                  # сколько формул показать
STRETCH_VARIANTS = [1.0, 1.5, 100.0]   # 100 = практически на весь кадр
AUG = False                            # True: показать с аугментацией
SEED = 0
OUT = "debug_images/preview.png"
STATS_N = 2000                         # по скольким картинкам считать статистику


def tensor_to_pil(t):
    """Тензор (штрихи=1, фон=0) -> обычная картинка (чёрные штрихи на белом)."""
    arr = ((1.0 - t[0].numpy()) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def raw_panel(img):
    """Оригинал после to_gray, вписанный в панель IMG_W x IMG_H для сравнения."""
    img = Base_dataset.to_gray(img)
    img.thumbnail((IMG_W, IMG_H))
    panel = Image.new("L", (IMG_W, IMG_H), 255)
    panel.paste(img, (0, 0))
    return panel


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds = MathWritting(mode="train", aug=AUG)

    idxs = random.Random(SEED).sample(range(len(ds)), N)
    cols = 1 + len(STRETCH_VARIANTS)
    gap, title_h, label_h = 8, 18, 14
    W = cols * IMG_W + (cols + 1) * gap
    H = title_h + N * (label_h + IMG_H + gap) + gap

    sheet = Image.new("L", (W, H), 200)
    draw = ImageDraw.Draw(sheet)

    headers = ["оригинал (to_gray)"] + [f"MAX_STRETCH={v}" for v in STRETCH_VARIANTS]
    for c, text in enumerate(headers):
        draw.text((gap + c * (IMG_W + gap), 2), text, fill=0)

    for r, i in enumerate(idxs):
        y = title_h + r * (label_h + IMG_H + gap)
        draw.text((gap, y), f"#{i}: {ds.formulas[i][:110]}", fill=0)
        y += label_h

        raw = ds.data[i]["image"]
        sheet.paste(raw_panel(raw), (gap, y))
        for c, v in enumerate(STRETCH_VARIANTS, start=1):
            md.MAX_STRETCH = v
            sheet.paste(tensor_to_pil(ds.preprocess(raw)), (gap + c * (IMG_W + gap), y))

    sheet.save(OUT)
    print(f"Сохранено: {OUT}")

    # ---- статистика по размерам (без растяжения) ----
    wide, small = [], []
    for i in range(min(STATS_N, len(ds))):
        img = Base_dataset.trim(Base_dataset.to_gray(ds.data[i]["image"]))
        w, h = img.size
        scale = min(IMG_H / h, IMG_W / w)
        wide.append(w / h > IMG_W / IMG_H)
        small.append(h * scale < 48)
    print(f"Картинок, где ограничивает ширина:     {np.mean(wide):.1%}")
    print(f"Формул ниже 48 px из {IMG_H} после масштаба: {np.mean(small):.1%}")


if __name__ == "__main__":
    main()
