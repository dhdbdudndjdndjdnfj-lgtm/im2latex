"""Оценка обученных весов на всём val-сплите MathWriting 

Считает: процент точных совпадений (exact match), средний нормированный edit distance,
долю формул с <=1 и <=2 ошибками, точность по группам длины и сохраняет ошибки в TSV.

Все параметры меняются в блоке НАСТРОЙКИ ниже, аргументы командной строки не нужны.
"""
import glob
import os
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

from MultyDataset import MathWritting, Base_dataset
from Model import Model
from metrics import edit_distance

END_ID, UNK_ID = 2, 3


# ===================== НАСТРОЙКИ =====================
CKPT = r"Project_path\checkpoints\first_educate_check\best.pth"   # путь к весам; None = самый новый checkpoints/*/best.pth
N = 100                                  # сколько примеров val взять (None = все)
BATCH = 4                                # при достатке памяти GPU можно поднять до 32
MAX_LEN = 250                            # максимум токенов при генерации
WORKERS = 0                              # на Windows при проблемах поставьте 0
SAVE_ERRORS = "val_errors.tsv"           # куда сохранить ошибки
# =====================================================


def main():
    args = SimpleNamespace(ckpt=CKPT, n=N, batch=BATCH, max_len=MAX_LEN,
                           workers=WORKERS, save_errors=SAVE_ERRORS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = args.ckpt
    if ckpt_path is None:
        found = sorted(glob.glob("checkpoints/*/best.pth"), key=os.path.getmtime)
        if not found:
            raise SystemExit("Чекпоинт не найден: укажите путь в CKPT")
        ckpt_path = found[-1]

    # Относительный путь ищем и от рабочей папки, и от папки скрипта
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(ckpt_path) and not os.path.exists(ckpt_path):
        alt = os.path.join(here, ckpt_path)
        if os.path.exists(alt):
            ckpt_path = alt
    if not os.path.exists(ckpt_path):
        found = []
        for root in dict.fromkeys([here, os.getcwd()]):   # папка скрипта и рабочая папка
            found += glob.glob(os.path.join(root, "**", "best.pth"), recursive=True)
        found = list(dict.fromkeys(found))
        lines = [f"Файл не найден: {ckpt_path}",
                 f"Рабочая папка: {os.getcwd()}",
                 f"Папка скрипта: {here}"]
        if found:
            lines.append("best.pth, найденные в папке скрипта и в рабочей папке (скопируйте нужный путь в CKPT):")
            lines += [f"  {p}" for p in found]
        else:
            lines.append("Ни в папке скрипта, ни в рабочей папке best.pth не найдено.")
        raise SystemExit("\n".join(lines))

    ckpt = torch.load(ckpt_path, map_location=device)
    model = Model(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Загружено {ckpt_path}: epoch={ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.4f}")

    ds = MathWritting(mode="val", aug=False)
    vocab = ckpt.get("vocab") or ds.vocab
    if vocab != ds.vocab:
        print("! Словарь датасета не совпадает со словарём чекпоинта, декодирую по словарю чекпоинта")

    n = len(ds) if args.n is None else min(args.n, len(ds))

    order = sorted(range(n), key=lambda i: len(ds.targets[i]))
    loader = DataLoader(Subset(ds, order), batch_size=args.batch, shuffle=False,
                        collate_fn=Base_dataset.collate_fn, num_workers=args.workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    print(f"Val: {n} формул, батчей: {len(loader)}")

    records = []   # (индекс в датасете, эталон, предсказание)
    pos, t0 = 0, time.time()
    for b, (images, formulas) in enumerate(loader):
        preds = model.predict(images.to(device), max_len=args.max_len)
        for pred, row in zip(preds, formulas.tolist()):
            ref = row[1:]
            ref = ref[:ref.index(END_ID)] if END_ID in ref else [t for t in ref if t != 0]
            records.append((order[pos], ref, pred))
            pos += 1
        if (b + 1) % 20 == 0 or b + 1 == len(loader):
            print(f"  {pos}/{n}  ({time.time() - t0:.0f} с)")
            
    exact = 0
    ed_sum = 0.0
    le1 = le2 = 0
    too_long = unk_in_ref = 0
    buckets = {"1-10": [0, 0], "11-25": [0, 0], "26-50": [0, 0], ">50": [0, 0]}  # [верно, всего]
    errors = []

    for idx, ref, pred in records:
        ed = edit_distance(pred, ref)
        ok = pred == ref
        exact += ok
        ed_sum += ed / max(len(ref), 1)
        le1 += ed <= 1
        le2 += ed <= 2
        too_long += len(ref) > args.max_len
        unk_in_ref += UNK_ID in ref

        L = len(ref)
        key = "1-10" if L <= 10 else "11-25" if L <= 25 else "26-50" if L <= 50 else ">50"
        buckets[key][0] += ok
        buckets[key][1] += 1

        if not ok:
            errors.append((idx, ed, " ".join(vocab[t] for t in ref), " ".join(vocab[t] for t in pred)))

    total = len(records)
    print("\n================ РЕЗУЛЬТАТЫ ================")
    print(f"Формул проверено:                 {total}")
    print(f"Точных совпадений (exact match):  {exact}/{total} = {exact / total:.2%}")
    print(f"С не более чем 1 ошибкой:         {le1 / total:.2%}")
    print(f"С не более чем 2 ошибками:        {le2 / total:.2%}")
    print(f"Средний норм. edit distance:      {ed_sum / total:.3f}   (0 = идеально, 1 = ошибка на каждый токен)")
    print("\nExact match по длине эталона (в токенах):")
    for key, (ok, cnt) in buckets.items():
        if cnt:
            print(f"  {key:>6}: {ok}/{cnt} = {ok / cnt:.2%}")
    print(f"\nЭталонов длиннее max_len={args.max_len} (угадать невозможно): {too_long}")
    print(f"Эталонов с <unk> (угадать невозможно):          {unk_in_ref}")

    errors.sort(key=lambda e: e[1])
    with open(args.save_errors, "w", encoding="utf-8") as f:
        f.write("idx\ted\tэталон\tпредсказание\n")
        for idx, ed, ref, pred in errors:
            f.write(f"{idx}\t{ed}\t{ref}\t{pred}\n")
    print(f"\nОшибки сохранены в {args.save_errors} (отсортированы от самых близких к верным)")


if __name__ == "__main__":
    main()
