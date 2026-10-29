import json
import os
import re

import numpy as np
import torch
from PIL import Image, ImageChops
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

PAD, START, END, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ['<pad>', '<start>', '<end>', '<unk>']

IMG_H, IMG_W = 128, 512
TOKENIZER_VERSION = 2

# Во сколько раз формулу можно растянуть сверх масштаба с сохранением пропорций,
# чтобы она плотнее заполняла кадр. 1.0 = без растяжения, большое число = на весь кадр.
MAX_STRETCH = 1.5

# Поля вокруг формулы в кадре (px): запас, чтобы аугментация не обрезала штрихи
PAD_PX = 8

DATA_ROOT = os.environ.get("IM2LATEX_ROOT", "dataset")


class Base_dataset(Dataset):
    _aug = transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), translate=(0.02, 0.03), fill=255)

    def __init__(self, mode="train", aug=False):
        self.mode = mode
        self.aug = aug

    @staticmethod
    def to_gray(image):
        """Прозрачность -> белый фон, в оттенки серого, тёмный фон -> инвертируем."""
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            image = image.convert("RGBA")
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(bg, image)
        image = image.convert("L")
        if np.asarray(image).mean() < 127:
            image = ImageChops.invert(image)
        return image

    @staticmethod
    def trim(image, margin=0, thresh=30):
        """Обрезка по прямоугольнику формулы. thresh отсекает шум и артефакты сжатия."""
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg).point(lambda p: 255 if p > thresh else 0)
        bbox = diff.getbbox()
        if bbox is None:
            return image
        l, t, r, b = bbox
        return image.crop((max(l - margin, 0), max(t - margin, 0),
                           min(r + margin, image.width), min(b + margin, image.height)))

    @staticmethod
    def fit_to_canvas(image):
        """Масштаб с сохранением пропорций + мягкое растяжение (до MAX_STRETCH).
        Формула прижата к левому краю (начало текста всегда в колонке 0), поля PAD_PX со всех сторон."""
        w, h = image.size
        aw, ah = IMG_W - 2 * PAD_PX, IMG_H - 2 * PAD_PX   # доступная область
        fx, fy = aw / w, ah / h                            # масштабы, при которых область по оси заполнена
        s = min(fx, fy)                                    # масштаб с сохранением пропорций
        sx = min(fx, s * MAX_STRETCH)
        sy = min(fy, s * MAX_STRETCH)
        nw, nh = max(1, round(w * sx)), max(1, round(h * sy))
        image = image.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("L", (IMG_W, IMG_H), 255)
        canvas.paste(image, (PAD_PX, PAD_PX + (ah - nh) // 2))
        return canvas

    def preprocess(self, image):
        image = self.fit_to_canvas(self.trim(self.to_gray(image)))
        if self.aug:
            image = self._aug(image)
        return 1.0 - TF.to_tensor(image)

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        formulas = [item[1] for item in batch]
        max_len = max(len(f) for f in formulas)
        buf = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, f in enumerate(formulas):
            buf[i, :len(f)] = torch.tensor(f, dtype=torch.long)
        return images, buf

    @staticmethod
    def tokenizer(formula):
        return re.findall(r"\\[a-zA-Z]+|\\.|\S", formula)

    @staticmethod
    def set_vocab(tokenized):
        all_tokens = set()
        for tokens in tokenized:
            all_tokens.update(tokens)
        return SPECIAL_TOKENS + sorted(all_tokens)

    @classmethod
    def load_or_build_vocab(cls, path, tokenized_fn):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("tokenizer_version") == TOKENIZER_VERSION:
                return data["vocab"]
            print(f"{path}: версия токенизатора устарела, словарь пересобирается")
        vocab = cls.set_vocab(tokenized_fn())
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tokenizer_version": TOKENIZER_VERSION, "vocab": vocab},
                      f, ensure_ascii=False)
        return vocab

    def _init_vocab(self, vocab):
        self.vocab = vocab
        self.tok2digit = {t: i for i, t in enumerate(vocab)}
        self.digit2tok = {i: t for i, t in enumerate(vocab)}

    def encode(self, tokens):
        return [START] + [self.tok2digit.get(t, UNK) for t in tokens] + [END]


class im2latex(Base_dataset):
    def __init__(self, mode="train", aug=False, root=DATA_ROOT, vocab_path="im2latex_vocab.json"):
        super().__init__(mode, aug)
        self.image_dir = os.path.join(root, "formula_images", "formula_images")

        def read(name):
            with open(os.path.join(root, name), 'r', encoding='latin-1') as f:
                return [line.rstrip("\n") for line in f]

        self.formulas = read("im2latex_formulas.lst")
        split_files = {"train": "im2latex_train.lst",
                       "val": "im2latex_validate.lst",
                       "validate": "im2latex_validate.lst"}
        if mode not in split_files:
            raise ValueError(f"Unsupported mode: {mode}")
        self.data = [line.split() for line in read(split_files[mode])]

        def train_tokens():
            rows = [line.split() for line in read("im2latex_train.lst")]
            return [self.tokenizer(self.formulas[int(r[0])]) for r in rows]

        self._init_vocab(self.load_or_build_vocab(vocab_path, train_tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formula_idx, name = int(self.data[idx][0]), self.data[idx][1]
        with Image.open(os.path.join(self.image_dir, name + ".png")) as im:
            image = self.preprocess(im)
        return image, self.encode(self.tokenizer(self.formulas[formula_idx]))


class MathWritting(Base_dataset):
    def __init__(self, mode="train", aug=False, vocab_path="mathwriting_vocab.json"):
        super().__init__(mode, aug)
        from datasets import load_dataset
        self.ds = load_dataset("deepcopy/MathWriting-Human")

        split = {"validate": "val"}.get(mode, mode)
        if split not in self.ds:
            raise ValueError(f"Unsupported mode '{mode}', доступны: {list(self.ds.keys())}")
        self.data = self.ds[split]

        self._init_vocab(self.load_or_build_vocab(
            vocab_path, lambda: [self.tokenizer(f) for f in self.ds["train"]["latex"]]))

        self.formulas = self.data["latex"]
        self.targets = [self.encode(self.tokenizer(f)) for f in self.formulas]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.preprocess(self.data[idx]["image"])
        return image, self.targets[idx]
