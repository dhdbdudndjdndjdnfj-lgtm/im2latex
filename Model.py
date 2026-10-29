import os
import time
import torch
import torch.nn as nn

from Encoder import CNN_encoder
from Decoder import Decoder
from loss import loss_fn
from validate import validate, evaluate_generation
from early_stoping import early_stop
from checkpoints import save_checkpoint, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"


def autocast():
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP)


class Model(nn.Module):
    def __init__(self, vocab_size, enc_dim, dropout=0.25):  # embed_dim = 2*enc_dim
        super().__init__()
        self.config = {"vocab_size": vocab_size, "enc_dim": enc_dim, "dropout": dropout}

        self.encoder = CNN_encoder(enc_dim=enc_dim)
        self.decoder = Decoder(enc_dim * 2, vocab_size=vocab_size, dropout=dropout)

    def forward(self, image, x_tokens):
        x_enc = self.encoder(image)
        return self.decoder(x_tokens, x_enc)

    def predict(self, image, max_len=200):
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            x_enc = self.encoder(image)
            tokens = self.decoder.generate(x_enc, max_len=max_len)
        if was_training:
            self.train()
        return tokens


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, epochs=100, lr=1e-4,
                 weight_decay=1e-2, clip_norm=1.0, patience=5, ckpt_dir=None,
                 vocab=None, eval_every=1, gen_eval_batches=5, gen_max_len=150):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.clip_norm = clip_norm
        self.ckpt_dir = ckpt_dir
        self.vocab = vocab
        self.eval_every = eval_every
        self.gen_eval_batches = gen_eval_batches
        self.gen_max_len = gen_max_len

        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "embed" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr,
        )
        self.scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2) if val_loader is not None else None)
        self.early = early_stop(patience=patience)

    def train_epoch(self, scaler):
        self.model.train()
        total_loss, count, skipped = 0.0, 0, 0

        for image, formula in self.train_loader:
            image = image.to(device)
            formula = formula.to(device)

            x_tokens = formula[:, :-1]
            y_tokens = formula[:, 1:]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = self.model(image, x_tokens)
                loss = loss_fn(logits, y_tokens)

            if not torch.isfinite(loss):
                skipped += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            scaler.step(self.optimizer)
            scaler.update()

            total_loss += loss.item()
            count += 1

        if skipped:
            print(f"  ! пропущено батчей с нечисловым loss: {skipped}")
        return total_loss / max(count, 1)

    def _save(self, name, scaler, epoch, best_val_loss):
        if self.ckpt_dir is None:
            return
        save_checkpoint(os.path.join(self.ckpt_dir, name), self.model, self.optimizer, scaler,
                        epoch, best_val_loss, vocab=self.vocab, scheduler=self.scheduler,
                        early=self.early, config=getattr(self.model, "config", None))

    def train(self, resume=None):
        scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP)
        best_val_loss = float("inf")
        start_epoch = 0

        if resume:
            ckpt = load_checkpoint(resume, self.model, self.optimizer, scaler,
                                   self.scheduler, self.early, device)
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt["best_val_loss"]
            print(f"Возобновление с эпохи {start_epoch}, best_val={best_val_loss:.4f}")

        for epoch in range(start_epoch, self.epochs):
            t0 = time.time()
            train_loss = self.train_epoch(scaler)
            msg = f"Epoch {epoch}: train={train_loss:.4f}"
            stop = False

            if self.val_loader is not None:
                val_loss = validate(self.model, self.val_loader, device)
                msg += f", val={val_loss:.4f}"

                if (epoch + 1) % self.eval_every == 0:
                    m = evaluate_generation(self.model, self.val_loader, device,
                                            max_batches=self.gen_eval_batches,
                                            max_len=self.gen_max_len)
                    msg += f", EM={m['exact_match']:.3f}, ED={m['norm_edit_distance']:.3f}"

                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save("best.pth", scaler, epoch, best_val_loss)
                    msg += "  [saved best]"

                stop = self.early.step(val_loss)

            msg += f", lr={self.optimizer.param_groups[0]['lr']:.1e}, {time.time() - t0:.0f}s"
            print(msg)

            self._save("last.pth", scaler, epoch, best_val_loss)

            if stop:
                print(f"Early stopping at epoch {epoch}")
                break

        return best_val_loss
