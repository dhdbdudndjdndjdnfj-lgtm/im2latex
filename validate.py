import torch
from loss import loss_fn
from metrics import edit_distance

END_ID = 2


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=(device.type == "cuda"))


@torch.no_grad()
def validate(model, val_loader, device):
    """Loss с teacher forcing (НЕ отражает качество реальной генерации!!!)."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    for image, formula in val_loader:
        image = image.to(device)
        formula = formula.to(device)
        x_tokens = formula[:, :-1]
        y_tokens = formula[:, 1:]
        with _autocast(device):
            logits = model(image, x_tokens)
            loss = loss_fn(logits, y_tokens)
        total_loss += loss.item()
    if was_training:
        model.train()
    return total_loss / len(val_loader)


@torch.no_grad()
def evaluate_generation(model, loader, device, max_batches=5, max_len=150):
    """Метрики на реальной генерации: exact match и нормированный edit distance."""
    was_training = model.training
    model.eval()
    exact, dist, n = 0, 0.0, 0
    for b, (image, formula) in enumerate(loader):
        if b >= max_batches:
            break
        preds = model.predict(image.to(device), max_len=max_len)
        for pred, row in zip(preds, formula.tolist()):
            ref = row[1:]
            ref = ref[:ref.index(END_ID)] if END_ID in ref else [t for t in ref if t != 0]
            exact += int(pred == ref)
            dist += edit_distance(pred, ref) / max(len(ref), 1)
            n += 1
    if was_training:
        model.train()
    n = max(n, 1)
    return {"exact_match": exact / n, "norm_edit_distance": dist / n}
