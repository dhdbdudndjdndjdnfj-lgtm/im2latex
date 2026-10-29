import os
import torch


def save_checkpoint(path, model, optimizer, scaler, epoch, best_val_loss,
                    vocab=None, scheduler=None, early=None, config=None):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'early_state': early.state_dict() if early is not None else None,
        'best_val_loss': best_val_loss,
        'vocab': vocab,
        'config': config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None, early=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler is not None and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if early is not None and ckpt.get('early_state') is not None:
        early.load_state_dict(ckpt['early_state'])
    return ckpt
