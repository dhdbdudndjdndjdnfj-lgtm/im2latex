import torch.nn as nn

_criterion = nn.CrossEntropyLoss(ignore_index=0)


def loss_fn(logits, answers):  # logits [B, seq_len, vocab], answers [B, seq_len]
    B, seq_len, voc_len = logits.shape
    logits_flat = logits.reshape(B * seq_len, voc_len).float()  
    answers_flat = answers.reshape(B * seq_len)
    return _criterion(logits_flat, answers_flat)
