import torch
import torch.nn as nn
from Attention import Self_CrossAttention

PAD_ID, START_ID, END_ID = 0, 1, 2


class Decoder(nn.Module):

    def __init__(self, embed_dim, vocab_size=2227, dropout=0.25):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        self.attention = Self_CrossAttention(embed_dim, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(2 * embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, x_tokens, x_enc):
        emb = self.embedding(x_tokens)
        out, _ = self.lstm(emb)

        self_out = self.norm(out + self.attention.forward_self(out))
        context = self.attention.forward_cross(self_out, x_enc)

        combined = self.dropout(torch.cat([self_out, context], dim=-1))
        return self.fc(combined)

    def generate(self, x_enc, max_len=200):
        """Возвращает список списков id токенов без <start> и <end>."""
        B = x_enc.size(0)
        device = x_enc.device

        token = torch.full((B, 1), START_ID, dtype=torch.long, device=device)
        hidden = None   
        cache = None    
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated = []

        for _ in range(max_len):
            emb = self.embedding(token)
            out, hidden = self.lstm(emb, hidden)

            sa, cache = self.attention.forward_self_step(out, cache)
            self_out = self.norm(out + sa)
            context = self.attention.forward_cross(self_out, x_enc)

            logits = self.fc(torch.cat([self_out, context], dim=-1))
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = next_token.masked_fill(finished, PAD_ID)
            generated.append(next_token)

            finished = finished | (next_token == END_ID)
            if finished.all():
                break
            token = next_token.unsqueeze(1)

        seqs = torch.stack(generated, dim=1).tolist()
        result = []
        for s in seqs:
            if END_ID in s:
                s = s[:s.index(END_ID)]
            result.append(s)
        return result
