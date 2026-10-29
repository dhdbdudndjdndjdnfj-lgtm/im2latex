import torch
import torch.nn as nn
import torch.nn.functional as F


class Self_CrossAttention(nn.Module):

    def __init__(self, enc_dim, dropout=0.25):
        super().__init__()

        self.Q_self = nn.Linear(enc_dim, enc_dim)
        self.K_self = nn.Linear(enc_dim, enc_dim)
        self.V_self = nn.Linear(enc_dim, enc_dim)

        self.Q_cross = nn.Linear(enc_dim, enc_dim)
        self.K_cross = nn.Linear(enc_dim, enc_dim)
        self.V_cross = nn.Linear(enc_dim, enc_dim)

        self.enc_dim = enc_dim
        self.dropout_p = dropout

    def forward_self(self, x):
        q, k, v = self.Q_self(x), self.K_self(x), self.V_self(x)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def forward_self_step(self, x, cache=None):
        q, k, v = self.Q_self(x), self.K_self(x), self.V_self(x)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=1)
            v = torch.cat([cache[1], v], dim=1)
        out = F.scaled_dot_product_attention(q, k, v)
        return out, (k, v)

    def forward_cross(self, x_dec, x_enc):
        q = self.Q_cross(x_dec)
        k = self.K_cross(x_enc)
        v = self.V_cross(x_enc)
        p = self.dropout_p if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, dropout_p=p)
