import torch
import torch.nn as nn


def conv_block(c_in, c_out):
    return [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]


class CNN_encoder(nn.Module):

    def __init__(self, enc_dim, max_h=32, max_w=128):
        super().__init__()
        layers = []
        layers += conv_block(1, 64)
        layers.append(nn.MaxPool2d(2, 2))           
        layers += conv_block(64, 128)
        layers.append(nn.MaxPool2d(2, 2))            
        layers += conv_block(128, 256)
        layers += conv_block(256, 256)
        layers.append(nn.MaxPool2d(2, 2))            
        layers += conv_block(256, 512)
        layers.append(nn.MaxPool2d((2, 1), (2, 1)))  
        layers += conv_block(512, enc_dim)
        self.encoder = nn.Sequential(*layers)

        self.row_encoder = nn.LSTM(enc_dim, enc_dim, batch_first=True, bidirectional=True)
        self.enc_dim = enc_dim * 2

        self.row_embed = nn.Embedding(max_h, self.enc_dim)
        self.col_embed = nn.Embedding(max_w, self.enc_dim)

    def forward(self, x):
        x = self.encoder(x)                  
        x = x.permute(0, 2, 3, 1)            
        B, H, W, C = x.shape

        x, _ = self.row_encoder(x.reshape(B * H, W, C))
        x = x.reshape(B, H, W, -1)

        rows = torch.arange(H, device=x.device)
        cols = torch.arange(W, device=x.device)
        x = x + self.row_embed(rows)[None, :, None, :] + self.col_embed(cols)[None, None, :, :]

        return x.reshape(B, H * W, -1)
