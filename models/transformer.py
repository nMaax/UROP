import math
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class HybridSensorPositionalEncoding(nn.Module):
    def __init__(self, max_len=6000):
        super().__init__()
        rope_dim = 6  # acc + gyro
        assert rope_dim % 2 == 0, "RoPE dimension must be even"

        self.rope_dim = rope_dim

        # RoPE frequency buffers
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer('inv_freq', inv_freq)  # [rope_dim/2]

        t = torch.arange(max_len, dtype=torch.float32)  # [T]
        freqs = torch.einsum('i,j->ij', t, inv_freq)    # [T, rope_dim/2]
        self.register_buffer('cos_cached', torch.cos(freqs))  # [T, rope_dim/2]
        self.register_buffer('sin_cached', torch.sin(freqs))  # [T, rope_dim/2]

        # Sinusoidal positional bias for mic
        div_term = 1.0 / (10000 ** 0.0)  # scalar mic dim
        pe = torch.sin(t.unsqueeze(1) * div_term)  # [T, 1]
        self.register_buffer('mic_pe', pe)  # [T, 1]

    def forward(self, x):
        B, T, D = x.shape
        assert D == 7, "Expected input shape [B, T, 7]"

        # Split based on new order
        x_mic = x[:, :, 0:1]      # [B, T, 1]
        x_motion = x[:, :, 1:7]   # [B, T, 6]

        # RoPE on motion
        x1 = x_motion[..., ::2]  # [B, T, 3]
        x2 = x_motion[..., 1::2] # [B, T, 3]
        cos = self.cos_cached[:T].unsqueeze(0)  # [1, T, 3]
        sin = self.sin_cached[:T].unsqueeze(0)  # [1, T, 3]

        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)  # [B, T, 6]

        # Positional bias on mic
        x_mic_pe = x_mic + self.mic_pe[:T].unsqueeze(0)  # [B, T, 1]

        # Reconstruct original feature order
        return torch.cat([x_mic_pe, x_rotated], dim=-1)  # [B, T, 7]

class RoPeTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=8, num_layers=4, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.pos_encoder = HybridSensorPositionalEncoding()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.config = {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
        }

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.pos_encoder(x)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        return x

    def get_config(self,):
        return self.config

    @staticmethod
    def from_config(config):
        return RoPeTimeSeriesTransformer(
            input_dim=config['input_dim'],
            d_model=config.get('d_model', 64),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 3),
            dim_feedforward=config.get('dim_feedforward', 128),
            dropout=config.get('dropout', 0.1),
        )