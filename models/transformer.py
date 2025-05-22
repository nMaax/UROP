import torch
import torch.nn as nn

class HybridSensorPositionalEncoding(nn.Module):
    def __init__(self, max_len=6000):
        super().__init__()
        # Motion sensors (acc + gyro) use RoPE encoding
        rope_dim = 6  # 3 acc + 3 gyro dimensions
        assert rope_dim % 2 == 0, "RoPE dimension must be even"

        # Generate RoPE frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute RoPE rotation matrices
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos_cached', torch.cos(freqs))
        self.register_buffer('sin_cached', torch.sin(freqs))

        # Sinusoidal positional bias for mic
        div_term = 1.0 / (10000 ** 0.0)  # scalar mic dim
        pe = torch.sin(t.unsqueeze(1) * div_term)  # [T, 1]
        self.register_buffer('mic_pe', pe)  # [T, 1]

    def forward(self, x):
        B, T, D = x.shape
        assert D == 7, "Input must have 7 features (1 mic + 3 acc + 3 gyro), as in [B, T, 7]"

        # Split into mic and motion sensors
        x_mic = x[:, :, 0:1]      # Microphone channel
        x_motion = x[:, :, 1:7]   # Motion sensors

        # Apply RoPE to motion sensors
        x1, x2 = x_motion[..., ::2], x_motion[..., 1::2]  # Split into even/odd
        cos = self.cos_cached[:T].unsqueeze(0)
        sin = self.sin_cached[:T].unsqueeze(0)
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,  # Rotate even components
            x1 * sin + x2 * cos   # Rotate odd components
        ], dim=-1)

        # Add positional encoding to mic
        x_mic_pe = x_mic + self.mic_pe[:T].unsqueeze(0)

        # Combine mic and motion features
        return torch.cat([x_mic_pe, x_rotated], dim=-1)

class RoPeTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=8, num_layers=4, dim_feedforward=64, dropout=0.1):
        super().__init__()
        # Components
        self.pos_encoder = HybridSensorPositionalEncoding()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Build transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection back to input dimensionality
        self.output_proj = nn.Linear(d_model, input_dim)

        # Save config for model serialization
        self.config = {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
        }

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        return self.output_proj(x)

    def get_config(self):
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