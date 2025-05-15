#!/usr/bin/env python
# coding: utf-8

# Work based on "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Dat" by Tuli et. al. https://arxiv.org/abs/2201.07284
# Original implementation https://github.com/imperial-qore/TranAD


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
os.chdir("..")  # Go up one level to the UROP directory


# Settings
SEED = None
TRAIN_BATCH_SIZE = 1 # on-line learning
TEST_BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 1e-3

# Model hyper-parameters
PE_DROPOUT = 0.1
TF_DROPOUT = 0.1
D_MODEL = 64
N_HEAD = 8
NUM_LAYERS = 4
DIM_FF = 128


import torch
from torch.utils.data import DataLoader
from src import LazyWindowedDataset, train_test_split

#torch.manual_seed(SEED)

# Initialize Dataset
full_train_source_dataset = LazyWindowedDataset(
    root_dir="datasets/RoboticArm",
    split="train",
    anomaly_type=['normal'],
    domain_type=['source', 'target'],
    window_size_ms=100,
    stride_ms=50,
)

train_source_dataset, val_source_dataset = train_test_split(full_train_source_dataset)

train_loader = DataLoader(train_source_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_loader = DataLoader(val_source_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)

test_source_dataset = LazyWindowedDataset(
    root_dir="datasets/RoboticArm",
    split="test",
    anomaly_type=['normal', 'anomaly'],
    domain_type=['source', 'target'],
    window_size_ms=100,
    stride_ms=50,
)
test_loader = DataLoader(test_source_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)


import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=6000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape to [1, max_len, d_model] so it can be broadcast across batch
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]  # Add positional encoding
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1, pe_dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=pe_dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

        self.config = {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'pe_dropout': pe_dropout
        }

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.output_proj(x)
        return x

    def get_config(self,):
        return self.config

    @staticmethod
    def from_config(config):
        return TimeSeriesTransformer(
            input_dim=config['input_dim'],
            d_model=config.get('d_model', 64),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 3),
            dim_feedforward=config.get('dim_feedforward', 128),
            dropout=config.get('dropout', 0.1),
            pe_dropout=config.get('pe_dropout', 0.1)
        )


config = {
    'input_dim': 7,
    'd_model': D_MODEL,
    'nhead': N_HEAD,
    'num_layers': NUM_LAYERS,
    'dim_feedforward': DIM_FF,
    'dropout': TF_DROPOUT,
    'pe_dropout': PE_DROPOUT
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TimeSeriesTransformer.from_config(config).to(device)
optimizer = torch.optim.AdamW(model.parameters())
criterior = torch.nn.MSELoss()


from torchinfo import summary
summary(model, input_size=(TRAIN_BATCH_SIZE, 1600, 7))


from src import train_model, evaluate

try:
    train_model(
        name='abra', 
        model=model, 
        criterion=criterior, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        val_loader=val_loader, # Skip validation to speed up
        merge_startegy='stack',
        num_epochs=1, 
        verbose=1,
        train_num_batches=50,
        val_num_batches=50,
        save_every=1,
    )
except KeyboardInterrupt:
    print("Training interrupted by user.")


loss, auc = evaluate(model, test_loader, criterior, merge_strategy='stack', verbose=True)
print(f"Overall S+T | Loss: {loss:.4f}, AUC: {auc:.4f}")




