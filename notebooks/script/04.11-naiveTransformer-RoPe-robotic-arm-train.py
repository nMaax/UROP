#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
os.chdir("..")  # Go up one level to the UROP directory


import torch


# Settings
SEED = 1
TRAIN_BATCH_SIZE = 1 # on-line learning
TEST_BATCH_SIZE = 128
NUM_WORKERS = 4
LR = 1e-3

# Model hyper-parameters
INPUT_DIM = 7
TF_DROPOUT = 0.0
D_MODEL = 64
N_HEAD = 8
NUM_LAYERS = 4
DIM_FF = 128

# Dataset hyper-parameters
WINDOW_SIZE_MS=100
STRIDE_MS=50


import torch
from torch.utils.data import DataLoader
from src import LazyWindowedDataset, train_test_split

# Set generator for reproducibility
generator = torch.Generator()
generator.manual_seed(SEED)
torch.manual_seed(generator.initial_seed())

# Initialize Dataset
full_train_source_dataset = LazyWindowedDataset(
    root_dir="datasets/RoboticArm",
    split="train",
    anomaly_type=['normal'],
    domain_type=['source', 'target'],
    window_size_ms=WINDOW_SIZE_MS,
    stride_ms=STRIDE_MS,
)

train_source_dataset, val_source_dataset = train_test_split(full_train_source_dataset)

train_loader = DataLoader(train_source_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_loader = DataLoader(val_source_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)

test_source_dataset = LazyWindowedDataset(
    root_dir="datasets/RoboticArm",
    split="test",
    anomaly_type=['normal', 'anomaly'],
    domain_type=['source', 'target'],
    window_size_ms=WINDOW_SIZE_MS,
    stride_ms=STRIDE_MS,
)
test_loader = DataLoader(test_source_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)


from models.transformer import RoPeTimeSeriesTransformer

config = {
    'input_dim': INPUT_DIM,
    'd_model': D_MODEL,
    'nhead': N_HEAD,
    'num_layers': NUM_LAYERS,
    'dim_feedforward': DIM_FF,
    'dropout': TF_DROPOUT,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RoPeTimeSeriesTransformer.from_config(config).to(device)
optimizer = torch.optim.AdamW(model.parameters())
criterior = torch.nn.MSELoss()


from torchinfo import summary
summary(model, input_size=(TRAIN_BATCH_SIZE, 1600, 7))


from src import train_model, evaluate

try:
    train_model(
        name='ra_aerodactyl', 
        model=model, 
        criterion=criterior, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        val_loader=val_loader, # Skip validation to speed up
        merge_startegy='stack',
        num_epochs=1, 
        verbose=1,
        train_num_batches=50, # Do not train on the whole epoch, but some random choosen batches
        val_num_batches=50, # Do not validate on the whole epoch, but some random choosen batches
        save_every=1,
        generator=generator,
    )
except KeyboardInterrupt:
    print("Training interrupted by user.")


loss, auc = evaluate(model, test_loader, criterior, merge_strategy='stack', verbose=True, generator=generator)
print(f"Overall S+T | Loss: {loss:.4f}, AUC: {auc:.4f}")

