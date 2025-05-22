#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
os.chdir("..")  # Go up one level to the UROP directory


# Settings
TEST_BATCH_SIZE = 128
NUM_WORKERS = 4

# Dataset hyper-parameters
WINDOW_SIZE_MS=100
STRIDE_MS=50


import torch
from torch.utils.data import DataLoader
from src import LazyWindowedDataset

test_source_dataset = LazyWindowedDataset(
    root_dir="datasets/RoboticArm",
    split="test",
    anomaly_type=['normal', 'anomaly'],
    domain_type=['source', 'target'],
    window_size_ms=WINDOW_SIZE_MS,
    stride_ms=STRIDE_MS,
)
test_loader = DataLoader(test_source_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)


from models import RoPeTimeSeriesTransformer
from src import evaluate, load_model_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, _, _, _, _, _, _ = load_model_checkpoint("checkpoints/ra_aerodactyl_RoPeTimeSeriesTransformer_epoch_1.pt", RoPeTimeSeriesTransformer, optimizer_class=None)
model.to(device)
criterion = torch.nn.MSELoss()


loss, auc = evaluate(model, test_loader, criterion, merge_strategy='stack', verbose=1)
print(f"Overall S+T | Loss: {loss:.4f}, AUC: {auc:.4f}")


acc_loss, acc_auc = evaluate(model, test_loader, criterion, sensors_to_test=['acc'], merge_strategy='stack', verbose=1)
print(f"Acc s+T | Loss: {acc_loss:.4f}, AUC: {acc_auc:.4f}")


gyro_loss, gyro_auc = evaluate(model, test_loader, criterion, sensors_to_test=['gyro'], merge_strategy='stack', verbose=1)
print(f"Gyro S+T | Loss: {gyro_loss:.4f}, AUC: {gyro_auc:.4f}")


mic_loss, mic_auc = evaluate(model, test_loader, criterion, sensors_to_test=['mic'], merge_strategy='stack', verbose=1)
print(f"Mic S+T | Loss: {mic_loss:.4f}, AUC: {mic_auc:.4f}")

