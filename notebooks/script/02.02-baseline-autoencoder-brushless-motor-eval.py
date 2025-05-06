#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
os.chdir("..")  # Go up one level to the UROP directory


import yaml

with open("config.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

batch_size = config["batch_size"]
lr = config["lr"]
num_epochs = config["num_epochs"]
save_every = config["save_every"]
save_dir = config["save_dir"]
num_workers = config["num_workers"]


import torch
from torch.utils.data import DataLoader
from src import LazyWindowedDataset

test_source_dataset = LazyWindowedDataset(
    root_dir="datasets/BrushlessMotor",
    split="test",
    anomaly_type=['normal', 'anomaly'],
    domain_type=['source', 'target'],
    window_size_ms=100,
    stride_ms=50,
)
test_loader = DataLoader(test_source_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)


from models import BaselineAutoencoder
from src import evaluate, load_model_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, _, _, _, _, _ = load_model_checkpoint("checkpoints/BrushlessMotorBaseline.pt", BaselineAutoencoder, optimizer_class=None)
model.to(device)
criterion = torch.nn.MSELoss()


# ![alt text](figures/brushlessMotorHighlightedTable.png)

loss, auc = evaluate(model, test_loader, criterion)
print(f"Overall S+T | Loss: {loss:.4f}, AUC: {auc:.4f}")


acc_loss, acc_auc = evaluate(model, test_loader, criterion, sensors_to_test=['acc'])
print(f"Acc s+T | Loss: {acc_loss:.4f}, AUC: {acc_auc:.4f}")


gyro_loss, gyro_auc = evaluate(model, test_loader, criterion, sensors_to_test=['gyro'])
print(f"Gyro S+T | Loss: {gyro_loss:.4f}, AUC: {gyro_auc:.4f}")


mic_loss, mic_auc = evaluate(model, test_loader, criterion, sensors_to_test=['mic'])
print(f"Mic S+T | Loss: {mic_loss:.4f}, AUC: {mic_auc:.4f}")




