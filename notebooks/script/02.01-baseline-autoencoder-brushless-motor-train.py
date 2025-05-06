#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
os.chdir("..")  # Go up one level to the UROP directory


import torch
from torch.utils.data import DataLoader
import yaml

import librosa
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

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


from src import LazyWindowedDataset, train_test_split

# Initialize Dataset
full_train_source_dataset = LazyWindowedDataset(
    root_dir="datasets/BrushlessMotor",
    split="train",
    anomaly_type=['normal'],
    domain_type=['source', 'target'],
    window_size_ms=100,
    stride_ms=50,
)

train_source_dataset, val_source_dataset = train_test_split(full_train_source_dataset)

train_loader = DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
val_loader = DataLoader(val_source_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)


test_source_dataset = LazyWindowedDataset(
    root_dir="datasets/BrushlessMotor",
    split="test",
    anomaly_type=['normal', 'anomaly'],
    domain_type=['source', 'target'],
    window_size_ms=100,
    stride_ms=50,
)
test_loader = DataLoader(test_source_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)


from models import BaselineAutoencoder
from src import train_one_epoch, evaluate, train_model, z_score_normalize, flatten_and_concat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BaselineAutoencoder()
model.to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


loss, auc = evaluate(model, test_loader, criterion)
print(f"Initial loss: {loss:.4f}, AUC: {auc:.4f}")


model, train_losses, val_losses, val_aucs = train_model(
    name="BrushlessMotor",
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    save_every=5,
    save_dir=save_dir,
    verbose=False,
)


loss, auc = evaluate(model, test_loader, criterion)
print(f"Final loss: {loss:.4f}, AUC: {auc:.4f}")


# 
