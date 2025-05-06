#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
os.chdir("..")  # Go up one level to the UROP directory


# Hyper parameters and settings
SEED = 1
BATCH_SIZE = 128
NUM_WORKERS = 4
LR = 1e-3


import torch
from torch.utils.data import DataLoader
from src import LazyWindowedDataset, train_test_split

torch.manual_seed(SEED)

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

train_loader = DataLoader(train_source_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_loader = DataLoader(val_source_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=True)


import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class TranAD(nn.Module):
	def __init__(self, feats, n_window):
		super(TranAD, self).__init__()
		self.name = 'TranAD' # unused
		self.lr = LR # unused
		self.batch = BATCH_SIZE # unused

		self.n_feats = feats
		self.n_window = n_window

		self.n = self.n_feats * self.n_window # unused

		self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)

		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)

		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)

		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2


import warnings
import time
from src import train_model, adjust_time_series_size, minmax_normalize, save_model_checkpoint
from torch.nn import MSELoss
from torch.optim import AdamW

def train_one_epoch(model, dataloader, optimizer, criterion, verbose=False):
    """Train the model for one epoch"""
    device = next(model.parameters()).device  # Get the device of the model

    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss

    for batch_idx, (mic, acc, gyro, labels) in enumerate(dataloader):
        # Normalize input tensors
        time_size = mic.shape[1]
        mic_norm = minmax_normalize(adjust_time_series_size(mic, time_size, 'repeat_start'))
        acc_norm = minmax_normalize(adjust_time_series_size(acc, time_size, 'repeat_start'))
        gyro_norm = minmax_normalize(adjust_time_series_size(gyro, time_size, 'repeat_start'))
        feat_size = mic.shape[2] + acc.shape[2] + gyro.shape[2]

        # Concatenate inputs
        inputs = torch.cat([mic_norm, acc_norm, gyro_norm], dim=2).to(device)
        print(inputs.shape)

        local_bs = inputs.shape[0]
        window = inputs.permute(1, 0, 2)
        print(window.shape)
        elem = window[-1, :, :].unsqueeze(dim=0)
        print(elem.shape)
        print(local_bs, feat_size)
        elem = elem.view(1, local_bs, feat_size)

        optimizer.zero_grad()  # Reset gradients
        outputs = model(window, elem)  # Forward pass on [128, 1600, 7]
        loss = criterion(outputs, inputs)  # Compute loss
        print(loss.shape)
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss

        if verbose:
            print(f"Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.6f}")

    return running_loss / len(dataloader)  # Return average loss for the epoch

def train_model(name, model, criterion, optimizer, train_loader, val_loader,
                start_epoch=0, num_epochs=10, save_every=1, 
                save_dir='checkpoints', verbose=True):
    """Function to train the model for multiple epochs"""
    train_losses, val_losses, val_aucs = [], [], []  # Initialize lists to store metrics

    # Get model config
    if hasattr(model, 'get_config'):
        model_config = model.get_config()
    else:
        warnings.warn(f"{model.__class__.__name__} does not have a 'get_config' method. Setting model_config to None.")
        model_config = None

    for epoch in range(num_epochs):
        adjusted_current_epoch = start_epoch + epoch

        # Start timing the epoch
        start_time = time.time()

        # Train for one epoch and evaluate on validation set
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, verbose)
        # val_loss, val_auc = evaluate(model, val_loader, criterion, ['mic', 'acc', 'gyro'], verbose)

        # End timing the epoch
        epoch_time = time.time() - start_time
        eta = epoch_time * (num_epochs - epoch - 1)

        # Store metrics
        train_losses.append(train_loss)
        # val_losses.append(val_loss)
        # val_aucs.append(val_auc)

        # Print metrics for the current epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] (Checkpoint Epoch: {adjusted_current_epoch + 1}) | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.4f}")
        print(f"Time Spent: {epoch_time:.2f}s | ETA: {eta:.2f}s | Current Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        if (save_every and (epoch + 1) % save_every == 0) or (epoch == num_epochs - 1):
            # Save model checkpoint periodically with adjusted epoch
            save_model_checkpoint(save_dir, name, model, model_config, optimizer,
                                  adjusted_current_epoch + 1, train_losses, val_losses, val_aucs)

    return model, train_losses, val_losses, val_aucs  # Return trained model and metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TranAD(
    feats=7, # number of features
    n_window=1600, # window size
).to(device)
optimizer = AdamW(model.parameters(), lr=LR)

train_model(
    name="TranAD",
    model=model,
    criterion=MSELoss,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=None,
    start_epoch=0,
    num_epochs=5,
    save_every=5,
    save_dir='checkpoints',
    verbose=True,
)




