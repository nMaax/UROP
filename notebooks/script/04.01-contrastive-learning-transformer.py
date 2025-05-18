#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
os.chdir("..")  # Go up one level to the UROP directory


# Settings
SEED = 1
TRAIN_BATCH_SIZE = 64 # on-line learning
TEST_BATCH_SIZE = 128
NUM_WORKERS = 4
LR = 1e-3

# Model hyper-parameters
TF_DROPOUT = 0.1
D_MODEL = 64
N_HEAD = 8
NUM_LAYERS = 4
DIM_FF = 128
DIM_OUT = 32


import torch
from torch import nn as nn
from torch.nn import functional as F
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


class MaskedContrastiveModel(nn.Module):
    def __init__(self, transformer_model, dim_ff, dim_out):
        super().__init__()
        self.transformer = transformer_model  # Your RoPeTimeSeriesTransformer
        self.projector = nn.Sequential(
            nn.Linear(transformer_model.config['d_model'], dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_out)
        )

    def forward(self, x_masked, x_masked_alt):
        # Encode both masked views
        emb1 = self.transformer(x_masked)    # shape: [B, T, input_dim]
        emb2 = self.transformer(x_masked_alt)

        # Pool embeddings, (mean over time)
        emb1 = emb1.mean(dim=1)  # [B, input_dim]
        emb2 = emb2.mean(dim=1)

        # Project to lower dim for contrastive loss
        z1 = self.projector(emb1)  # [B, dim_out]
        z2 = self.projector(emb2)

        return z1, z2


import time
from tqdm import tqdm
from src.save import save_model_checkpoint
from src.utils import adjust_time_series_size, stack_on_last_dim, z_score_normalize


def jitter(x, sigma=0.1):
    """Add Gaussian noise"""
    return x + sigma * torch.randn_like(x)

def zero_mask(x, mask_ratio=0.1):
    """Randomly zero out segments"""
    B, T, D = x.shape
    mask = torch.rand(B, T, 1, device=x.device) < mask_ratio
    return x.masked_fill(mask, 0.)

def create_views(x, augmentation_fns):
    """
    Given input x [B,T,D], create two augmented views
    """
    view1 = x.clone()
    view2 = x.clone()
    for fn in augmentation_fns:
        view1 = fn(view1)
        view2 = fn(view2)
    return view1, view2

def info_nce_loss(z1, z2, temperature=0.5):
    """Compute InfoNCE contrastive loss"""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.T) / temperature  # [B, B]
    targets = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, targets)
    return loss

def train_one_epoch_contrastive(
        model, dataloader, optimizer, 
        temperature=0.5,
        augmentations=(jitter, zero_mask),
        verbose=False
    ):
    device = next(model.parameters()).device
    model.train()

    running_loss = 0.0
    total = 0

    iterator = dataloader
    if verbose:
        iterator = tqdm(dataloader, desc="Contrastive Training", unit="batch")

    for batch_idx, (mic, acc, gyro, _) in enumerate(iterator):

        # prepare input [B,T,D]
        acc_adjusted = adjust_time_series_size(acc, mic.shape[1], 'resample')
        gyro_adjusted = adjust_time_series_size(gyro, mic.shape[1], 'resample')

         # Normalize input tensors
        mic_norm = z_score_normalize(mic)
        acc_norm = z_score_normalize(acc_adjusted)
        gyro_norm = z_score_normalize(gyro_adjusted)

        # Stack inputs along the feature dimension
        x = stack_on_last_dim(mic_norm, acc_norm, gyro_norm).to(device)

        # create two views
        v1, v2 = create_views(x, augmentations) 

        # encode
        z1, z2 = model(v1, v2)  # # [B, dim_out]

        # contrastive loss
        loss = info_nce_loss    (z1, z2, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += 1

    return running_loss / total

def train_model_contrastive(
        name, model, optimizer, train_loader,
        temperature=0.5,
        num_epochs=10,
        save_every=1, 
        verbose=False
    ):
    train_losses = []
    # no val loop for contrastive pretrain
    for epoch in range(1, num_epochs+1):
        start = time.time()
        loss = train_one_epoch_contrastive(model, train_loader, optimizer,
                                           temperature,
                                           verbose=verbose)
        train_losses.append(loss)
        print(f"Epoch {epoch}/{num_epochs} | Contrastive Loss: {loss:.4f} | Time: {time.time()-start:.1f}s")

        if epoch % save_every == 0:
            save_model_checkpoint(name + '_contrastive', model, None, optimizer,
                                  epoch, [loss], [], [])
    return model, train_losses


from models import RoPeTimeSeriesTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'input_dim': 7,
    'd_model': D_MODEL,
    'nhead': N_HEAD,
    'num_layers': NUM_LAYERS,
    'dim_feedforward': DIM_FF,
    'dropout': TF_DROPOUT,
}
transformer = RoPeTimeSeriesTransformer.from_config(config).to(device)
constrastive_model = MaskedContrastiveModel(transformer_model=transformer, dim_ff=DIM_FF, dim_out=32).to(device)
contrastive_optimizer = torch.optim.AdamW(constrastive_model.parameters())


try:
    train_model_contrastive(
        name='aerodactyl', 
        model=constrastive_model, 
        optimizer=contrastive_optimizer, 
        train_loader=train_loader,
        temperature=0.5,
        num_epochs=1, 
        save_every=1, 
        verbose=True
    )
except KeyboardInterrupt:
    print("Training interrupted by user.")


class PretrainedTransformerModel(nn.Module):
    def __init__(self, pretrained_transformer, dim_ff, dim_out):
        super().__init__()
        self.transformer = pretrained_transformer
        self.projection_head = nn.Sequential(
            nn.Linear(pretrained_transformer.config['d_model'], dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_out)
        )

    def forward(self, x):
        # Pass input through the transformer
        emb = self.transformer(x)  # shape: [B, T, d_model]

        # Pass through the projection head
        output = self.projection_head(emb)  # shape: [B, T, dim_out]

        return output

model = PretrainedTransformerModel(pretrained_transformer=transformer, dim_ff=DIM_FF, dim_out=7).to(device)
optimizer = torch.optim.AdamW(model.parameters())
criterion = torch.nn.MSELoss()


from torchinfo import summary
summary(model, input_size=(TRAIN_BATCH_SIZE, 1600, 7))


from src import train_model, evaluate

try:
    train_model(
        name='aerodactyl',
        model=model, 
        criterion=criterion, 
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


loss, auc = evaluate(model, test_loader, criterion, merge_strategy='stack', verbose=True)
print(f"Overall S+T | Loss: {loss:.4f}, AUC: {auc:.4f}")




