import os
import time
import warnings
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from src.utils import minmax_normalize, stack_on_last_dim, z_score_normalize, flatten_and_concat, adjust_time_series_size
from src.save import save_model_checkpoint

def train_one_epoch(model, dataloader, optimizer, criterion, merge_strategy='default', num_batches=None, verbose=False):
    """Train the model for one epoch"""
    device = next(model.parameters()).device  # Get the device of the model
    
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss

    len_dataloader = len(dataloader)
    if num_batches:
        indices = np.random.choice(len_dataloader, num_batches, replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=sampler)
    else:
        num_batches = len_dataloader

    iter = dataloader
    if verbose == 1:
        iter = tqdm(dataloader, desc="Training", unit="batch") if verbose else dataloader

    for batch_idx, (mic, acc, gyro, _) in enumerate(iter):
        if merge_strategy in {'default', 'flatten'}:
            # Normalize input tensors
            mic_norm = z_score_normalize(mic)
            acc_norm = z_score_normalize(acc)
            gyro_norm = z_score_normalize(gyro)

            # Flatten and concatenate inputs
            inputs = flatten_and_concat(mic_norm, acc_norm, gyro_norm).to(device)
        elif merge_strategy == 'stack':
            # Adjust time series size for acc and gyro to match mic
            acc_adjusted = adjust_time_series_size(acc, mic.shape[1], 'resample')
            gyro_adjusted = adjust_time_series_size(gyro, mic.shape[1], 'resample')
            
            # Normalize input tensors
            mic_norm = z_score_normalize(mic)
            acc_norm = z_score_normalize(acc_adjusted)
            gyro_norm = z_score_normalize(gyro_adjusted)

            # Stack inputs along the feature dimension
            inputs = stack_on_last_dim(mic_norm, acc_norm, gyro_norm).to(device)
        else:
            raise ValueError(f"Unknown merge_strategy: {merge_strategy}")

        if verbose >= 3:
            print(f"Input to the model of shape: {inputs.shape}")
        
        outputs = model(inputs)  # Forward pass
        
        if verbose >= 3:
            print(f"Output from the model of shape: {outputs.shape}")

        optimizer.zero_grad()  # Reset gradients
        loss = criterion(inputs, outputs)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss

        if verbose >=2 and batch_idx % (10 if verbose == 2 else 1):
            print(f"[Batch {batch_idx + 1}/{len_dataloader}] Loss: {loss.item():.6f}")

        if batch_idx >= num_batches:
            break

    return running_loss / num_batches  # Return average loss for the epoch

def evaluate(model, dataloader, criterion, sensors_to_test=None, merge_strategy='default', num_batches=None, verbose=False):
    """Function to evaluate the model on a validation dataset"""
    device = next(model.parameters()).device  # Get the device of the model
    
    if sensors_to_test is None:
        sensors_to_test = ['mic', 'acc', 'gyro']  # Default to using all sensors

    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0  # Initialize validation loss
    segment_errors = defaultdict(list)  # Dictionary to store errors per segment
    segment_labels = {}  # Dictionary to store labels per segment

    # Wrap the dataloader with tqdm if verbose is enabled
    len_dataloader = len(dataloader)
    if num_batches:
        indices = np.random.choice(len_dataloader, num_batches, replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, sampler=sampler)
    else:
        num_batches = len_dataloader
    
    iter = dataloader
    if verbose == 1:
        iter = tqdm(dataloader, desc="Evaluation", unit="batch") if verbose else dataloader

    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (mic, acc, gyro, labels) in enumerate(iter):
            # Mask sensors not requested for testing
            mic = mic if 'mic' in sensors_to_test else torch.zeros_like(mic)
            acc = acc if 'acc' in sensors_to_test else torch.zeros_like(acc)
            gyro = gyro if 'gyro' in sensors_to_test else torch.zeros_like(gyro)

            if merge_strategy in {'default', 'flatten'}:
                # Normalize input tensors
                mic_norm = z_score_normalize(mic)
                acc_norm = z_score_normalize(acc)
                gyro_norm = z_score_normalize(gyro)

                # Flatten and concatenate inputs
                inputs = flatten_and_concat(mic_norm, acc_norm, gyro_norm).to(device)
            elif merge_strategy == 'stack':
                # Normalize input tensors
                mic_norm = z_score_normalize(mic)
                acc_norm = z_score_normalize(acc)
                gyro_norm = z_score_normalize(gyro)

                # Adjust time series size for acc and gyro to match mic
                acc_adjusted = adjust_time_series_size(acc_norm, mic_norm.shape[1], 'balanced')
                gyro_adjusted = adjust_time_series_size(gyro_norm, mic_norm.shape[1], 'balanced')

                # Stack inputs along the feature dimension
                inputs = torch.cat([mic_norm, acc_adjusted, gyro_adjusted], dim=2).to(device)
            else:
                raise ValueError(f"Unknown merge_strategy: {merge_strategy}")

            outputs = model(inputs)  # Forward pass
            loss = criterion(inputs, outputs)  # Compute loss
            val_loss += loss.item()  # Accumulate validation loss

            # Compute reconstruction errors for each sample
            batch_errors = ((inputs - outputs) ** 2).mean(dim=1).cpu().numpy()
            # Extract segment IDs and anomaly labels from labels
            batch_segments = labels['segment_id'] if isinstance(labels, dict) else [l['segment_id'] for l in labels]
            batch_labels = labels['anomaly_label'] if isinstance(labels, dict) else [l['anomaly_label'] for l in labels]

            # Ensure batch_segments and batch_labels are lists
            if not isinstance(batch_segments, list):
                batch_segments = [batch_segments]
                batch_labels = [batch_labels]

            # Store errors and labels for each segment
            for seg_id, err, lbl in zip(batch_segments, batch_errors, batch_labels):
                segment_errors[seg_id].append(err)
                segment_labels[seg_id] = lbl

            if verbose >=2 and batch_idx % (10 if verbose == 2 else 1):
                median_error = np.median(batch_errors)  # Aggregate errors using median
                average_error = np.mean(batch_errors)  # Aggregate errors using mean
                print(f"[Batch {batch_idx+1}/{len_dataloader}] Validation | Loss: {loss.item():.6f} | Median Batch Error: {median_error:.6f} | Average Batch Error: {average_error:.6f}")

    val_loss /= len(dataloader)  # Compute average validation loss

    # Compute median error per segment
    med_errors = [torch.median(torch.tensor(np.array(errors))).item() for errors in segment_errors.values()]
    med_labels = [segment_labels[seg_id] for seg_id in segment_errors.keys()]
    
    # Handle edge case where only one class is present
    if len(set(med_labels)) < 2:
        val_auc = float('nan')  # ROC AUC is undefined for a single class
    else:
        # Compute ROC AUC score
        val_auc = roc_auc_score(
            [0 if lbl == 'normal' else 1 for lbl in med_labels], 
            med_errors
        )
    
    return val_loss, val_auc  # Return validation loss and AUC

def train_model(name, model, criterion, optimizer, train_loader, val_loader, merge_startegy='default',
                start_epoch=0, num_epochs=10, save_every=1, 
                save_dir='checkpoints', verbose=True, train_num_batches=None, val_num_batches=None):
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
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, merge_startegy, train_num_batches, verbose)
        if val_loader:
            val_loss, val_auc = evaluate(model, val_loader, criterion, ['mic', 'acc', 'gyro'], merge_startegy, val_num_batches, verbose)
        else:
            val_loss, val_auc = np.nan, np.nan

        # End timing the epoch
        epoch_time = time.time() - start_time
        eta = epoch_time * (num_epochs - epoch - 1)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        # Print metrics for the current epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] (Checkpoint Epoch: {adjusted_current_epoch + 1}) | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.4f}")
        print(f"Time Spent: {epoch_time:.2f}s | ETA: {eta:.2f}s | Current Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        if save_every and (((epoch + 1) % save_every == 0) or (epoch == (num_epochs - 1))):
            # Save model checkpoint periodically with adjusted epoch
            save_model_checkpoint(save_dir, name, model, model_config, optimizer,
                                  adjusted_current_epoch + 1, train_losses, val_losses, val_aucs)

    return model, train_losses, val_losses, val_aucs  # Return trained model and metrics
