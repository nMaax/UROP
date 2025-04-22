import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import warnings

def z_score_normalize(tensor):
    """Normalize a tensor using z-score normalization"""
    mean = tensor.mean(dim=1, keepdim=True) # Tensor.shape = (Batch, Time, Channels)
    std = tensor.std(dim=1, keepdim=True) + 1e-8  # Add small value to avoid division by zero
    return (tensor - mean) / std

def flatten_and_concat(*tensors):
    """Flatten tensors and concatenate them along the feature dimension"""
    return torch.cat([t.view(t.size(0), -1) for t in tensors], dim=1)

def train_one_epoch(model, dataloader, optimizer, criterion, verbose=False):
    """Train the model for one epoch"""
    device = next(model.parameters()).device  # Get the device of the model
    
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss

    for batch_idx, (mic, acc, gyro, labels) in enumerate(dataloader):
        # Normalize input tensors
        mic_norm = z_score_normalize(mic)
        acc_norm = z_score_normalize(acc)
        gyro_norm = z_score_normalize(gyro)

        # Flatten and concatenate inputs
        inputs = flatten_and_concat(mic_norm, acc_norm, gyro_norm).to(device)

        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, inputs)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss

        if verbose:
            print(f"Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.6f}")

    return running_loss / len(dataloader)  # Return average loss for the epoch

def evaluate(model, dataloader, criterion):
    """Function to evaluate the model on a validation dataset"""
    device = next(model.parameters()).device  # Get the device of the model
    
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0  # Initialize validation loss
    segment_errors = defaultdict(list)  # Dictionary to store errors per segment
    segment_labels = {}  # Dictionary to store labels per segment

    with torch.no_grad():  # Disable gradient computation
        for mic, acc, gyro, labels in dataloader:
            # Normalize input tensors
            mic_norm = z_score_normalize(mic)
            acc_norm = z_score_normalize(acc)
            gyro_norm = z_score_normalize(gyro)

            # Flatten and concatenate inputs
            inputs = flatten_and_concat(mic_norm, acc_norm, gyro_norm).to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, inputs)  # Compute loss
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

    val_loss /= len(dataloader)  # Compute average validation loss

    # Compute median error per segment
    med_errors = [torch.median(torch.tensor(errors)).item() for errors in segment_errors.values()]
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

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, verbose=True):
    """Function to train the model for multiple epochs"""
    train_losses, val_losses, val_aucs = [], [], []  # Initialize lists to store metrics

    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train for one epoch and evaluate on validation set
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, verbose)
        val_loss, val_auc = evaluate(model, val_loader, criterion)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        # Print metrics for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.4f}")

    return model, train_losses, val_losses, val_aucs  # Return trained model and metrics