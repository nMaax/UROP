import os
import time
import warnings
import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def z_score_normalize(tensor):
    """Normalize a tensor using z-score normalization"""
    mean = tensor.mean(dim=1, keepdim=True)  # Tensor.shape = (Batch, Time, Channels)
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

def evaluate(model, dataloader, criterion, sensors_to_test=None):
    """Function to evaluate the model on a validation dataset"""
    device = next(model.parameters()).device  # Get the device of the model
    
    if sensors_to_test is None:
        sensors_to_test = ['mic', 'acc', 'gyro']  # Default to using all sensors

    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0  # Initialize validation loss
    segment_errors = defaultdict(list)  # Dictionary to store errors per segment
    segment_labels = {}  # Dictionary to store labels per segment

    with torch.no_grad():  # Disable gradient computation
        for mic, acc, gyro, labels in dataloader:
            # Mask sensors not requested for testing
            mic = mic if 'mic' in sensors_to_test else torch.zeros_like(mic)
            acc = acc if 'acc' in sensors_to_test else torch.zeros_like(acc)
            gyro = gyro if 'gyro' in sensors_to_test else torch.zeros_like(gyro)

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

def save_model_checkpoint(save_dir, name, model, config, optimizer, epoch, train_losses, val_losses, val_aucs):
    """Save the model checkpoint"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_class_name = model.__class__.__name__

    checkpoint = {
        'epoch': epoch,
        'model_class_name': model_class_name,
        'config': config,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_aucs': val_aucs,
    }

    checkpoint_path = os.path.join(save_dir, f"{name}_{model_class_name}_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_model_checkpoint(checkpoint_path, model_class, optimizer_class=None):
    """Load the model from a checkpoint file and return the model, optimizer, and training metrics"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Retrieve the model class name and config from checkpoint
    model_class_name = checkpoint['model_class_name']
    model_config = checkpoint['config']

    # Ensure that the checkpoint's model class matches the passed class name
    if model_class_name != model_class.__name__:
        raise ValueError(f"Checkpoint model class {model_class_name} does not match the provided model class {model_class.__name__}")

    # Recreate the model using the saved config
    model = model_class.from_config(model_config)

    # Load the model state dict (weights)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Ensure the optimizer class matches the expected class
    if optimizer_class and 'optimizer_state_dict' in checkpoint:
        optimizer = optimizer_class(model.parameters())  # Initialize optimizer
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = None  # If no optimizer state is saved, return None

    # Return model, optimizer, and saved training metrics
    return model, optimizer, checkpoint['epoch'], checkpoint['train_losses'], checkpoint['val_losses'], checkpoint['val_aucs']


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
        val_loss, val_auc = evaluate(model, val_loader, criterion)

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

        if (save_every and (epoch + 1) % save_every == 0) or (epoch == num_epochs - 1):
            # Save model checkpoint periodically with adjusted epoch
            save_model_checkpoint(save_dir, name, model, model_config, optimizer,
                                  adjusted_current_epoch + 1, train_losses, val_losses, val_aucs)

    return model, train_losses, val_losses, val_aucs  # Return trained model and metrics
