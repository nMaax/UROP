import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import warnings

def z_score_normalize(tensor):
    mean = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True) + 1e-8
    return (tensor - mean) / std

def flatten_and_concat(*tensors):
    return torch.cat([t.view(t.size(0), -1) for t in tensors], dim=1)

def train_one_epoch(model, dataloader, optimizer, criterion, verbose=False):
    device = next(model.parameters()).device
    
    model.train()
    running_loss = 0.0

    for batch_idx, (mic, acc, gyro, labels) in enumerate(dataloader):
        mic_norm = z_score_normalize(mic)
        acc_norm = z_score_normalize(acc)
        gyro_norm = z_score_normalize(gyro)

        inputs = flatten_and_concat(mic_norm, acc_norm, gyro_norm).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if verbose:
            print(f"Batch {batch_idx + 1}/{len(dataloader)} | Loss: {loss.item():.6f}")

    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    device = next(model.parameters()).device
    
    model.eval()
    val_loss = 0.0
    segment_errors = defaultdict(list)
    segment_labels = {}

    with torch.no_grad():
        for mic, acc, gyro, labels in dataloader:
            mic_norm = z_score_normalize(mic)
            acc_norm = z_score_normalize(acc)
            gyro_norm = z_score_normalize(gyro)

            inputs = flatten_and_concat(mic_norm, acc_norm, gyro_norm).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item()

            batch_errors = ((inputs - outputs) ** 2).mean(dim=1).cpu().numpy()
            batch_segments = labels['segment_id'] if isinstance(labels, dict) else [l['segment_id'] for l in labels]
            batch_labels = labels['anomaly_label'] if isinstance(labels, dict) else [l['anomaly_label'] for l in labels]

            if not isinstance(batch_segments, list):
                batch_segments = [batch_segments]
                batch_labels = [batch_labels]

            for seg_id, err, lbl in zip(batch_segments, batch_errors, batch_labels):
                segment_errors[seg_id].append(err)
                segment_labels[seg_id] = lbl

    val_loss /= len(dataloader)

    # Compute median error per segment
    med_errors = [torch.median(torch.tensor(errors)).item() for errors in segment_errors.values()]
    med_labels = [segment_labels[seg_id] for seg_id in segment_errors.keys()]
    #print(set(med_labels))
    
    # Handle edge case where only one class is present
    if len(set(med_labels)) < 2:
        #warnings.warn("Only one class present in y_true. ROC AUC score is undefined.")
        val_auc = float('nan')
    else:
        val_auc = roc_auc_score(
            [0 if lbl == 'normal' else 1 for lbl in med_labels], 
            med_errors
        )
    
    return val_loss, val_auc

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, verbose=True):

    train_losses, val_losses, val_aucs = [], [], []

    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, verbose)
        val_loss, val_auc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.4f}")

    return model, train_losses, val_losses, val_aucs