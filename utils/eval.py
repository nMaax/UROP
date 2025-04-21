import torch

# Validation loop
def validate(model, val_loader, criterion, verbose=False):
    if verbose:
      print(f"Started validation")
    model.eval()
    if verbose:
      print("Setted model in eval mode")
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):

            if verbose and (batch_idx % 100 == 0):
              print(f'Processing batch {batch_idx}')

            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    return val_loss, val_accuracy