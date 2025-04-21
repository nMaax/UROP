def train(model, train_loader, criterion, optimizer, verbose=False):
    model.train()
    if verbose:
      print("Setted model in train mode")
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        if verbose and (batch_idx % 100 == 0):
          print(f'Processing batch {batch_idx} ({batch_idx/len(train_loader)*100:.2f}%)')

        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    return train_loss, train_accuracy