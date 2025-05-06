import torch
import os

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
