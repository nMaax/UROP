import torch
from scipy.signal import resample_poly


def minmax_normalize(tensor, min_val=0.0, max_val=1.0):
    """Normalize a tensor using min-max normalization"""
    tensor_min = tensor.min(dim=1, keepdim=True)[0]  # Tensor.shape = (Batch, Time, Channels)
    tensor_max = tensor.max(dim=1, keepdim=True)[0]
    # Avoid division by zero
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
    return normalized_tensor * (max_val - min_val) + min_val

def z_score_normalize(tensor):
    """Normalize a tensor using z-score normalization"""
    mean = tensor.mean(dim=1, keepdim=True)  # Tensor.shape = (Batch, Time, Channels)
    std = tensor.std(dim=1, keepdim=True) + 1e-8  # Add small value to avoid division by zero
    return (tensor - mean) / std

def flatten_and_concat(*tensors):
    """Flatten tensors and concatenate them along the feature dimension"""
    return torch.cat([t.view(t.size(0), -1) for t in tensors], dim=1).unsqueeze(-1) # Will return (batch, time), without dim

def stack_on_last_dim(*tensors):
    """Concat tensors along the last dimension (feature dimension)"""
    return torch.cat(tensors, dim=-1)

def adjust_time_series_size(tensor, target_length, mode='zero', up=1600, down=670):
    """
    Adjust the size of a time series tensor to a target length.
    mode (str): Padding mode. Options are:
        - 'zero': Pad with zeros.
        - 'repeat_start': Pad by repeating the first entry.
        - 'repeat_end': Pad by repeating the last entry.
        - 'truncate': Truncate the time dimension if it exceeds the target length.
    """
    current_length = tensor.size(1)

    if current_length == target_length:
        return tensor  # No adjustment needed

    if current_length < target_length:
        # Padding is required
        pad_size = target_length - current_length
        if mode == 'zero':
            padding = torch.zeros(tensor.size(0), pad_size, tensor.size(2), device=tensor.device)
        elif mode == 'repeat_start':
            padding = tensor[:, :1, :].repeat(1, pad_size, 1)
        elif mode == 'repeat_end':
            padding = tensor[:, -1:, :].repeat(1, pad_size, 1)
        elif mode == 'balanced':
            indices = torch.linspace(0, current_length - 1, steps=target_length, device=tensor.device).long()
            tensor = tensor[:, indices, :]
            return tensor
        elif mode == 'resample':
            device = tensor.device
            signal = tensor.cpu().numpy()
            upsampled_signal = resample_poly(signal, up=1600, down=670, axis=1)
            tensor = torch.tensor(upsampled_signal, device=device, dtype=tensor.dtype)
            return tensor
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")
        return torch.cat([padding, tensor], dim=1) if mode == 'repeat_start' else torch.cat([tensor, padding], dim=1)

    elif current_length > target_length:
        # Truncation is required
        if mode == 'truncate':
            return tensor[:, :target_length, :]
        else:
            raise ValueError(f"Truncation required but unsupported mode: {mode}")

    raise ValueError(f"Invalid mode: {mode}")
