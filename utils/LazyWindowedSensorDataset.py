import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset

class LazyWindowedSensorDataset(Dataset):
    def __init__(self, root_dir, split='train', sensors=None, window_size=100, stride=50):
        """
        Args:
            root_dir (str): path to the dataset folder, e.g. 'datasets/BrushlessMotor'
            split (str): 'train' or 'test'
            sensors (list of str or None): list of sensor keywords to include
            window_size (int): length of each window (number of time steps)
            stride (int): stride between windows
        """
        self.window_size = window_size
        self.stride = stride
        self.window_index = []  # List of (file_path, start_idx)

        self.sensors = [s.lower() for s in sensors] if sensors else []
        data_path = os.path.join(root_dir, split)
        all_files = glob.glob(os.path.join(data_path, '*.parquet'))

        if self.sensors:
            self.files = [
                f for f in all_files
                if any(sensor in os.path.basename(f).lower() for sensor in self.sensors)
            ]
        else:
            self.files = all_files

        self.files.sort()
        if not self.files:
            raise RuntimeError(f"No .parquet files found in {data_path} for sensors={self.sensors}")

        # Precompute window positions for each file
        self.file_lengths = {}
        for file_path in self.files:
            df = pd.read_parquet(file_path)
            data_length = len(df) - window_size
            if data_length >= 0:
                self.file_lengths[file_path] = data_length
                for start in range(0, data_length + 1, stride):
                    self.window_index.append((file_path, start))

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        file_path, start_idx = self.window_index[idx]
        df = pd.read_parquet(file_path)
        data = df.iloc[:, 1:].values  # drop timestamp or index column
        window = data[start_idx:start_idx + self.window_size]
        return torch.tensor(window, dtype=torch.float32)

if __name__ == "__main__":
    dataset = LazyWindowedSensorDataset(root_dir='datasets/BrushlessMotor', split='train', sensors=['mic'], window_size=100, stride=50)
    print(f"Number of windows: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample window shape: {sample.shape}")
