import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset

class WindowedSensorDataset(Dataset):
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
        self.windows = []

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

        # Preload windows from all files
        for file_path in self.files:
            df = pd.read_parquet(file_path)
            data = df.iloc[:, 1:].values  # drop the first column (timestamp or index)
            num_windows = (len(data) - window_size) // stride + 1
            for i in range(num_windows):
                start = i * stride
                end = start + window_size
                window = data[start:end]
                self.windows.append(torch.tensor(window, dtype=torch.float32))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]

if __name__ == "__main__":
    dataset = WindowedSensorDataset(root_dir='datasets/BrushlessMotor', split='train', sensors=['mic'], window_size=100, stride=50)
    print(f"Number of windows: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample window shape: {sample.shape}")
