import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset

class SensorDataset(Dataset):
    def __init__(self, root_dir, split='train', sensors=None):
        """
        Args:
            root_dir (str): path to the dataset folder, e.g. 'datasets/BrushlessMotor'
            split (str): 'train' or 'test'
            sensors (list of str or None): list of sensor keywords to include
                e.g. ['mic'], ['gyro','acc'], or None for all.
        """
        self.sensors = [s.lower() for s in sensors] if sensors else []
        data_path = os.path.join(root_dir, split)
        
        # find all parquet files
        all_files = glob.glob(os.path.join(data_path, '*.parquet'))
        
        # filter by sensor keywords if requested
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # load parquet via pandas
        df = pd.read_parquet(file_path)
        # drop the first column (timestamp or index)
        data = df.iloc[:, 1:].values  
        # convert to torch.Tensor
        return torch.tensor(data, dtype=torch.float32)

if __name__ == "__main__":
    # Example usage
    dataset = SensorDataset(root_dir='datasets/BrushlessMotor', split='train', sensors=['mic'])
    print(f"Number of files: {len(dataset)}")
    sample_data = dataset[0]
    print(f"Sample data shape: {sample_data.shape}")