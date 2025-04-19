import pandas
import glob
import torch

class SensorDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob(folder_path + '/*.parquet')

    def __len__(self,):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        sample = pandas.read_parquet(file_path) # There is no way to avoid Pandas
        sample = sample.iloc[:, 1:]
        data = torch.tensor(sample.values, dtype=torch.float)
        return data.squeeze()