import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from functools import lru_cache
import logging

# Setup logger for logging warnings and debug information
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_test_split(dataset, val_ratio=0.2):
    total_len = len(dataset)
    val_len = int(val_ratio * total_len)
    train_len = total_len - val_len

    train_dataset, val_dataset = random_split(
        dataset, [train_len, val_len]
    )

    return train_dataset, val_dataset

class LazyWindowedDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split,
        anomaly_type=None,
        domain_type=None,
        window_size_ms=100,
        stride_ms=50
    ):
        # Set default values for anomaly_type and domain_type if not provided
        self.anomaly_types = anomaly_type if anomaly_type else ['normal', 'anomaly']
        self.domain_types = domain_type if domain_type else ['source', 'target']

        # Store window size and stride in milliseconds
        self.window_size_ms = window_size_ms
        self.stride_ms = stride_ms

        # Define sampling rates for each sensor
        self.sample_rates = {
            'mic': 16000,
            'acc': 6700,
            'gyro': 6700,
        }

        # Convert window size and stride from milliseconds to number of samples for each sensor
        self.window_samples = {
            s: int(self.window_size_ms * rate / 1000)
            for s, rate in self.sample_rates.items()
        }
        self.stride_samples = {
            s: int(self.stride_ms * rate / 1000)
            for s, rate in self.sample_rates.items()
        }

        # List of (entry index, window start time in ms)
        self.window_index = []

        # Load metadata and parquet files
        self.data_path = os.path.join(root_dir, split)
        meta_path_format = "attributes_{anomaly}_{domain}_{split}.csv"
        meta_dfs = []
        for anomaly in self.anomaly_types:
            for domain in self.domain_types:
                metadata_file = meta_path_format.format(anomaly=anomaly, domain=domain, split=split)
                meta_path = os.path.join(self.data_path, metadata_file)
                if os.path.isfile(meta_path):
                    meta_dfs.append(pd.read_csv(meta_path))
                else:
                    logger.warning(f"Metadata file not found: {meta_path}")
        
        if not meta_dfs:
            raise RuntimeError(f"No metadata files found for the specified anomaly and domain types in {self.data_path}")
        
        # Concatenate all metadata DataFrames into one
        self.meta_df = pd.concat(meta_dfs, ignore_index=True)
        self.entries = []

        # For each row in metadata, map sensor files and attach labels
        for _, row in self.meta_df.iterrows():
            mic_fp  = os.path.join(self.data_path, row['imp23absu_mic'])
            acc_fp  = os.path.join(self.data_path, row['ism330dhcx_acc'])
            gyro_fp = os.path.join(self.data_path, row['ism330dhcx_gyro'])

            # Skip entries with missing files and log warning
            if not (os.path.isfile(mic_fp) and os.path.isfile(acc_fp) and os.path.isfile(gyro_fp)):
                logger.warning(f"Skipping row with missing files: mic={mic_fp}, acc={acc_fp}, gyro={gyro_fp}")
                continue

            labels = {
                'segment_id':        row['segment_id'],
                'split_label':       row['split_label'],
                'anomaly_label':     row['anomaly_label'],
                'domain_shift_op':   row['domain_shift_op'],
                'domain_shift_env':  row['domain_shift_env'],
            }
            self.entries.append({
                'mic':   mic_fp,
                'acc':   acc_fp,
                'gyro':  gyro_fp,
                'labels': labels,
            })

        if not self.entries:
            raise RuntimeError(f"No valid entries found in metadata: {meta_path}")

        # Precompute the list of valid (entry_idx, start_time) pairs for windowing
        self._compute_window_index()

    @lru_cache(maxsize=128)
    def _load_parquet(self, filepath):
        # Cache loading parquet files to minimize I/O
        return pd.read_parquet(filepath)

    def _compute_window_index(self):
        # For each entry, calculate how many full windows fit
        for entry_idx, entry in enumerate(self.entries):
            len_m = len(self._load_parquet(entry['mic']))
            len_a = len(self._load_parquet(entry['acc']))
            len_g = len(self._load_parquet(entry['gyro']))

            # Calculate duration in milliseconds for each sensor file
            duration_ms_m = len_m / self.sample_rates['mic'] * 1000
            duration_ms_a = len_a / self.sample_rates['acc'] * 1000
            duration_ms_g = len_g / self.sample_rates['gyro'] * 1000

            # Minimum shared duration across all sensors
            min_duration_ms = min(duration_ms_m, duration_ms_a, duration_ms_g)
            max_start_ms = min_duration_ms - self.window_size_ms

            # Skip entries too short to accommodate one full window
            if max_start_ms < 0:
                logger.warning(
                    f"Skipping entry {entry['labels']['segment_id']} due to insufficient duration (ms) for windows size {self.window_size_ms} (ms): "
                    f"mic={duration_ms_m:.1f}, acc={duration_ms_a:.1f}, gyro={duration_ms_g:.1f}"
                )
                continue

            # Compute window start times with stride
            start_ms = 0
            while start_ms <= max_start_ms:
                self.window_index.append((entry_idx, start_ms))
                start_ms += self.stride_ms

    def __len__(self):
        return len(self.window_index)

    def _pad_to_length(self, array, length):
        # Pad array with zeros if it's shorter than the required window length
        if len(array) >= length:
            return array[:length]
        pad_len = length - len(array)
        pad_shape = (pad_len, array.shape[1]) if len(array.shape) > 1 else (pad_len,)
        pad_array = torch.zeros(pad_shape, dtype=torch.float32)
        return torch.cat([array, pad_array], dim=0)

    def __getitem__(self, idx):
        # Retrieve entry and window start time
        entry_idx, start_ms = self.window_index[idx]
        entry = self.entries[entry_idx]

        # Load sensor data and remove timestamp column (assumed to be first)
        df_m = self._load_parquet(entry['mic']).iloc[:, 1:].values
        df_a = self._load_parquet(entry['acc']).iloc[:, 1:].values
        df_g = self._load_parquet(entry['gyro']).iloc[:, 1:].values

        # Compute sample indices from milliseconds
        start_m = int(start_ms * self.sample_rates['mic'] / 1000)
        start_a = int(start_ms * self.sample_rates['acc'] / 1000)
        start_g = int(start_ms * self.sample_rates['gyro'] / 1000)

        w_m = self.window_samples['mic']
        w_a = self.window_samples['acc']
        w_g = self.window_samples['gyro']

        # Extract window of samples and convert to tensors
        mic_window  = torch.tensor(df_m[start_m:start_m + w_m], dtype=torch.float32)
        acc_window  = torch.tensor(df_a[start_a:start_a + w_a], dtype=torch.float32)
        gyro_window = torch.tensor(df_g[start_g:start_g + w_g], dtype=torch.float32)

        # Pad windows if they are shorter than expected
        mic_window  = self._pad_to_length(mic_window, w_m)
        acc_window  = self._pad_to_length(acc_window, w_a)
        gyro_window = self._pad_to_length(gyro_window, w_g)

        return mic_window, acc_window, gyro_window, entry['labels']
