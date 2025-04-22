#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import os
os.chdir("..")  # Go up one level to the UROP directory
print(os.getcwd())


import torch
from utils import SensorDataset, LazyWindowedMultiSensorDataset
from torch.utils.data import DataLoader
import yaml

import librosa
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

with open("config.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


# Initialize Dataset
dataset = LazyWindowedMultiSensorDataset(root_dir="datasets/BrushlessMotor", split="train", metadata_file="attributes_normal_source_train.csv")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

print(f"Number of samples in dataset: {len(dataset)}")
print(f"Number of batches in dataloader: {len(dataloader)}")


for batch_idx, data in enumerate(dataloader):
    mic, acc, gyro, labels = data
    print(f"Mic shape: {mic.shape}")
    print(f"Acc shape: {acc.shape}")
    print(f"Gyro shape: {gyro.shape}")
    for key, value in labels.items():
        print(f"Length of {key}: {len(value)}")
    if batch_idx == 3:
        break
    else:
        print()


random_index = torch.randint(0, 64, (1,)).item()
mel_spectrogram = librosa.feature.melspectrogram(y=mic[random_index, :, 0].numpy(), sr=16000, n_fft=512, hop_length=256, n_mels=64)
mel_log_spectrogram = librosa.power_to_db(mel_spectrogram)

plt.figure()
librosa.display.specshow(mel_log_spectrogram, x_axis='time', y_axis='mel', sr=16000, cmap='plasma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.show()




