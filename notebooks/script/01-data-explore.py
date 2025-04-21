#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# import os
# os.chdir("..")  # Go up one level to the UROP directory
# print(os.getcwd())


import torch
from utils import SensorDataset
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

sensor = 'acc'
sr = config['sampling_rate'][sensor]


# Initialize Dataset
dataset = SensorDataset(root_dir="datasets/BrushlessMotor", split="train", sensors=[sensor])
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

print(f"Number of samples in dataset: {len(dataset)}")
print(dataset[0].shape)
print(dataset[5].shape)
print(dataset[6].shape)
print(dataset[7].shape)

print(f"Number of batches in dataloader: {len(dataloader)}")


for batch_idx, data in enumerate(dataloader):
    for sample in data:
        print(sample.shape)


random_index = torch.randint(0, 64, (1,)).item()
mel_spectrogram = librosa.feature.melspectrogram(y=data[random_index, :, 0].numpy(), sr=sr)
mel_log_spectrogram = librosa.power_to_db(mel_spectrogram)

plt.figure()
librosa.display.specshow(mel_log_spectrogram, x_axis='time', y_axis='mel', sr=sr, cmap='plasma')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.show()




