#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import os
os.chdir("..")  # Go up one level to the UROP directory


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


# Initialize Dataset
dataset_mic = SensorDataset(root_dir="datasets/BrushlessMotor", split="train", sensors=["mic"]) 
dataloader_mic = DataLoader(dataset_mic, batch_size=64, shuffle=True, num_workers=2)


from models import Autoencoder

AE = Autoencoder()




