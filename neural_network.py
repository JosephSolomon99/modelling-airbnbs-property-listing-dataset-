import pandas as pd
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, SubsetRandomSampler

dataset = AirbnbNightlyPriceImageDataset("airbnb_data.csv")
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(dataset_size * 0.8)

train_indices, val_indices = indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=64, shuffle=True, sampler=None)
