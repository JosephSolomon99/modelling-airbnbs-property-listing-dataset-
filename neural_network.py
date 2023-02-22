import pandas as pd
import torch
from torch.utils.data import Dataset

class AirbnbNightlyPriceImageDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)