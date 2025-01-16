# src/data/dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np

class FakeDataset(Dataset):
    def __init__(self, data_path):
        data_dict = np.load(data_path, allow_pickle=True).item()
        self.X = data_dict['X']
        self.Y = data_dict['Y']
        
        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            x = torch.tensor(self.X[idx], dtype=torch.float32)
            y = torch.tensor(self.Y[idx], dtype=torch.long)
            
            return x, y