import torch
from torch.utils.data import DataLoader
from .dataset import MyDataset

def create_dataloader(data_path, batch_size, shuffle=True, num_workers=4):
    """
    DataLoader 생성 함수.
    """
    dataset = MyDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader