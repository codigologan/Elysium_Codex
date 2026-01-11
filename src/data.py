from __future__ import annotations
import torch
from torch.utils.data import TensorDataset, DataLoader

def make_synthetic_classification(train_size: int, val_size: int, input_dim: int):
    X = torch.randn(train_size + val_size, input_dim)
    y = (X.sum(dim=1) > 0).long()
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    return (X_train, y_train), (X_val, y_val)

def make_loaders(train, val, batch_size: int):
    dl_train = DataLoader(TensorDataset(*train), batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(TensorDataset(*val), batch_size=batch_size, shuffle=False)
    return dl_train, dl_val
