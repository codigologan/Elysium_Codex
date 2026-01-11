import torch
from src.data import make_synthetic_classification, make_loaders


def test_data_shapes():
    train, val = make_synthetic_classification(100, 20, 10)
    X_train, y_train = train
    X_val, y_val = val
    assert X_train.shape == (100, 10)
    assert y_train.shape == (100,)
    assert X_val.shape == (20, 10)
    assert y_val.shape == (20,)

    dl_train, dl_val = make_loaders(train, val, batch_size=16)
    xb, yb = next(iter(dl_train))
    assert xb.shape[1] == 10
    assert xb.shape[0] <= 16
