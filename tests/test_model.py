import torch
from src.model import build_mlp


def test_model_output_shape():
    model = build_mlp(8, 16, 3)
    x = torch.randn(4, 8)
    out = model(x)
    assert out.shape == (4, 3)
