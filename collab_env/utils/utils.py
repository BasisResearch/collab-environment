import numpy as np
import torch

def get_backend(x):
    """Return numpy or torch module depending on input type."""
    if isinstance(x, torch.Tensor):
        return torch
    return np

def array_nan_equal(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.allclose(a[m], b[m])
