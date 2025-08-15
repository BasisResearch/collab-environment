import numpy as np

def array_nan_equal(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.allclose(a[m], b[m])