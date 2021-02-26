import torch
from torch import nn
from torch_scatter import segment_csr


from schnetpack.nn import Dense


def cfconv(x, Wij, seg_i, idx_j, reduce="sum"):
    """
    Continuous-filter convolution.

    Args:
        x: input values
        Wij: filter
        seg_i: segments of neighbors belonging to i
        idx_j: index of neighbors j
        reduce: reduction method (sum, mean, ...)

    Returns:
        convolved inputs

    """
    x_ij = x[idx_j] * Wij
    y = segment_csr(x_ij, seg_i, reduce=reduce)
    return y
