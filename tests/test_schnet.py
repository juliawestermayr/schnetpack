import pytest
import torch
from tests.fixtures import *

import schnetpack as spk
import numpy as np
from ase.neighborlist import neighbor_list

from schnetpack.representation.schnet_new import cfconv


@pytest.fixture
def schnet_old():
    return spk.SchNet(
        n_atom_basis=128, n_filters=128, n_interactions=3, cutoff=5.0, n_gaussians=20
    )


# def test_benchmark_schnet_old(benchmark, schnet_old, schnet_batch):
#     benchmark(schnet_old, schnet_batch)


@pytest.fixture
def indexed_data(example_data):
    Z = []
    R = []
    # C = []
    seg_m = []
    seg_i = []
    ind_j = []
    ind_S = []

    n_atoms = 0
    for i in range(len(example_data)):
        seg_m.append(n_atoms)
        atoms = example_data[i][0]
        atoms.set_pbc(False)
        Z.append(atoms.numbers)
        R.append(atoms.positions)
        # C.append(atoms.cell)
        idx_i, idx_j, idx_S = neighbor_list("ijS", atoms, 20.0, self_interaction=False)
        _, idx_i = np.unique(idx_i, return_counts=True)
        idx_i = np.cumsum(np.hstack((np.zeros((1,), dtype=np.int), idx_i)))
        seg_i.append(idx_i + n_atoms)
        ind_j.append(idx_j + n_atoms)
        ind_S.append(idx_S)
        n_atoms += len(atoms)
    seg_m.append(n_atoms)

    Z = np.hstack(Z)
    R = np.vstack(R)
    # C = np.array(C)
    seg_m = np.hstack(seg_m)
    seg_i = np.hstack(seg_i)
    ind_j = np.hstack(ind_j)
    ind_S = np.vstack(ind_S)

    inputs = {
        spk.Properties.Z: torch.tensor(Z).cuda(),
        spk.Properties.position: torch.tensor(R).cuda(),
        # spk.Properties.cell: C,
        "seg_m": torch.tensor(seg_m).cuda(),
        "seg_i": torch.tensor(seg_i).cuda(),
        "idx_j": torch.tensor(ind_j).cuda(),
        spk.Properties.cell_offset: torch.tensor(ind_S).cuda(),
    }

    return inputs


def test_benchmark_schnet_new(indexed_data):
    Z, R, seg_m, seg_i, ind_j, ind_S = (
        indexed_data[spk.Properties.Z],
        indexed_data[spk.Properties.R],
        indexed_data["seg_m"],
        indexed_data["seg_i"],
        indexed_data["idx_j"],
        indexed_data[spk.Properties.cell_offset],
    )
    print(Z.shape, R.shape, seg_m.shape, seg_i.shape, ind_j.shape, ind_S.shape)


def test_cfconv(indexed_data, benchmark):
    Z, R, seg_m, seg_i, idx_j, ind_S = (
        indexed_data[spk.Properties.Z],
        indexed_data[spk.Properties.R],
        indexed_data["seg_m"],
        indexed_data["seg_i"],
        indexed_data["idx_j"],
        indexed_data[spk.Properties.cell_offset],
    )

    benchmark(cfconv, R, R[idx_j], seg_i, idx_j)
