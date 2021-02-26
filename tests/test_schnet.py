import pytest
from tests.fixtures import *

import schnetpack as spk
import numpy as np
from ase.neighborlist import neighbor_list


@pytest.fixture
def schnet_old():
    return spk.SchNet(
        n_atom_basis=128, n_filters=128, n_interactions=3, cutoff=5.0, n_gaussians=20
    )


def test_benchmark_schnet_old(benchmark, schnet_old, schnet_batch):
    benchmark(schnet_old, schnet_batch)


@pytest.fixture
def indexed_data(example_data):
    Z = []
    R = []
    #C = []
    ind_i = []
    ind_j = []
    ind_S = []

    n_atoms = 0
    for i in range(len(example_data)):
        atoms = example_data[i][0]
        atoms.set_pbc(False)
        Z.append(atoms.numbers)
        R.append(atoms.positions)
        #C.append(atoms.cell)
        idx_i, idx_j, idx_S = neighbor_list("ijS", atoms, 20.0, self_interaction=False)
        ind_i.append(idx_i + n_atoms)
        ind_j.append(idx_j + n_atoms)
        ind_S.append(idx_S)
        n_atoms += len(atoms)

    Z = np.hstack(Z)
    R = np.vstack(R)
    #C = np.array(C)
    ind_i = np.hstack(ind_i)
    ind_j = np.hstack(ind_j)
    ind_S = np.vstack(ind_S)

    inputs = {
        spk.Properties.Z: Z,
        spk.Properties.position: R,
        #spk.Properties.cell: C,
        "ind_i": ind_i,
        "ind_j": ind_j,
        spk.Properties.cell_offset: ind_S,
    }

    return inputs


def test_benchmark_schnet_new(indexed_data):
    Z, R, ind_i, ind_j, ind_S = (
        indexed_data[spk.Properties.Z],
        indexed_data[spk.Properties.R],
        indexed_data["ind_i"],
        indexed_data["ind_j"],
        indexed_data[spk.Properties.cell_offset],
    )
    print(Z.shape, R.shape, ind_i.shape, ind_j.shape, ind_S.shape)
    print(R[ind_i].shape, R[ind_j].shape)
