import pytest
import torch
import numpy as np

__all__ = [
    # input
    "schnet_batch",
    "max_atoms_in_batch",
    "neighbors",
    "neighbor_mask",
    "positions",
    "cell",
    "cell_offset",
    "r_ij",
    "f_ij",
    "random_input_dim",
    "random_output_dim",
    "random_shape",
    "random_float_input",
    "random_int_input",
    "rnd_natoms",
    "rnd_allatoms",
    "rnd_allnbh",
    "rnd_atomic_numbers",
    "rnd_nbh_idx",
    "rnd_idx_i",
    "rnd_idx_j",
    "rnd_atomic_environments",
    "rnd_atomic_environments_filter",
    "rnd_filters",
    "rnd_r_ij",
    "rnd_d_ij",
    "rnd_f_ij",
    "rnd_rcut_ij",
    # output
    "schnet_output_shape",
    "radial_basis_shape",
]


# inputs
# from data
@pytest.fixture
def schnet_batch(example_loader):
    return next(iter(example_loader))


# components of batch
@pytest.fixture
def max_atoms_in_batch(schnet_batch):
    return schnet_batch["_positions"].shape[1]


@pytest.fixture
def neighbors(schnet_batch):
    return schnet_batch["_neighbors"]


@pytest.fixture
def neighbor_mask(schnet_batch):
    return schnet_batch["_neighbor_mask"]


@pytest.fixture
def positions(schnet_batch):
    return schnet_batch["_positions"]


@pytest.fixture
def cell(schnet_batch):
    return schnet_batch["_cell"]


@pytest.fixture
def cell_offset(schnet_batch):
    return schnet_batch["_cell_offset"]


@pytest.fixture
def r_ij(atom_distances, positions, neighbors, cell, cell_offset, neighbor_mask):
    return atom_distances(positions, neighbors, cell, cell_offset, neighbor_mask)


@pytest.fixture
def f_ij(gaussion_smearing_layer, r_ij):
    return gaussion_smearing_layer(r_ij)


def max_atoms():
    return 10


@pytest.fixture
def rnd_natoms(batch_size, max_atoms):
    return torch.randint(2, max_atoms, size=(batch_size,))


@pytest.fixture
def rnd_nbh_idx(rnd_natoms):
    count = 0
    idx_i = []
    idx_j = []
    for natom in rnd_natoms:
        nnbh = torch.randint(1, natom, size=(natom,))
        idx_i.append(
            torch.repeat_interleave(torch.arange(0, natom) + count, nnbh, dim=0)
        )
        idx_j.append(torch.randint(count, natom + count, size=(int(torch.sum(nnbh)),)))
        count += natom

    idx_i = torch.hstack(idx_i)
    idx_j = torch.hstack(idx_j)
    return (idx_i, idx_j)


@pytest.fixture
def rnd_idx_i(rnd_nbh_idx):
    return rnd_nbh_idx[0]


@pytest.fixture
def rnd_idx_j(rnd_nbh_idx):
    return rnd_nbh_idx[1]


@pytest.fixture
def rnd_allatoms(rnd_natoms):
    return torch.sum(rnd_natoms)


@pytest.fixture
def rnd_atomic_numbers(rnd_allatoms, max_z):
    return torch.randint(1, max_z, size=(rnd_allatoms,))


@pytest.fixture
def rnd_allnbh(rnd_idx_i):
    return rnd_idx_i.shape[0]


@pytest.fixture
def rnd_atomic_environments(rnd_allatoms, n_atom_basis):
    return torch.rand((rnd_allatoms, n_atom_basis))


@pytest.fixture
def rnd_atomic_environments_filter(rnd_allatoms, n_filters):
    return torch.rand((rnd_allatoms, n_filters))


@pytest.fixture
def rnd_filters(rnd_allnbh, n_filters):
    return torch.rand((rnd_allnbh, n_filters))


@pytest.fixture
def rnd_r_ij(rnd_allnbh, cutoff):
    return torch.rand((rnd_allnbh, 3)) * cutoff


@pytest.fixture
def rnd_d_ij(rnd_r_ij):
    return torch.norm(rnd_r_ij, dim=1)


@pytest.fixture
def rnd_f_ij(radial_basis, rnd_d_ij):
    return radial_basis(rnd_d_ij)


@pytest.fixture
def rnd_rcut_ij(cutoff_fn, rnd_d_ij):
    return cutoff_fn(rnd_d_ij)


@pytest.fixture
def random_input_dim(random_shape):
    return random_shape[-1]


@pytest.fixture
def random_output_dim():
    return np.random.randint(1, 20, 1).item()


@pytest.fixture
def random_shape():
    return list(np.random.randint(1, 8, 3))


@pytest.fixture
def random_float_input(random_shape):
    return torch.rand(random_shape, dtype=torch.float32)


@pytest.fixture
def random_int_input(random_shape):
    return torch.randint(0, 20, random_shape)


# outputs
# spk.representation
@pytest.fixture
def schnet_output_shape(batch_size, max_atoms_in_batch, n_atom_basis):
    return [batch_size, max_atoms_in_batch, n_atom_basis]


@pytest.fixture
def interaction_output_shape(batch_size, max_atoms_in_batch, n_filters):
    return [batch_size, max_atoms_in_batch, n_filters]


# spk.nn
@pytest.fixture
def radial_basis_shape(rnd_allnbh, n_rbf):
    return [rnd_allnbh, n_rbf]
