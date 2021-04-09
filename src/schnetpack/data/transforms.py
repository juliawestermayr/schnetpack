from typing import Dict

import torch
import torch.nn as nn
from schnetpack import Structure

from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import atomic_masses

from numba import jit
import numpy as np

__all__ = [
    "ASENeighborList",
    "TorchNeighborList",
    "NumbaNeighborList",
    "CastMap",
    "CastTo32",
    "SubtractCenterOfMass",
    "SubtractCenterOfGeometry",
]


## neighbor lists


class ASENeighborList(nn.Module):
    """
    Calculate neighbor list using ASE.

    Note: This is quite slow and should only used as a baseline for faster implementations!
    """

    def __init__(self, cutoff):
        """
        Args:
            cutoff: cutoff radius for neighbor search
        """
        super().__init__()
        self.cutoff = cutoff

    def forward(self, inputs):
        Z = inputs[Structure.Z]
        R = inputs[Structure.R]
        cell = inputs[Structure.cell]
        pbc = inputs[Structure.pbc]
        at = Atoms(numbers=Z, positions=R, cell=cell, pbc=pbc)
        idx_i, idx_j, idx_S, Rij = neighbor_list(
            "ijSD", at, self.cutoff, self_interaction=False
        )
        inputs[Structure.idx_i] = torch.tensor(idx_i)
        inputs[Structure.idx_j] = torch.tensor(idx_j)
        inputs[Structure.Rij] = torch.tensor(Rij)
        inputs[Structure.cell_offset] = torch.tensor(idx_S)
        return inputs


class TorchNeighborList(nn.Module):
    """
    Environment provider making use of neighbor lists as implemented in TorchAni
    (https://github.com/aiqm/torchani/blob/master/torchani/aev.py).
    Supports cutoffs and PBCs and can be performed on either CPU or GPU.

    Args:
        cutoff: cutoff radius for neighbor search
    """

    def __init__(self, cutoff):
        super(TorchNeighborList, self).__init__()
        self.cutoff = cutoff

    def forward(self, inputs):
        positions = inputs[Structure.R]
        pbc = inputs[Structure.pbc]
        cell = inputs[Structure.cell]

        # Check if shifts are needed for periodic boundary conditions
        if torch.all(pbc == 0):
            shifts = torch.zeros(0, 3, device=cell.device).long()
        else:
            shifts = self._get_shifts(cell, pbc)

        idx_i, idx_j, idx_S, Rij = self._get_neighbor_pairs(positions, cell, shifts)

        # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
        bi_idx_i = torch.cat((idx_i, idx_j), dim=0)
        bi_idx_j = torch.cat((idx_j, idx_i), dim=0)
        bi_idx_S = torch.cat((-idx_S, idx_S), dim=0)
        bi_Rij = torch.cat((-Rij, Rij), dim=0)

        # Sort along first dimension (necessary for atom-wise pooling)
        sorted_idx = torch.argsort(bi_idx_i)

        inputs[Structure.idx_i] = bi_idx_i[sorted_idx]
        inputs[Structure.idx_j] = bi_idx_j[sorted_idx]
        inputs[Structure.Rij] = bi_Rij[sorted_idx]
        inputs[Structure.cell_offset] = bi_idx_S[sorted_idx]

        return inputs

    def _get_neighbor_pairs(self, positions, cell, shifts):
        """Compute pairs of atoms that are neighbors
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Arguments:
            positions (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
        """
        num_atoms = positions.shape[0]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # 1) Central cell
        pi_center, pj_center = torch.combinations(all_atoms).unbind(-1)
        shifts_center = shifts.new_zeros(pi_center.shape[0], 3)

        # 2) cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        shift_index, pi, pj = torch.cartesian_prod(
            all_shifts, all_atoms, all_atoms
        ).unbind(-1)
        shifts_outside = shifts.index_select(0, shift_index)

        # 3) combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        pi_all = torch.cat([pi_center, pi])
        pj_all = torch.cat([pj_center, pj])

        # 4) Compute shifts and distance vectors
        shift_values = torch.mm(shifts_all.to(cell.dtype), cell)
        Rij_all = positions[pi_all] - positions[pj_all] + shift_values

        # 5) Compute distances, and find all pairs within cutoff
        distances = torch.norm(Rij_all, dim=1)
        in_cutoff = torch.nonzero(distances < self.cutoff, as_tuple=False)

        # 6) Reduce tensors to relevant components
        pair_index = in_cutoff.squeeze()
        atom_index_i = pi_all[pair_index]
        atom_index_j = pj_all[pair_index]
        shifts = shifts_all.index_select(0, pair_index)
        Rij = Rij_all.index_select(0, pair_index)

        return atom_index_i, atom_index_j, shifts, Rij

    def _get_shifts(self, cell, pbc):
        """Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration.
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)

        Arguments:
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
                if pbc is enabled for that direction.

        Returns:
            :class:`torch.Tensor`: long tensor of shifts. the center cell and
                symmetric cells are not included.
        """
        reciprocal_cell = cell.inverse().t()
        inverse_lengths = torch.norm(reciprocal_cell, dim=1)

        num_repeats = torch.ceil(self.cutoff * inverse_lengths).long()
        num_repeats = torch.where(
            pbc, num_repeats, torch.Tensor([0], device=cell.device).long()
        )

        r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
        r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
        r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
        o = torch.zeros(1, dtype=torch.long, device=cell.device)

        return torch.cat(
            [
                torch.cartesian_prod(r1, r2, r3),
                torch.cartesian_prod(r1, r2, o),
                torch.cartesian_prod(r1, r2, -r3),
                torch.cartesian_prod(r1, o, r3),
                torch.cartesian_prod(r1, o, o),
                torch.cartesian_prod(r1, o, -r3),
                torch.cartesian_prod(r1, -r2, r3),
                torch.cartesian_prod(r1, -r2, o),
                torch.cartesian_prod(r1, -r2, -r3),
                torch.cartesian_prod(o, r2, r3),
                torch.cartesian_prod(o, r2, o),
                torch.cartesian_prod(o, r2, -r3),
                torch.cartesian_prod(o, o, r3),
            ]
        )


class NumbaNeighborList(nn.Module):
    """
    Calculate neighbor list using ASE.

    Note: This is quite slow and should only used as a baseline for faster implementations!
    """

    def __init__(self, cutoff, max_neighbors=10000000):
        """
        Args:
            cutoff: cutoff radius for neighbor search
        """
        super(NumbaNeighborList, self).__init__()
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

    def forward(self, inputs):
        R = inputs[Structure.R].numpy()
        cell = inputs[Structure.cell].numpy()
        pbc = inputs[Structure.pbc].numpy()

        idx_i, idx_j, idx_S, Rij = numba_neighbors(
            R, cell, self.cutoff, pbc, max_neighbors=self.max_neighbors
        )

        inputs[Structure.idx_i] = torch.tensor(idx_i)
        inputs[Structure.idx_j] = torch.tensor(idx_j)
        inputs[Structure.Rij] = torch.tensor(Rij)
        inputs[Structure.cell_offset] = torch.tensor(idx_S)

        return inputs


@jit(
    "(float64[:, :], float64[:, :], float64, boolean[:], int64)",
    nopython=True,
    nogil=True,
    fastmath=True,
)
def numba_neighbors(positions, cell, cutoff, pbc, max_neighbors=1000000):
    n_atoms = positions.shape[0]
    box = np.diag(cell)

    atom_offset = np.zeros_like(positions).astype(np.int64)
    cutoff_sq = cutoff * cutoff

    cell_vec_idx = np.zeros(3).astype(np.int64)
    offset = np.zeros(3).astype(np.int64)

    n_cells = (box / cutoff).astype(np.int64)

    n_total = n_cells[0] * n_cells[1] * n_cells[2]
    lyz = n_cells[1] * n_cells[2]
    lz = n_cells[2]

    l_cells = box / n_cells

    head = -np.ones(n_total).astype(np.int64)
    lscl = -np.ones(n_atoms).astype(np.int64)

    all_Rij = np.empty((max_neighbors, 3)).astype(np.float64)
    all_offsets = np.empty((max_neighbors, 3)).astype(np.int64)
    all_idx_i = np.empty(max_neighbors).astype(np.int64)
    all_idx_j = np.empty(max_neighbors).astype(np.int64)

    # 1) Build the cell list, assumes origin is the center of the cell
    for idx_atom in range(n_atoms):
        dx = positions[idx_atom] + 0.5 * box

        for idx_xyz in range(3):
            if pbc[idx_xyz]:
                if dx[idx_xyz] < 0.0:
                    dx[idx_xyz] = dx[idx_xyz] + box[idx_xyz]
                if dx[idx_xyz] >= box[idx_xyz]:
                    dx[idx_xyz] = dx[idx_xyz] - box[idx_xyz]

            cell_vec_idx[idx_xyz] = dx[idx_xyz] / l_cells[idx_xyz]

            if not pbc[idx_xyz]:
                if cell_vec_idx[idx_xyz] < 0:
                    cell_vec_idx[idx_xyz] = 0
                if cell_vec_idx[idx_xyz] >= n_cells[idx_xyz]:
                    cell_vec_idx[idx_xyz] = n_cells[idx_xyz] - 1

        idx_cell = cell_vec_idx[0] * lyz + cell_vec_idx[1] * lz + cell_vec_idx[2]

        lscl[idx_atom] = head[idx_cell]
        head[idx_cell] = idx_atom

    n_neigh = 0

    # 2) translate the cell list into a Verlet list and collect the offsets
    for idx_atom in range(n_atoms):

        # TODO:
        # a) store cell idx
        # b) store cell vector idx
        # c) more elegant way to dal with pbc and offsets only ONCE!

        dx = positions[idx_atom] + 0.5 * box

        # Wrap PBC
        for idx_xyz in range(3):
            if pbc[idx_xyz]:
                if dx[idx_xyz] < 0.0:
                    dx[idx_xyz] = dx[idx_xyz] + box[idx_xyz]
                    atom_offset[idx_atom, idx_xyz] = 1
                if dx[idx_xyz] >= box[idx_xyz]:
                    dx[idx_xyz] = dx[idx_xyz] - box[idx_xyz]
                    atom_offset[idx_atom, idx_xyz] = -1

            cell_vec_idx[idx_xyz] = dx[idx_xyz] / l_cells[idx_xyz]

            if not pbc[idx_xyz]:
                if cell_vec_idx[idx_xyz] < 0:
                    cell_vec_idx[idx_xyz] = 0
                if cell_vec_idx[idx_xyz] >= n_cells[idx_xyz]:
                    cell_vec_idx[idx_xyz] = n_cells[idx_xyz] - 1

        for cell_x in range(cell_vec_idx[0] - 1, cell_vec_idx[0] + 2):

            offset[0] = 0
            idx_x = cell_x

            # Compute offsets if pbc are requested
            if pbc[0]:
                if cell_x < 0:
                    offset[0] = -1
                if cell_x >= n_cells[0]:
                    offset[0] = 1

                # Wrap index back into box
                idx_x = (cell_x + n_cells[0]) % n_cells[0]
            else:
                if cell_x < 0:
                    continue
                if cell_x >= n_cells[0]:
                    continue

            for cell_y in range(cell_vec_idx[1] - 1, cell_vec_idx[1] + 2):

                offset[1] = 0
                idx_y = cell_y

                # Compute offsets if pbc are requested
                if pbc[1]:
                    if cell_y < 0:
                        offset[1] = -1
                    if cell_y >= n_cells[1]:
                        offset[1] = 1

                    # Wrap index back into box
                    idx_y = (cell_y + n_cells[1]) % n_cells[1]
                else:
                    if cell_y < 0:
                        continue
                    if cell_y >= n_cells[1]:
                        continue

                for cell_z in range(cell_vec_idx[2] - 1, cell_vec_idx[2] + 2):

                    offset[2] = 0
                    idx_z = cell_z

                    # Compute offsets if pbc are requested
                    if pbc[2]:
                        if cell_z < 0:
                            offset[2] = -1
                        if cell_z >= n_cells[2]:
                            offset[2] = 1

                        # Wrap index back into box
                        idx_z = (cell_z + n_cells[2]) % n_cells[2]
                    else:
                        if cell_z < 0:
                            continue
                        if cell_z >= n_cells[2]:
                            continue

                    idx_cell_nbh = idx_x * lyz + idx_y * lz + idx_z

                    # Get head of current neighbor chain
                    idx_atom_nbh = head[idx_cell_nbh]

                    while idx_atom_nbh != -1:

                        # Skip self interaction
                        if idx_atom_nbh != idx_atom:

                            effective_offset = (
                                offset
                                - atom_offset[idx_atom]
                                + atom_offset[idx_atom_nbh]
                            )
                            # print(offset, atom_offset[idx_atom], atom_offset[idx_atom_nbh], "OOO")
                            Rij = (
                                positions[idx_atom_nbh]
                                - positions[idx_atom]
                                + effective_offset * box
                            )

                            dist = Rij[0] * Rij[0] + Rij[1] * Rij[1] + Rij[2] * Rij[2]

                            if dist < cutoff_sq:
                                all_idx_i[n_neigh] = idx_atom
                                all_idx_j[n_neigh] = idx_atom_nbh
                                all_offsets[n_neigh] = effective_offset
                                all_Rij[n_neigh] = Rij

                                n_neigh = n_neigh + 1

                        # Get next atom in link
                        idx_atom_nbh = lscl[idx_atom_nbh]

    # Reduce arrays to proper size
    all_idx_i = all_idx_i[:n_neigh]
    all_idx_j = all_idx_j[:n_neigh]
    all_offsets = all_offsets[:n_neigh]
    all_Rij = all_Rij[:n_neigh]

    return all_idx_i, all_idx_j, all_offsets, all_Rij


## casting


class CastMap(nn.Module):
    """
    Cast all inputs according to type map.
    """

    def __init__(self, type_map: Dict[torch.dtype, torch.dtype]):
        """
        Args:
            type_map: dict with soource_type: target_type
        """
        super().__init__()
        self.type_map = type_map

    def forward(self, inputs):
        for k, v in inputs.items():
            if v.dtype in self.type_map:
                inputs[k] = v.to(dtype=self.type_map[v.dtype])
        return inputs


class CastTo32(CastMap):
    """ Cast all float64 tensors to float32 """

    def __init__(self):
        super().__init__(type_map={torch.float64: torch.float32})


## centering


class SubtractCenterOfMass(nn.Module):
    """
    Subtract center of mass from positions. Can only be used for single structures. Batches of structures are not supported.

    """

    def forward(self, inputs):
        masses = torch.tensor(atomic_masses[inputs[Structure.Z]])
        inputs[Structure.position] -= (
            masses.unsqueeze(-1) * inputs[Structure.position]
        ).sum(0) / masses.sum()
        return inputs


class SubtractCenterOfGeometry(nn.Module):
    """
    Subtract center of geometry from positions. Can only be used for single structures. Batches of structures are not supported.

    """

    def forward(self, inputs):
        inputs[Structure.position] -= inputs[Structure.position].mean(0)
        return inputs
