from typing import Callable

import torch
from torch import nn
from torch_scatter import segment_csr, segment_coo


from schnetpack.nn import Dense
from schnetpack import Properties
from schnetpack.nn.activations import shifted_softplus


def cfconv(x, Wij, idx_i, idx_j, reduce="sum"):
    """
    Continuous-filter convolution.

    Args:
        x: input values
        Wij: filter
        idx_i: index of center atom i
        idx_j: index of neighbors j
        reduce: reduction method (sum, mean, ...)

    Returns:
        convolved inputs

    """
    x_ij = x[idx_j] * Wij
    y = segment_coo(x_ij, idx_i, reduce=reduce)

    return y


class CFConv(nn.Module):
    """
    Continuous-filter convolution.

    Args:
        reduce: reduction method (sum, mean, ...)
    """

    def __init__(self, reduce="sum"):
        super().__init__()
        self.reduce = reduce

    def forward(self, x, Wij, idx_i, idx_j):
        """
        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            convolved inputs
        """
        return cfconv(x, Wij, idx_i, idx_j, self.reduce)


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems.

    Args:
        n_atom_basis: number of features to describe atomic environments.
        n_rbf (int): number of radial basis functions.
        n_filters: number of filters used in continuous-filter convolution.
        normalize_filter: if True, divide aggregated filter by number
            of neighbors over which convolution is applied.
        activation: if None, no activation function is used.
    """

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        normalize_filter: bool = False,
        activation: Callable = shifted_softplus,
    ):
        super(SchNetInteraction, self).__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.cfconv = CFConv(reduce="mean" if normalize_filter else "sum")
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation),
            Dense(n_filters, n_filters),
        )

    def forward(self, x, f_ij, idx_i, idx_j, rcut_ij):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)

        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]
        x = self.cfconv(x, Wij, idx_i, idx_j)
        x = self.f2out(x)
        return x


class SchNet(nn.Module):
    def __init__(
        self,
        n_atom_basis,
        n_interactions,
        radial_basis,
        cutoff_fn,
        n_filters=None,
        normalize_filter=False,
        coupled_interactions=False,
        return_intermediate=False,
        max_z=100,
        charged_systems=False,
        activation=shifted_softplus,
    ):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn

        self.return_intermediate = return_intermediate
        self.charged_systems = charged_systems
        if charged_systems:
            self.charge = nn.Parameter(torch.Tensor(1, self.n_atom_basis))
            self.charge.data.normal_(0, 1.0 / self.n_atom_basis ** 0.5)

        # layers
        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        if coupled_interactions:
            # use the same SchNetInteraction instance (hence the same weights)
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=self.n_atom_basis,
                        n_rbf=self.radial_basis.n_rbf,
                        n_filters=self.n_filters,
                        normalize_filter=normalize_filter,
                        activation=activation,
                    )
                ]
                * n_interactions
            )
        else:
            # use one SchNetInteraction instance for each interaction
            self.interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=self.n_atom_basis,
                        n_rbf=self.radial_basis.n_rbf,
                        n_filters=self.n_filters,
                        normalize_filter=normalize_filter,
                        activation=activation,
                    )
                    for _ in range(n_interactions)
                ]
            )

    def forward(self, atomic_numbers, r_ij, idx_i, idx_j):
        # compute atom and pair features
        x = self.embedding(atomic_numbers)
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # store intermediate representations
        if self.return_intermediate:
            xs = [x]

        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v
            if self.return_intermediate:
                xs.append(x)

        if self.return_intermediate:
            return x, xs
        return x
