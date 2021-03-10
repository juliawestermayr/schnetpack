import pytest
import numpy as np
from schnetpack.nn.cutoff import HardCutoff
import schnetpack as spk

__all__ = [
    # spk.representation
    "n_atom_basis",
    "n_filters",
    "n_interactions",
    "cutoff",
    "n_rbf",
    "normalize_filter",
    "coupled_interactions",
    "return_intermediate",
    "max_z",
    "cutoff_fn",
    "trainable_rbf",
    "charged_systems",
    "schnet",
    "schnet_interaction",
    "radial_basis",
    # spk.atomistic
    "properties1",
    "properties2",
    "output_module_1",
    "output_module_2",
    "output_modules",
    "atomistic_model",
    # spk.nn
    "gaussion_smearing_layer",
    "cfconv_layer",
    "dense_layer",
    "mlp_layer",
    "n_mlp_tiles",
    "tiled_mlp_layer",
    "elements",
    "elemental_gate_layer",
    "atom_distances",
]


# spk.representation
## settings
@pytest.fixture
def n_atom_basis():
    return 128


@pytest.fixture
def n_filters():
    return 128


@pytest.fixture
def n_interactions():
    return 3


@pytest.fixture
def cutoff():
    return 5.0


@pytest.fixture
def n_rbf():
    return 25


@pytest.fixture
def normalize_filter():
    return False


@pytest.fixture
def coupled_interactions():
    return False


@pytest.fixture
def return_intermediate():
    return False


@pytest.fixture
def max_z():
    return 100


@pytest.fixture
def cutoff_fn(cutoff):
    return HardCutoff(cutoff)


@pytest.fixture(params=[True, False])
def trainable_rbf(request):
    return request.param


@pytest.fixture
def radial_basis(n_rbf, cutoff, trainable_rbf):
    return spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff, trainable=trainable_rbf)


@pytest.fixture
def charged_systems():
    return False


## models
@pytest.fixture
def schnet(
    n_atom_basis,
    n_filters,
    n_interactions,
    n_rbf,
    normalize_filter,
    coupled_interactions,
    max_z,
    cutoff_fn,
    radial_basis,
):
    return spk.SchNet(
        n_atom_basis=n_atom_basis,
        n_filters=n_filters,
        n_interactions=n_interactions,
        normalize_filter=normalize_filter,
        coupled_interactions=coupled_interactions,
        max_z=max_z,
        cutoff_fn=cutoff_fn,
        radial_basis=radial_basis,
    )


@pytest.fixture
def schnet_interaction(
    n_atom_basis, n_rbf, n_filters, cutoff, cutoff_fn, normalize_filter
):
    return spk.representation.SchNetInteraction(
        n_atom_basis=n_atom_basis,
        n_rbf=n_rbf,
        n_filters=n_filters,
        normalize_filter=normalize_filter,
    )


# spk.atomistic
@pytest.fixture
def properties1(available_properties):
    return [prop for prop in available_properties if prop.endswith("1")]


@pytest.fixture
def properties2(available_properties):
    return [prop for prop in available_properties if prop.endswith("2")]


@pytest.fixture
def output_module_1(n_atom_basis, properties1):
    om_properties = get_module_properties(properties1)
    return spk.atomistic.Atomwise(
        n_in=n_atom_basis,
        property=om_properties["property"],
        contributions=om_properties["contributions"],
        derivative=om_properties["derivative"],
    )


@pytest.fixture
def output_module_2(n_atom_basis, properties2):
    om_properties = get_module_properties(properties2)
    return spk.atomistic.Atomwise(
        n_in=n_atom_basis,
        property=om_properties["property"],
        contributions=om_properties["contributions"],
        derivative=om_properties["derivative"],
    )


@pytest.fixture
def output_modules(output_module_1, output_module_2):
    return [output_module_1, output_module_2]


@pytest.fixture
def atomistic_model(schnet, output_modules):
    return spk.AtomisticModel(schnet, output_modules)


# spk.nn
@pytest.fixture
def gaussion_smearing_layer(n_rbf, trainable_rbf):
    return spk.nn.GaussianSmearing(n_gaussians=n_rbf, trainable=trainable_rbf)


@pytest.fixture
def cfconv_layer(n_atom_basis, n_filters, schnet_interaction):
    return spk.nn.CFConv(
        reduce="mean" if normalize_filter else "sum",
    )


@pytest.fixture
def dense_layer(random_input_dim, random_output_dim):
    return spk.nn.Dense(random_input_dim, random_output_dim)


@pytest.fixture
def mlp_layer(random_input_dim, random_output_dim):
    print(random_input_dim, "MLP", random_output_dim)
    return spk.nn.MLP(random_input_dim, random_output_dim)


@pytest.fixture
def n_mlp_tiles():
    return np.random.randint(1, 6, 1).item()


@pytest.fixture
def tiled_mlp_layer(random_input_dim, random_output_dim, n_mlp_tiles):
    return spk.nn.TiledMultiLayerNN(random_input_dim, random_output_dim, n_mlp_tiles)


@pytest.fixture
def elements():
    return list(set(np.random.randint(1, 30, 10)))


@pytest.fixture
def elemental_gate_layer(elements):
    return spk.nn.ElementalGate(elements=elements)


@pytest.fixture
def atom_distances():
    return spk.nn.AtomDistances()


# utility functions
def get_module_properties(properties):
    """
    Get dict of properties for output module.

    Args:
        properties (list): list of properties

    Returns:
        (dict): dict with prop, der and contrib

    """
    module_props = dict(property=None, derivative=None, contributions=None)
    for prop in properties:
        if prop.startswith("property"):
            module_props["property"] = prop
        elif prop.startswith("derivative"):
            module_props["derivative"] = prop
        elif prop.startswith("contributions"):
            module_props["contributions"] = prop
    return module_props
