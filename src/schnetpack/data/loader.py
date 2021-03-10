import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

logger = logging.getLogger(__name__)

from schnetpack import Properties
from .stats import StatisticsAccumulator

__all__ = ["AtomsLoader"]


def _collate_aseatoms(batch):
    """
    Build batch from systems and properties & apply padding

    Args:
        examples (list):

    Returns:
        dict[str->torch.Tensor]: mini-batch of atomistic systems
    """
    elem = batch[0]

    coll_batch = {}
    n_atoms = torch.tensor([0] + [d[Properties.Z].shape[0] for d in batch])
    seg_m = torch.cumsum(n_atoms, dim=0)
    coll_batch[Properties.seg_m] = seg_m
    coll_batch[Properties.n_atoms] = n_atoms[1:]

    for key in elem:
        if key in [Properties.idx_i, Properties.idx_j]:
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], 0
            )
        else:
            coll_batch[key] = torch.cat([d[key] for d in batch], 0)

    return coll_batch


class AtomsLoader(DataLoader):
    r"""
    Specialized ``torch.data.DataLoader`` which uses the correct
    collate_fn for AtomsData and provides functionality for calculating mean
    and stddev.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        sampler (Sampler, optional): defines the strategy to draw samples from
            the dataset. If specified, ``shuffle`` must be False.
        batch_sampler (Sampler, optional): like sampler, but returns a batch of
            indices at a time. Mutually exclusive with batch_size, shuffle,
            sampler, and drop_last.
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.
            (default: 0)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch (default: collate_atons).
        pin_memory (bool, optional): If ``True``, the data loader will copy
            tensors into CUDA pinned memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete
            batch, if the dataset size is not divisible by the batch size.
            If ``False`` and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller. (default: False)
        timeout (numeric, optional): if positive, the timeout value for
            collecting a batch from workers. Should always be non-negative.
            (default: 0)
        worker_init_fn (callable, optional): If not None, this will be called
            on each worker subprocess with the worker id (an int in
            ``[0, num_workers - 1]``) as input, after seeding and before data
            loading. (default: None)

    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=_collate_aseatoms,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        num_workers = 0
        super(AtomsLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )

    def get_statistics(
        self, property_names, divide_by_atoms=False, single_atom_ref=None
    ):
        """
        Compute mean and variance of a property. Uses the incremental Welford
        algorithm implemented in StatisticsAccumulator

        Args:
            property_names (str or list): Name of the property for which the
                mean and standard deviation should be computed
            divide_by_atoms (dict or bool): divide mean by number of atoms if True
                (default: False)
            single_atom_ref (dict or bool): reference values for single atoms (default:
                None)

        Returns:
            mean: Mean value
            stddev: Standard deviation

        """
        if type(property_names) is not list:
            property_names = [property_names]
        if type(divide_by_atoms) is not dict:
            divide_by_atoms = {prop: divide_by_atoms for prop in property_names}
        if single_atom_ref is None:
            single_atom_ref = {prop: None for prop in property_names}

        with torch.no_grad():
            statistics = {
                prop: StatisticsAccumulator(batch=True) for prop in property_names
            }
            logger.info("statistics will be calculated...")

            for row in self:
                for prop in property_names:
                    self._update_statistic(
                        divide_by_atoms[prop],
                        single_atom_ref[prop],
                        prop,
                        row,
                        statistics[prop],
                    )

            means = {prop: s.get_mean() for prop, s in statistics.items()}
            stddevs = {prop: s.get_stddev() for prop, s in statistics.items()}

        return means, stddevs

    def _update_statistic(
        self, divide_by_atoms, single_atom_ref, property_name, row, statistics
    ):
        """
        Helper function to update iterative mean / stddev statistics
        """
        property_value = row[property_name]
        if single_atom_ref is not None:
            z = row["_atomic_numbers"]
            p0 = torch.sum(torch.from_numpy(single_atom_ref[z]).float(), dim=1)
            property_value -= p0
        if divide_by_atoms:
            property_value /= row[Properties.n_atoms]
        statistics.add_sample(property_value)
