from torch import nn as nn
import torch

from schnetpack import Properties
import logging

__all__ = ["AtomisticModel"]


class ModelError(Exception):
    pass


class AtomisticModel(nn.Module):
    """
    Join a representation model with output modules.

    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_modules (list or nn.ModuleList or spk.output_modules.Atomwise): Output
            block of the model. Needed for predicting properties.

    Returns:
         dict: property predictions
    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])
        # For stress tensor
        self.requires_stress = any([om.stress for om in self.output_modules])

    def forward(self, inputs):
        """
        Forward representation output through output modules.
        """
        R = inputs[Properties.R]
        if self.requires_dr:
            R.requires_grad_()
        # if self.requires_stress:
        #     # Check if cell is present
        #     if inputs[Properties.cell] is None:
        #         raise ModelError("No cell found for stress computation.")
        #
        #     # Generate Cartesian displacement tensor
        #     displacement = torch.zeros_like(inputs[Properties.cell]).to(
        #         inputs[Properties.R].device
        #     )
        #     displacement.requires_grad = True
        #     inputs["displacement"] = displacement
        #
        #     # Apply to coordinates and cell
        #     R = R + torch.matmul(
        #         R, displacement
        #     )
        #     inputs[Properties.cell] = inputs[Properties.cell] + torch.matmul(
        #         inputs[Properties.cell], displacement
        #     )

        atomic_numbers = inputs[Properties.Z]
        seg_m = inputs[Properties.seg_m]
        C = inputs[Properties.cell]
        cell_offset = inputs[Properties.cell_offset]
        idx_i = inputs[Properties.idx_i]
        idx_j = inputs[Properties.idx_j]

        segdiff = seg_m[1:] - seg_m[:-1]
        C_i = torch.repeat_interleave(C, segdiff, dim=0)[idx_i]
        Cij = torch.bmm(C_i, cell_offset.float()[:, :, None]).squeeze(-1)
        r_ij = (R[idx_j] + Cij) - R[idx_i]

        inputs["representation"] = self.representation(
            atomic_numbers, r_ij, idx_i, idx_j
        )
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs
