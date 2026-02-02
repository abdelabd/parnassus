"""
PyTorch implementation of Delphes Merger module.

The Merger module combines multiple input arrays into a single output array.
In the tensor representation, this means applying a mask to indicate which
particles should be included in the merged output.

For TrackMerger specifically, this combines:
- Charged hadrons (after momentum smearing)
- Electrons (after momentum smearing)
- Muons (after momentum smearing)
into a unified "tracks" collection.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP

#TODO: Update docstrings

class Merger(nn.Module):
    """
    TODO: Update docstring
    PyTorch implementation of Delphes Merger module.
    
    This module:
    1. Filters particles based on PID (which particle types to include)
    
    For TrackMerger, this combines charged hadrons, electrons, and muons
    that have passed propagation, efficiency, and momentum smearing.
    
    Input shape: (N_events, N_particles, N_FEATURES)
        Must contain IS_NOT_PAD, PASS_PROP mask columns
    
    Output shape: (N_events, N_particles, D)
    """
    
    def __init__(self) -> None:
        """
        Args:
            particle_types: List of particle types to include in merger
                           Options: 'charged_hadron', 'electron', 'muon', 'neutral'
        """
        super().__init__()

    def forward(self, different_particle_type_tensors: List[torch.Tensor],) -> torch.Tensor:
        """
        TODO: Update docstring
        Apply merger to create unified output
        
        Args:
            different_particle_type_tensors: List of tensors for each particle type
                Each tensor shape: (N_events, N_particles_type, N_FEATURES)
                
        Returns:
            track_tensors: tensor of shape (N_events, N_particles, D)
        """

        track_tensors = torch.cat(different_particle_type_tensors, dim=0)
        return track_tensors
