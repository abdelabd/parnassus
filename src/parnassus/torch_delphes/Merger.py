"""
PyTorch implementation of Delphes Merger module.

Combines multiple input particle collections into a single output tensor.
This is a simple concatenation operation that preserves all particle properties.

Common uses:
- **TrackMerger**: Combines charged hadrons, electrons, and muons into unified tracks
- **CalorimeterMerger**: Combines ECal and HCal towers (and optionally muons for ATLAS)

Reference:
    C++ Delphes: modules/Merger.cc
"""

import torch
import torch.nn as nn
from typing import List

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP


class Merger(nn.Module):
    """
    PyTorch implementation of Delphes Merger module.
    
    Concatenates multiple particle tensors into a single output tensor along
    the particle dimension. No transformations are applied to the particles;
    all properties are preserved exactly as input.
    
    This is the simplest merger that just combines collections. For mergers
    that need to transform particles (e.g., setting PID, zeroing positions),
    use EFlowMerger instead.
    
    Example:
        >>> merger = Merger()
        >>> tracks = merger([charged_hadrons, electrons, muons])
        >>> towers = merger([ecal_towers, hcal_towers])
    """
    
    def __init__(self) -> None:
        """Initialize the Merger module (no parameters needed)."""
        super().__init__()

    def forward(self, input_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate multiple particle tensors.
        
        Args:
            input_tensors: List of tensors, each of shape (N_i, N_FEATURES).
                All tensors must have the same number of features (columns).
                Empty tensors (N_i = 0) are handled gracefully.
                
        Returns:
            Concatenated tensor of shape (sum(N_i), N_FEATURES) containing
            all particles from all input tensors.
        """
        return torch.cat(input_tensors, dim=0)
