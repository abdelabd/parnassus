"""

"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Callable, Union, Tuple

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP
from parnassus.torch_delphes import pdg_filters


class SimpleCalorimeter(nn.Module):
    """
    """

    def __init__(self, 
        eta_bins: List[float],           # From TCL EtaPhiBins
        phi_bins: List[List[float]],     # Per-eta phi bins
        energy_fractions: Dict[int, float],  # PDG → fraction
        resolution_formula: str,         # 'ecal_cms' or callable
        energy_min: float = 0.5,
        energy_significance_min: float = 2.0,
        is_ecal: bool = True
    ) -> None:
        super().__init__()
    
    def forward(self, 
        particles: torch.Tensor,  # (N_particles, N_FEATURES)
        tracks: torch.Tensor      # (N_tracks, N_FEATURES)
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            towers: (N_towers, N_FEATURES) - calorimeter towers
            eflow_tracks: (N_eflow_tracks, N_FEATURES) - tracks for particle flow
            eflow_photons: (N_eflow_photons, N_FEATURES) - neutral excess towers
        """

        ######## 1. Compute Energy Fractions ########


        ######## 2. Bin tracks into Towers ########


        ######## 3. Bin particles into Towers ########


        ######## 4. Aggregate Energies per Tower ########


        ######## 5. Compute Tower Centers ########


        ######## 6. Apply Resolution Smearing ########


        ######## 7. Compute track sigma per Tower ########


        ######## 8. Identify Neutral Excess and Create eflow objects ########
