"""
PyTorch implementation of Delphes SimpleCalorimeter module.

Fills calorimeter towers, performs energy resolution smearing,
and creates energy flow objects (tracks and photons).

This module:
1. Bins particles into (η, φ) towers based on position
2. Accumulates energy per tower with PDG-dependent fractions
3. Applies log-normal energy resolution smearing
4. Computes energy flow by comparing calorimeter and track energies
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Callable, Union, Tuple

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP
from parnassus.torch_delphes import pdg_filters


class SimpleCalorimeter(nn.Module):
    """
    PyTorch implementation of Delphes SimpleCalorimeter module.
    
    Input:
        particles: (N_particles, N_FEATURES) - stable particles after propagation
        tracks: (N_tracks, N_FEATURES) - merged tracks after momentum smearing
    
    Output:
        towers: (N_towers, N_FEATURES) - calorimeter towers with smeared energy
        eflow_tracks: (N_eflow_tracks, N_FEATURES) - tracks for particle flow
        eflow_photons: (N_eflow_photons, N_FEATURES) - neutral excess towers
    """

    def __init__(self, 
        eta_bins: List[float],           # From TCL EtaPhiBins
        phi_bins: List[float],           # Phi bin edges (same for all eta bins)
        energy_fractions: Dict[int, float],  # PDG → fraction
        resolution_formula: Union[str, Callable] = 'ecal_cms',
        energy_min: float = 0.5,
        energy_sig_min: float = 2.0,
        is_ecal: bool = True
    ) -> None:
        super().__init__()
        
        # Store configuration
        self.energy_min = energy_min
        self.energy_sig_min = energy_sig_min
        self.is_ecal = is_ecal
        
        # Store eta bins as tensor
        self.register_buffer('eta_bins', torch.tensor(eta_bins, dtype=torch.float64))
        
        # Store phi bins as tensor (flat list - same phi bins for all eta bins)
        self.register_buffer('phi_bins', torch.tensor(phi_bins, dtype=torch.float64))
        
        # Store energy fractions as lookup table
        # We'll use a tensor for common PDGs and handle edge cases
        self.energy_fractions = energy_fractions
        self.default_fraction = energy_fractions.get(0, 0.0)
        
        # Create lookup tensor for common PDG IDs (up to some max)
        # PDG IDs we care about: 11, 12, 13, 14, 16, 22, 111, 310, 3122, 1000022, etc.
        self._setup_fraction_lookup()
        
        # Resolution formula
        if resolution_formula == 'ecal_cms':
            self.resolution_func = self._ecal_cms_resolution
        elif callable(resolution_formula):
            self.resolution_func = resolution_formula
        else:
            raise ValueError(f"Unknown resolution formula: {resolution_formula}")

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
        # Get PDG IDs
        particle_pids = particles[:, CMAP["PID"]]
        track_pids = tracks[:, CMAP["PID"]]
        
        # Compute energy fractions based on PDG ID
        # This matches fTowerFractions and fTrackFractions in C++
        particle_energy_fractions = self._compute_energy_fractions(particle_pids)
        track_energy_fractions = self._compute_energy_fractions(track_pids)


        ######## 2. Bin tracks into Towers ########


        ######## 3. Bin particles into Towers ########


        ######## 4. Aggregate Energies per Tower ########


        ######## 5. Compute Tower Centers ########


        ######## 6. Apply Resolution Smearing ########


        ######## 7. Compute track sigma per Tower ########


        ######## 8. Identify Neutral Excess and Create eflow objects ########

        return particle_energy_fractions, track_energy_fractions

        
    def _setup_fraction_lookup(self) -> None:
        """Setup efficient PDG → energy fraction lookup."""
        # Store known PDG IDs and their fractions as tensors for vectorized lookup
        pdg_ids = list(self.energy_fractions.keys())
        fractions = [self.energy_fractions[pid] for pid in pdg_ids]
        
        self.register_buffer('known_pdg_ids', torch.tensor(pdg_ids, dtype=torch.int64))
        self.register_buffer('known_fractions', torch.tensor(fractions, dtype=torch.float64))
    
    def _compute_energy_fractions(self, pids: torch.Tensor) -> torch.Tensor:
        """
        Compute energy fractions for particles based on their PDG IDs.
        
        This matches the C++ Delphes behavior:
        1. Look up |PDG ID| in the fraction map
        2. If not found, use the default fraction (PDG=0)
        
        Args:
            pids: (N,) tensor of PDG IDs
            
        Returns:
            fractions: (N,) tensor of energy fractions
        """
        abs_pids = torch.abs(pids).to(torch.int64)
        
        # Initialize with default fraction
        fractions = torch.full_like(pids, self.default_fraction, dtype=torch.float64)
        
        # Look up each known PDG ID and set its fraction
        for i, (known_pid, known_frac) in enumerate(zip(self.known_pdg_ids, self.known_fractions)):
            if known_pid == 0:
                continue  # Skip default, already set
            mask = (abs_pids == known_pid)
            fractions = torch.where(mask, known_frac, fractions)
        
        return fractions
    
    
    @staticmethod
    def _ecal_cms_resolution(eta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """
        ECAL resolution formula from delphes_card_CMS_5_0.tcl.
        
        Formula:
            |eta| <= 1.5: (1 + 0.64*eta^2) * sqrt(E^2*0.008^2 + E*0.11^2 + 0.40^2)
            1.5 < |eta| <= 2.5: (2.16 + 5.6*(|eta|-2)^2) * sqrt(E^2*0.008^2 + E*0.11^2 + 0.40^2)
            2.5 < |eta| <= 5.0: sqrt(E^2*0.107^2 + E*2.08^2)
        """
        abs_eta = torch.abs(eta)
        
        # Common term for barrel and endcap
        common_term = torch.sqrt(energy**2 * 0.008**2 + energy * 0.11**2 + 0.40**2)
        
        # Barrel: |eta| <= 1.5
        barrel_factor = 1.0 + 0.64 * eta**2
        barrel_sigma = barrel_factor * common_term
        
        # Endcap: 1.5 < |eta| <= 2.5
        endcap_factor = 2.16 + 5.6 * (abs_eta - 2.0)**2
        endcap_sigma = endcap_factor * common_term
        
        # Forward: 2.5 < |eta| <= 5.0
        forward_sigma = torch.sqrt(energy**2 * 0.107**2 + energy * 2.08**2)
        
        # Select based on eta region
        sigma = torch.where(abs_eta <= 1.5, barrel_sigma,
                    torch.where(abs_eta <= 2.5, endcap_sigma, forward_sigma))
        
        return sigma
