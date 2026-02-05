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
        phi_bins: List[List[float]],     # Phi bins per eta bin - len must equal len(eta_bins)
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
        
        # TODO: Use constant phi bins in future after debugging
            # This is much slower for not much benefit, but matches Delphes behavior
        # Store phi bins per eta as list of tensors
        # phi_bins[i] gives the phi bin edges for eta bin i
        self.phi_bins_per_eta = [torch.tensor(pb, dtype=torch.float64) for pb in phi_bins]
        # Register a dummy buffer for device tracking
        self.register_buffer('_device_tracker', torch.tensor([0.0], dtype=torch.float64))
        
        # Store energy fractions as lookup table
        self.energy_fractions = energy_fractions
        self.default_fraction = energy_fractions.get(0, 0.0)
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


        ######## 2. Bin particles into Towers ########
        # C++: if(fraction < 1.0E-9) continue;  // particles with zero fraction are skipped
        # Get particle positions (eta, phi from Position, not Momentum)
        particle_eta = particles[:, CMAP["ETA_OUTER"]]  # Position-based eta
        particle_phi = particles[:, CMAP["PHI_OUTER"]]  # Position-based phi
        
        # Find eta bin for each particle
        particle_eta_bin = torch.searchsorted(self.eta_bins, particle_eta)
        
        # Find phi bin for each particle - variable per eta bin
        particle_phi_bin, particle_valid = self._compute_phi_bins(particle_phi, particle_eta_bin)
        
        # Particles: filter by energy fraction (C++: if(fraction < 1.0E-9) continue)
        particle_valid = particle_valid & (particle_energy_fractions > 1e-9)


        ######## 3. Bin tracks into Towers ########
        # C++: tracks are NOT filtered by fraction before binning
        # Get track positions (eta and phi from outer position, set by ParticlePropagator)
        # C++ uses track->Position.Eta() and track->Position.Phi()
        track_eta = tracks[:, CMAP["ETA_OUTER"]]  # Position-based eta
        track_phi = tracks[:, CMAP["PHI_OUTER"]]  # Position-based phi
        
        # Find eta bin for each track
        track_eta_bin = torch.searchsorted(self.eta_bins, track_eta)
        
        # Find phi bin for each track - variable per eta bin
        # Tracks: do NOT filter by energy fraction (C++ doesn't skip tracks with fraction < 1e-9)
        track_phi_bin, track_valid = self._compute_phi_bins(track_phi, track_eta_bin)


        ######## 4. Aggregate Energies per Tower ########
        # C++ Delphes:
        #   - Sorts all hits by (etaBin, phiBin)
        #   - For each tower (unique etaBin, phiBin):
        #       fTowerEnergy += momentum.E() * fTowerFractions[number]  (particles)
        #       fTrackEnergy += momentum.E() * fTrackFractions[number]  (tracks with fraction > 1e-9)
        #   - Time weighting is also done but we handle that separately
        
        # Get energies
        particle_energy = particles[:, CMAP["E"]]
        track_energy = tracks[:, CMAP["E"]]
        
        # Compute weighted energies (energy × fraction)
        particle_weighted_energy = particle_energy * particle_energy_fractions
        track_weighted_energy = track_energy * track_energy_fractions
        
        # Create unique tower index from (eta_bin, phi_bin)
        # Using max_phi_bins to create unique indices
        max_phi_bins = max(len(pb) for pb in self.phi_bins_per_eta)
        n_eta_bins = len(self.eta_bins)
        
        # Tower index = eta_bin * max_phi_bins + phi_bin
        particle_tower_idx = particle_eta_bin * max_phi_bins + particle_phi_bin
        track_tower_idx = track_eta_bin * max_phi_bins + track_phi_bin
        
        # For tracks, only those with fraction > 1e-9 contribute to fTrackEnergy
        # C++: if(fTrackFractions[number] > 1.0E-9) { fTrackEnergy += energy; ... }
        track_has_fraction = track_energy_fractions > 1e-9
        
        # Find unique towers from valid particles AND valid tracks
        # NOTE: In C++, particles are filtered by fraction BEFORE creating tower hits,
        #       but tracks are NOT filtered by fraction for tower creation.
        #       Tracks create tower hits regardless of fraction; the fraction check
        #       only affects whether they contribute to fTrackEnergy.
        valid_particle_tower_idx = torch.where(
            particle_valid, 
            particle_tower_idx, 
            torch.full_like(particle_tower_idx, -1)
        )
        valid_track_tower_idx = torch.where(
            track_valid,  # NOT filtered by fraction for tower creation
            track_tower_idx,
            torch.full_like(track_tower_idx, -1)
        )
        
        # Combine all valid tower indices to find unique towers
        all_tower_idx = torch.cat([
            valid_particle_tower_idx[particle_valid],
            valid_track_tower_idx[track_valid]  # NOT filtered by fraction
        ])
        unique_tower_idx = torch.unique(all_tower_idx[all_tower_idx >= 0])
        n_towers = len(unique_tower_idx)
        
        # Create mapping from global tower index to compact tower index [0, n_towers)
        tower_idx_map = torch.full((n_eta_bins * max_phi_bins,), -1, 
                                    dtype=torch.long, device=particles.device)
        tower_idx_map[unique_tower_idx] = torch.arange(n_towers, device=particles.device)
        
        # Map particles and tracks to compact tower indices
        # Clamp tower indices to valid range before lookup (invalid particles have valid=False anyway)
        max_idx = n_eta_bins * max_phi_bins - 1
        particle_tower_idx_clamped = particle_tower_idx.clamp(0, max_idx)
        track_tower_idx_clamped = track_tower_idx.clamp(0, max_idx)
        
        particle_compact_idx = torch.where(
            particle_valid,
            tower_idx_map[particle_tower_idx_clamped],
            torch.full_like(particle_tower_idx, -1)
        )
        track_compact_idx = torch.where(
            track_valid & track_has_fraction,
            tower_idx_map[track_tower_idx_clamped],
            torch.full_like(track_tower_idx, -1)
        )
        
        # Aggregate particle energies per tower using scatter_add
        tower_energy = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        valid_particle_mask = particle_compact_idx >= 0
        tower_energy.scatter_add_(
            0,
            particle_compact_idx[valid_particle_mask],
            particle_weighted_energy[valid_particle_mask]
        )
        
        # Aggregate track energies per tower
        tower_track_energy = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        valid_track_mask = track_compact_idx >= 0
        tower_track_energy.scatter_add_(
            0,
            track_compact_idx[valid_track_mask],
            track_weighted_energy[valid_track_mask]
        )
        
        # Extract eta_bin and phi_bin for each unique tower
        tower_eta_bin = unique_tower_idx // max_phi_bins
        tower_phi_bin = unique_tower_idx % max_phi_bins


        ######## 5. Compute Tower Centers ########
        # C++:
        #   fTowerEta = 0.5 * (fEtaBins[etaBin - 1] + fEtaBins[etaBin]);
        #   fTowerPhi = 0.5 * ((*phiBins)[phiBin - 1] + (*phiBins)[phiBin]);
        #   fTowerEdges[0] = fEtaBins[etaBin - 1];  // eta_lo
        #   fTowerEdges[1] = fEtaBins[etaBin];       // eta_hi
        #   fTowerEdges[2] = (*phiBins)[phiBin - 1]; // phi_lo
        #   fTowerEdges[3] = (*phiBins)[phiBin];     // phi_hi
        
        # Compute tower eta centers and edges
        # tower_eta_bin is in range [1, n_eta_bins-1] for valid towers
        tower_eta_lo = self.eta_bins[tower_eta_bin - 1]
        tower_eta_hi = self.eta_bins[tower_eta_bin]
        tower_eta = 0.5 * (tower_eta_lo + tower_eta_hi)
        
        # Compute tower phi centers and edges (variable per eta bin)
        # Need to loop over eta bins since phi bins differ
        tower_phi = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        tower_phi_lo = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        tower_phi_hi = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        
        for eb in range(1, len(self.eta_bins)):
            mask = (tower_eta_bin == eb)
            if not mask.any():
                continue
            
            phi_bins_eb = self.phi_bins_per_eta[eb].to(particles.device)
            pb = tower_phi_bin[mask]
            
            # phi_lo = phi_bins[phiBin - 1], phi_hi = phi_bins[phiBin]
            tower_phi_lo[mask] = phi_bins_eb[pb - 1]
            tower_phi_hi[mask] = phi_bins_eb[pb]
            tower_phi[mask] = 0.5 * (phi_bins_eb[pb - 1] + phi_bins_eb[pb])


        ######## 6. Apply Resolution Smearing ########


        ######## 7. Compute track sigma per Tower ########


        ######## 8. Identify Neutral Excess and Create eflow objects ########

        # Return intermediate results for validation
        # TODO: Update return signature once full pipeline is implemented
        return {
            'particle_energy_fractions': particle_energy_fractions,
            'track_energy_fractions': track_energy_fractions,
            'particle_eta_bin': particle_eta_bin,
            'particle_phi_bin': particle_phi_bin,
            'particle_valid': particle_valid,
            'track_eta_bin': track_eta_bin,
            'track_phi_bin': track_phi_bin,
            'track_valid': track_valid,
            # Tower aggregation outputs
            'n_towers': n_towers,
            'unique_tower_idx': unique_tower_idx,
            'tower_eta_bin': tower_eta_bin,
            'tower_phi_bin': tower_phi_bin,
            'tower_energy': tower_energy,           # Sum of particle energies × fractions
            'tower_track_energy': tower_track_energy,  # Sum of track energies × fractions
            'max_phi_bins': max_phi_bins,
            # Tower center and edge outputs
            'tower_eta': tower_eta,
            'tower_phi': tower_phi,
            'tower_eta_lo': tower_eta_lo,
            'tower_eta_hi': tower_eta_hi,
            'tower_phi_lo': tower_phi_lo,
            'tower_phi_hi': tower_phi_hi,
        }

    def _compute_phi_bins(self, 
        phi: torch.Tensor,           # (N,) phi values
        eta_bin: torch.Tensor,       # (N,) eta bin indices
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute phi bin for each particle/track using variable phi binning per eta.
        
        In C++ Delphes:
            phiBins = fPhiBins[etaBin];
            itPhiBin = lower_bound(phiBins->begin(), phiBins->end(), position.Phi());
            if(itPhiBin == phiBins->begin() || itPhiBin == phiBins->end()) continue;
            phiBin = distance(phiBins->begin(), itPhiBin);
        
        Returns:
            phi_bin: (N,) phi bin indices
            valid: (N,) boolean mask for valid bins (eta and phi both in range)
        """
        n = len(phi)
        phi_bin = torch.zeros(n, dtype=torch.long, device=phi.device)
        valid = torch.zeros(n, dtype=torch.bool, device=phi.device)
        
        # Filter by valid eta bin first: [1, len(eta_bins) - 1]
        valid_eta = (eta_bin > 0) & (eta_bin < len(self.eta_bins))
        
        # For each eta bin, compute phi bins for all particles/tracks in that eta bin
        for eb in range(1, len(self.eta_bins)):
            mask = (eta_bin == eb) & valid_eta
            if not mask.any():
                continue
            
            # Get phi bins for this eta bin
            phi_bins_eb = self.phi_bins_per_eta[eb].to(phi.device)
            
            # Compute phi bin (equivalent to lower_bound + distance)
            phi_vals = phi[mask]
            pb = torch.searchsorted(phi_bins_eb, phi_vals)
            
            # Valid phi bin range: [1, len(phi_bins_eb) - 1]
            # (not at begin or end, matching C++ continue conditions)
            valid_phi = (pb > 0) & (pb < len(phi_bins_eb))
            
            # Store results
            phi_bin[mask] = pb
            valid[mask] = valid_phi
        
        return phi_bin, valid

        
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
