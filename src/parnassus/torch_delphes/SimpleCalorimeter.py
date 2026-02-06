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

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP, N_FEATURES
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
        is_ecal: bool = True,
        smear_tower_center: bool = True  # If True, smear eta/phi uniformly within bin
    ) -> None:
        super().__init__()
        
        # Store configuration
        self.energy_min = energy_min
        self.energy_sig_min = energy_sig_min
        self.is_ecal = is_ecal
        self.smear_tower_center = smear_tower_center
        
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
        elif resolution_formula == 'hcal_cms':
            self.resolution_func = self._hcal_cms_resolution
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
        
        # Compute tower phi centers and edges (variable per eta bin)
        # Need to loop over eta bins since phi bins differ
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
        
        # Compute tower eta and phi (either center or smeared uniformly within bin)
        if self.smear_tower_center:
            # C++: eta = gRandom->Uniform(fTowerEdges[0], fTowerEdges[1])
            #      phi = gRandom->Uniform(fTowerEdges[2], fTowerEdges[3])
            tower_eta = tower_eta_lo + torch.rand(n_towers, dtype=torch.float64, device=particles.device) * (tower_eta_hi - tower_eta_lo)
            tower_phi = tower_phi_lo + torch.rand(n_towers, dtype=torch.float64, device=particles.device) * (tower_phi_hi - tower_phi_lo)
        else:
            tower_eta = 0.5 * (tower_eta_lo + tower_eta_hi)
            tower_phi = 0.5 * (tower_phi_lo + tower_phi_hi)


        ######## 6. Apply Resolution Smearing ########
        # C++:
        #   sigma = fResolutionFormula->Eval(0.0, fTowerEta, 0.0, fTowerEnergy);
        #   energy = LogNormal(fTowerEnergy, sigma);
        #   sigma = fResolutionFormula->Eval(0.0, fTowerEta, 0.0, energy);  // recompute with smeared
        #   if(energy < fEnergyMin || energy < fEnergySignificanceMin * sigma) energy = 0.0;
        
        # Compute sigma before smearing
        sigma_before = self.resolution_func(tower_eta, tower_energy)
        
        # Apply LogNormal smearing
        tower_energy_smeared = self._log_normal_smear(tower_energy, sigma_before)
        
        # Recompute sigma with smeared energy
        sigma_after = self.resolution_func(tower_eta, tower_energy_smeared)
        
        # Apply energy thresholds
        # Tower energy is zeroed if below minimum or below significance threshold
        below_min = tower_energy_smeared < self.energy_min
        below_sig = tower_energy_smeared < self.energy_sig_min * sigma_after
        tower_energy_final = torch.where(below_min | below_sig, 
                                         torch.zeros_like(tower_energy_smeared),
                                         tower_energy_smeared)


        ######## 7. Compute track sigma per Tower ########
        # C++:
        #   sigma = fResolutionFormula->Eval(0.0, fTowerEta, 0.0, momentum.E());
        #   if(sigma / momentum.E() < track->TrackResolution)
        #       energyGuess = energy;  // energy = momentum.E() * fraction
        #   else
        #       energyGuess = momentum.E();
        #   fTrackSigma += (track->TrackResolution * energyGuess) * (track->TrackResolution * energyGuess);
        #   ...
        #   fTrackSigma = TMath::Sqrt(fTrackSigma);  // in FinalizeTower
        
        # Get track resolution from MomentumSmearing (stored in TRACK_RESOLUTION column)
        track_momentum_resolution = tracks[:, CMAP["TRACK_RESOLUTION"]]
        n_tracks = tracks.shape[0]
        
        # For valid tracks, get the tower eta using the track's tower assignment
        # Only tracks with fraction > 1e-9 contribute to fTrackSigma
        track_tower_eta = torch.zeros(n_tracks, dtype=torch.float64, device=tracks.device)
        track_calo_sigma = torch.zeros(n_tracks, dtype=torch.float64, device=tracks.device)
        track_energy_guess = torch.zeros(n_tracks, dtype=torch.float64, device=tracks.device)
        track_sigma_sq = torch.zeros(n_tracks, dtype=torch.float64, device=tracks.device)
        
        # Mask for tracks that contribute to track sigma (valid AND has fraction)
        track_sigma_valid = track_valid & track_has_fraction
        
        # Get tower eta for each valid track from the tower mapping
        for i in range(n_towers):
            tower_mask = (track_compact_idx == i)
            if tower_mask.any():
                track_tower_eta[tower_mask] = tower_eta[i]
        
        # Compute calorimeter sigma at tower eta using track energy
        # C++: sigma = fResolutionFormula->Eval(0.0, fTowerEta, 0.0, momentum.E())
        track_calo_sigma[track_sigma_valid] = self.resolution_func(
            track_tower_eta[track_sigma_valid],
            track_energy[track_sigma_valid]
        )
        
        # Determine energy_guess based on resolution comparison
        # C++: if(sigma / momentum.E() < track->TrackResolution) energyGuess = energy; else energyGuess = momentum.E()
        calo_relative_sigma = track_calo_sigma / (track_energy + 1e-30)  # Avoid div by zero
        use_weighted_energy = calo_relative_sigma < track_momentum_resolution
        
        # energy_guess = fraction * track_energy if calo resolution better, else track_energy
        track_energy_guess = torch.where(
            use_weighted_energy & track_sigma_valid,
            track_weighted_energy,  # energy = momentum.E() * fraction
            torch.where(track_sigma_valid, track_energy, torch.zeros_like(track_energy))
        )
        
        # Compute per-track sigma squared: (track_resolution * energy_guess)^2
        track_sigma_sq = (track_momentum_resolution * track_energy_guess) ** 2
        
        # Aggregate track_sigma_sq per tower using scatter_add
        tower_track_sigma_sq = torch.zeros(n_towers, dtype=torch.float64, device=tracks.device)
        tower_track_sigma_sq.scatter_add_(
            0,
            track_compact_idx[track_sigma_valid],
            track_sigma_sq[track_sigma_valid]
        )
        
        # Final tower track sigma = sqrt(sum of squares)
        tower_track_sigma = torch.sqrt(tower_track_sigma_sq)


        ######## 8. Identify Neutral Excess and Create eflow objects ########
        # C++:
        #   neutralEnergy = max((energy - fTrackEnergy), 0.0);
        #   neutralSigma = neutralEnergy / TMath::Sqrt(fTrackSigma * fTrackSigma + sigma * sigma);
        #   
        #   if(neutralEnergy > fEnergyMin && neutralSigma > fEnergySignificanceMin) {
        #       // Create EFlowTower with neutralEnergy
        #       // Clone tracks to EFlowTrack unchanged
        #   } else if(fTrackEnergy > 0.0) {
        #       // Rescale tracks based on weighted average of calo and track measurements
        #       weightTrack = 1 / (fTrackSigma^2)
        #       weightCalo = 1 / (sigma^2)
        #       bestEnergyEstimate = (weightTrack * fTrackEnergy + weightCalo * energy) / (weightTrack + weightCalo)
        #       rescaleFactor = bestEnergyEstimate / fTrackEnergy
        #       // Clone tracks to EFlowTrack with rescaled pT
        #   }
        
        # Use final (thresholded) tower energy and sigma_after for this computation
        energy = tower_energy_final  # After smearing and thresholds
        sigma = sigma_after
        
        # Compute neutral energy per tower
        neutral_energy = torch.clamp(energy - tower_track_energy, min=0.0)
        
        # Compute neutral sigma per tower
        # neutralSigma = neutralEnergy / sqrt(trackSigma² + sigma²)
        denominator = torch.sqrt(tower_track_sigma**2 + sigma**2)
        neutral_sigma = torch.where(
            denominator > 0,
            neutral_energy / denominator,
            torch.zeros_like(neutral_energy)
        )
        
        # Case A: Neutral excess is significant
        # Condition: neutralEnergy > EnergyMin AND neutralSigma > EnergySignificanceMin
        significant_neutral = (neutral_energy > self.energy_min) & (neutral_sigma > self.energy_sig_min)
        
        # Case B: Neutral excess is NOT significant but has track energy
        # Condition: NOT significant_neutral AND tower_track_energy > 0
        rescale_tracks = (~significant_neutral) & (tower_track_energy > 0)
        
        # Compute rescale factor for Case B towers
        # weightTrack = 1 / (trackSigma^2), weightCalo = 1 / (sigma^2)
        # bestEnergyEstimate = (weightTrack * trackEnergy + weightCalo * energy) / (weightTrack + weightCalo)
        weight_track = torch.where(
            tower_track_sigma > 0,
            1.0 / (tower_track_sigma**2),
            torch.zeros_like(tower_track_sigma)
        )
        weight_calo = torch.where(
            sigma > 0,
            1.0 / (sigma**2),
            torch.zeros_like(sigma)
        )
        
        total_weight = weight_track + weight_calo
        best_energy_estimate = torch.where(
            total_weight > 0,
            (weight_track * tower_track_energy + weight_calo * energy) / total_weight,
            tower_track_energy  # Fallback to track energy if no weights
        )
        
        rescale_factor = torch.where(
            tower_track_energy > 0,
            best_energy_estimate / tower_track_energy,
            torch.ones_like(tower_track_energy)
        )
        
        # ===== Create Tower output =====
        # Towers with energy > 0 after thresholds
        tower_has_energy = tower_energy_final > 0
        
        # Tower output tensor: [PT, Eta, Phi, E, Eem, Ehad, T, Edges...]
        # For ECal: Eem = energy, Ehad = 0
        tower_pt = tower_energy_final / torch.cosh(tower_eta)
        
        # Build tower output (towers with energy > 0)
        n_valid_towers = tower_has_energy.sum().item()
        
        # ===== Create EFlowTower output (neutral excess) =====
        # Only for towers with significant neutral excess
        n_eflow_towers = significant_neutral.sum().item()
        
        eflow_tower_energy = neutral_energy[significant_neutral]
        eflow_tower_eta = tower_eta[significant_neutral]
        eflow_tower_phi = tower_phi[significant_neutral]
        eflow_tower_pt = eflow_tower_energy / torch.cosh(eflow_tower_eta)
        
        # ===== Create EFlowTrack output =====
        # Tracks are output in two cases:
        # 1. Significant neutral: clone track unchanged
        # 2. Rescale: apply rescale factor to track momentum
        
        # For each track, determine which case applies based on its tower
        # track_compact_idx maps tracks to tower indices
        
        # Get the tower status for each track
        # Only tracks with valid sigma (track_sigma_valid) are in towers
        track_in_significant_tower = torch.zeros(n_tracks, dtype=torch.bool, device=tracks.device)
        track_in_rescale_tower = torch.zeros(n_tracks, dtype=torch.bool, device=tracks.device)
        track_rescale_factor = torch.ones(n_tracks, dtype=torch.float64, device=tracks.device)
        
        # Map tower properties to tracks
        for i in range(n_towers):
            tower_mask = (track_compact_idx == i) & track_sigma_valid
            if tower_mask.any():
                if significant_neutral[i]:
                    track_in_significant_tower[tower_mask] = True
                elif rescale_tracks[i]:
                    track_in_rescale_tower[tower_mask] = True
                    track_rescale_factor[tower_mask] = rescale_factor[i]
        
        # Tracks that become EFlowTracks: either in significant tower OR in rescale tower
        track_is_eflow = track_in_significant_tower | track_in_rescale_tower
        
        # Also include tracks with fraction < 1e-9 (they go directly to EFlowTrack in C++)
        # These are tracks that are valid but don't contribute to tower energy
        track_no_fraction = track_valid & (~track_has_fraction)
        track_is_eflow = track_is_eflow | track_no_fraction
        
        # Create EFlowTrack tensor
        # Clone the track and apply rescale factor if applicable
        eflow_tracks = tracks.clone()
        
        # Apply rescale factor to tracks in rescale towers
        # PT is rescaled, then PX, PY, PZ, E are recomputed
        original_pt = eflow_tracks[:, CMAP["PT"]]
        rescaled_pt = original_pt * track_rescale_factor
        
        # Only apply to tracks in rescale towers
        eflow_tracks[:, CMAP["PT"]] = torch.where(
            track_in_rescale_tower,
            rescaled_pt,
            original_pt
        )
        
        # Recompute PX, PY from rescaled PT
        eta = eflow_tracks[:, CMAP["ETA"]]
        phi = eflow_tracks[:, CMAP["PHI"]]
        mass = eflow_tracks[:, CMAP["MASS"]]
        
        eflow_tracks[:, CMAP["PX"]] = torch.where(
            track_in_rescale_tower,
            rescaled_pt * torch.cos(phi),
            eflow_tracks[:, CMAP["PX"]]
        )
        eflow_tracks[:, CMAP["PY"]] = torch.where(
            track_in_rescale_tower,
            rescaled_pt * torch.sin(phi),
            eflow_tracks[:, CMAP["PY"]]
        )
        eflow_tracks[:, CMAP["PZ"]] = torch.where(
            track_in_rescale_tower,
            rescaled_pt * torch.sinh(eta),
            eflow_tracks[:, CMAP["PZ"]]
        )
        
        # Recompute E from P and mass
        p_sq = eflow_tracks[:, CMAP["PX"]]**2 + eflow_tracks[:, CMAP["PY"]]**2 + eflow_tracks[:, CMAP["PZ"]]**2
        eflow_tracks[:, CMAP["E"]] = torch.where(
            track_in_rescale_tower,
            torch.sqrt(p_sq + mass**2),
            eflow_tracks[:, CMAP["E"]]
        )
        
        # Filter to only EFlow tracks
        eflow_track_output = eflow_tracks[track_is_eflow]
        
        # Set PASS_EFLOW_TRACK mask for all output eflow tracks
        if eflow_track_output.shape[0] > 0:
            eflow_track_output[:, CMAP["PASS_EFLOW_TRACK"]] = 1.0

        # ===== Create Tower Tensor with COLUMN_MAP format =====
        # Tower tensor: (n_valid_towers, N_FEATURES)
        r_calo = 1.29  # meters, CMS ECAL radius
        tower_tensor = torch.zeros(n_valid_towers, N_FEATURES, dtype=torch.float64, device=particles.device)
        
        if n_valid_towers > 0:
            valid_tower_energy = tower_energy_final[tower_has_energy]
            valid_tower_eta = tower_eta[tower_has_energy]
            valid_tower_phi = tower_phi[tower_has_energy]
            valid_tower_pt = tower_pt[tower_has_energy]
            
            # Set tower properties
            tower_tensor[:, CMAP["PID"]] = 22.0  # Photon for ECAL towers
            tower_tensor[:, CMAP["STATUS"]] = 1.0
            tower_tensor[:, CMAP["CHARGE"]] = 0.0
            tower_tensor[:, CMAP["E"]] = valid_tower_energy
            tower_tensor[:, CMAP["PT"]] = valid_tower_pt
            tower_tensor[:, CMAP["ETA"]] = valid_tower_eta
            tower_tensor[:, CMAP["PHI"]] = valid_tower_phi
            
            # Compute PX, PY, PZ from PT, ETA, PHI (massless)
            tower_tensor[:, CMAP["PX"]] = valid_tower_pt * torch.cos(valid_tower_phi)
            tower_tensor[:, CMAP["PY"]] = valid_tower_pt * torch.sin(valid_tower_phi)
            tower_tensor[:, CMAP["PZ"]] = valid_tower_pt * torch.sinh(valid_tower_eta)
            
            # Position (approximate at calorimeter surface)
            tower_tensor[:, CMAP["X"]] = r_calo * torch.cos(valid_tower_phi) * 1000  # mm
            tower_tensor[:, CMAP["Y"]] = r_calo * torch.sin(valid_tower_phi) * 1000  # mm
            tower_tensor[:, CMAP["Z"]] = r_calo * torch.sinh(valid_tower_eta) * 1000  # mm
            tower_tensor[:, CMAP["T"]] = 0.0  # TODO: Time weighted average
            tower_tensor[:, CMAP["MASS"]] = 0.0
            
            # Outer position same as momentum direction
            tower_tensor[:, CMAP["ETA_OUTER"]] = valid_tower_eta
            tower_tensor[:, CMAP["PHI_OUTER"]] = valid_tower_phi
            
            # Set masks
            tower_tensor[:, CMAP["IS_NOT_PAD"]] = 1.0
            tower_tensor[:, CMAP["PASS_ECAL_TOWER"]] = 1.0

        # ===== Create EFlowPhoton Tensor with COLUMN_MAP format =====
        # EFlowPhoton tensor: (n_eflow_towers, N_FEATURES) 
        # These are towers with significant neutral excess
        eflow_photon_tensor = torch.zeros(n_eflow_towers, N_FEATURES, dtype=torch.float64, device=particles.device)
        
        if n_eflow_towers > 0:
            # Set eflow photon properties
            eflow_photon_tensor[:, CMAP["PID"]] = 22.0  # Photon
            eflow_photon_tensor[:, CMAP["STATUS"]] = 1.0
            eflow_photon_tensor[:, CMAP["CHARGE"]] = 0.0
            eflow_photon_tensor[:, CMAP["E"]] = eflow_tower_energy
            eflow_photon_tensor[:, CMAP["PT"]] = eflow_tower_pt
            eflow_photon_tensor[:, CMAP["ETA"]] = eflow_tower_eta
            eflow_photon_tensor[:, CMAP["PHI"]] = eflow_tower_phi
            
            # Compute PX, PY, PZ from PT, ETA, PHI (massless)
            eflow_photon_tensor[:, CMAP["PX"]] = eflow_tower_pt * torch.cos(eflow_tower_phi)
            eflow_photon_tensor[:, CMAP["PY"]] = eflow_tower_pt * torch.sin(eflow_tower_phi)
            eflow_photon_tensor[:, CMAP["PZ"]] = eflow_tower_pt * torch.sinh(eflow_tower_eta)
            
            # Position (approximate at calorimeter surface)
            eflow_photon_tensor[:, CMAP["X"]] = r_calo * torch.cos(eflow_tower_phi) * 1000  # mm
            eflow_photon_tensor[:, CMAP["Y"]] = r_calo * torch.sin(eflow_tower_phi) * 1000  # mm
            eflow_photon_tensor[:, CMAP["Z"]] = r_calo * torch.sinh(eflow_tower_eta) * 1000  # mm
            eflow_photon_tensor[:, CMAP["T"]] = 0.0
            eflow_photon_tensor[:, CMAP["MASS"]] = 0.0
            
            # Outer position same as momentum direction
            eflow_photon_tensor[:, CMAP["ETA_OUTER"]] = eflow_tower_eta
            eflow_photon_tensor[:, CMAP["PHI_OUTER"]] = eflow_tower_phi
            
            # Set masks
            eflow_photon_tensor[:, CMAP["IS_NOT_PAD"]] = 1.0
            eflow_photon_tensor[:, CMAP["PASS_EFLOW_PHOTON"]] = 1.0

        # Return results
        return {
            # Steps 1-3: Fractions and binning
            'particle_energy_fractions': particle_energy_fractions,
            'track_energy_fractions': track_energy_fractions,
            'particle_eta_bin': particle_eta_bin,
            'particle_phi_bin': particle_phi_bin,
            'particle_valid': particle_valid,
            'track_eta_bin': track_eta_bin,
            'track_phi_bin': track_phi_bin,
            'track_valid': track_valid,
            # Step 4: Tower aggregation
            'n_towers': n_towers,
            'unique_tower_idx': unique_tower_idx,
            'tower_eta_bin': tower_eta_bin,
            'tower_phi_bin': tower_phi_bin,
            'tower_energy': tower_energy,
            'tower_track_energy': tower_track_energy,
            'max_phi_bins': max_phi_bins,
            # Step 5: Tower centers and edges
            'tower_eta': tower_eta,
            'tower_phi': tower_phi,
            'tower_eta_lo': tower_eta_lo,
            'tower_eta_hi': tower_eta_hi,
            'tower_phi_lo': tower_phi_lo,
            'tower_phi_hi': tower_phi_hi,
            # Step 6: Resolution smearing
            'sigma_before': sigma_before,
            'tower_energy_smeared': tower_energy_smeared,
            'sigma_after': sigma_after,
            'tower_energy_final': tower_energy_final,
            # Step 7: Track sigma
            'tower_track_sigma': tower_track_sigma,
            'track_momentum_resolution': track_momentum_resolution,
            'track_tower_eta': track_tower_eta,
            'track_calo_sigma': track_calo_sigma,
            'track_energy_guess': track_energy_guess,
            'track_sigma_sq': track_sigma_sq,
            'track_sigma_valid': track_sigma_valid,
            'track_compact_idx': track_compact_idx,
            'track_energy': track_energy,
            # Step 8: EFlow outputs
            'neutral_energy': neutral_energy,
            'neutral_sigma': neutral_sigma,
            'significant_neutral': significant_neutral,
            'rescale_tracks': rescale_tracks,
            'rescale_factor': rescale_factor,
            # Final outputs
            'tower_output': {
                'energy': tower_energy_final[tower_has_energy],
                'eta': tower_eta[tower_has_energy],
                'phi': tower_phi[tower_has_energy],
                'pt': tower_pt[tower_has_energy],
                'eta_lo': tower_eta_lo[tower_has_energy],
                'eta_hi': tower_eta_hi[tower_has_energy],
                'phi_lo': tower_phi_lo[tower_has_energy],
                'phi_hi': tower_phi_hi[tower_has_energy],
            },
            'eflow_tower_output': {
                'energy': eflow_tower_energy,
                'eta': eflow_tower_eta,
                'phi': eflow_tower_phi,
                'pt': eflow_tower_pt,
            },
            'eflow_track_output': eflow_track_output,
            'n_valid_towers': n_valid_towers,
            'n_eflow_towers': n_eflow_towers,
            'n_eflow_tracks': eflow_track_output.shape[0],
            # COLUMN_MAP format tensors for ROOT output
            'tower_tensor': tower_tensor,
            'eflow_photon_tensor': eflow_photon_tensor,
            'eflow_track_tensor': eflow_track_output,
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

    @staticmethod
    def _hcal_cms_resolution(eta: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """
        HCAL resolution formula from delphes_card_CMS_5_1.tcl.
        
        Formula:
            |eta| <= 3.0: sqrt(E^2*0.050^2 + E*1.50^2)
            3.0 < |eta| <= 5.0: sqrt(E^2*0.130^2 + E*2.70^2)
        """
        abs_eta = torch.abs(eta)
        
        # Central: |eta| <= 3.0
        central_sigma = torch.sqrt(energy**2 * 0.050**2 + energy * 1.50**2)
        
        # Forward: 3.0 < |eta| <= 5.0
        forward_sigma = torch.sqrt(energy**2 * 0.130**2 + energy * 2.70**2)
        
        # Select based on eta region
        sigma = torch.where(abs_eta <= 3.0, central_sigma, forward_sigma)
        
        return sigma

    @staticmethod
    def _log_normal_smear(mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Apply log-normal smearing to energy values.
        
        C++ implementation:
            if(mean > 0.0) {
                b = sqrt(log((1.0 + (sigma * sigma) / (mean * mean))));
                a = log(mean) - 0.5 * b * b;
                return exp(a + b * gRandom->Gaus(0.0, 1.0));
            } else {
                return 0.0;
            }
        
        Args:
            mean: Tower energy (before smearing)
            sigma: Resolution sigma
            
        Returns:
            Smeared energy values
        """
        # For mean > 0, apply log-normal
        # For mean <= 0, return 0
        
        # Avoid division by zero
        safe_mean = torch.where(mean > 0, mean, torch.ones_like(mean))
        
        # b = sqrt(log(1 + sigma^2/mean^2))
        b = torch.sqrt(torch.log(1.0 + (sigma * sigma) / (safe_mean * safe_mean)))
        
        # a = log(mean) - 0.5 * b^2
        a = torch.log(safe_mean) - 0.5 * b * b
        
        # Sample from standard normal
        z = torch.randn_like(mean)
        
        # exp(a + b * z)
        smeared = torch.exp(a + b * z)
        
        # Zero out where mean <= 0
        smeared = torch.where(mean > 0, smeared, torch.zeros_like(smeared))
        
        return smeared
