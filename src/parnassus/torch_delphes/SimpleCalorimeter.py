"""
PyTorch implementation of Delphes SimpleCalorimeter module.

Implements calorimeter tower binning, energy smearing, and energy flow object creation.
This module:
1. Bins particles into eta-phi calorimeter towers
2. Applies energy resolution smearing
3. Creates energy flow objects (eflowTracks and eflowPhotons)
"""
import torch
import torch.nn as nn
import numpy as np

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP


class SimpleCalorimeter(nn.Module):
    """
    PyTorch implementation of Delphes SimpleCalorimeter module.
    
    Simulates electromagnetic or hadronic calorimeter response by:
    - Binning particles into eta-phi towers
    - Summing energy deposits with PDG-based energy fractions
    - Applying energy resolution smearing
    - Creating energy flow objects by comparing track vs calorimeter energy
    
    Input: genevent_tensors with shape (N_events, N_particles, D)
           Must have masks: IS_NOT_PAD, PASS_PROP, PASS_MERGER
    
    Output: genevent_tensors with shape (N_events, N_particles + N_towers, D+3)
            New masks: PASS_ECAL_TOWER, PASS_EFLOW_TRACK, PASS_EFLOW_PHOTON
            
            Dictionary with:
            - 'ecalTowers': List of tower tensors per event
            - 'eflowTracks': List of energy flow track tensors per event  
            - 'eflowPhotons': List of energy flow photon tensors per event
    """
    
    def __init__(self,
                 eta_bins,
                 phi_bins, 
                 energy_min=0.5,
                 energy_sig_min=2.0,
                 resolution_formula='ecal_cms',
                 is_ecal=True,
                 smear_tower_center=True,
                 energy_fractions=None,
                 max_towers_per_event=500,
                 device='cpu'):
        """
        Args:
            eta_bins: List of eta bin edges (sorted)
            phi_bins: List of phi bin edges (sorted)
            energy_min: Minimum energy for tower to be saved (GeV)
            energy_sig_min: Minimum energy significance (E/sigma) for tower
            resolution_formula: Energy resolution formula name or callable
            is_ecal: If True, this is ECAL; if False, HCAL
            smear_tower_center: If True, dither tower center position
            energy_fractions: Dict mapping PDG IDs to energy fractions (default: essentials)
            max_towers_per_event: Maximum number of towers per event for padding
            device: torch device
        """
        super().__init__()
        self.device = device
        self.energy_min = energy_min
        self.energy_sig_min = energy_sig_min
        self.is_ecal = is_ecal
        self.smear_tower_center = smear_tower_center
        self.max_towers_per_event = max_towers_per_event
        
        # Store bin edges as tensors
        self.eta_bins = torch.tensor(eta_bins, dtype=torch.float64, device=device)
        self.phi_bins = torch.tensor(phi_bins, dtype=torch.float64, device=device)
        
        # Energy fractions: default essentials (e/gamma/pi0=1.0, muons/neutrinos=0.0, hadrons=0.3)
        if energy_fractions is None:
            energy_fractions = {
                0: 1.0,      # default
                11: 1.0,     # electron
                -11: 1.0,    # positron
                22: 1.0,     # photon
                111: 1.0,    # pi0
                12: 0.0,     # nu_e
                14: 0.0,     # nu_mu
                16: 0.0,     # nu_tau
                13: 0.0,     # muon
                -13: 0.0,    # antimuon
                310: 0.3,    # K0short
                3122: 0.3,   # Lambda
            }
        self.energy_fractions = energy_fractions
        
        # Load resolution formula
        if resolution_formula == 'ecal_cms':
            self.resolution_fn = self._ecal_cms_resolution
        elif resolution_formula == 'hcal_cms':
            self.resolution_fn = self._hcal_cms_resolution
        elif callable(resolution_formula):
            self.resolution_fn = resolution_formula
        else:
            raise ValueError(f"Unknown resolution formula: {resolution_formula}")
    
    @staticmethod
    def _ecal_cms_resolution(energy, eta):
        """
        CMS ECAL energy resolution formula.
        From delphes_card_CMS_5_0.tcl
        """
        abs_eta = torch.abs(eta)
        
        # Barrel: |eta| <= 1.5
        barrel_mask = abs_eta <= 1.5
        barrel_res = (1.0 + 0.64 * eta**2) * torch.sqrt(
            energy**2 * 0.008**2 + energy * 0.11**2 + 0.40**2
        )
        
        # Endcap: 1.5 < |eta| <= 2.5
        endcap_mask = (abs_eta > 1.5) & (abs_eta <= 2.5)
        endcap_res = (2.16 + 5.6 * (abs_eta - 2.0)**2) * torch.sqrt(
            energy**2 * 0.008**2 + energy * 0.11**2 + 0.40**2
        )
        
        # HF: 2.5 < |eta| <= 5.0
        hf_mask = (abs_eta > 2.5) & (abs_eta <= 5.0)
        hf_res = torch.sqrt(energy**2 * 0.107**2 + energy * 2.08**2)
        
        resolution = torch.zeros_like(energy)
        resolution = torch.where(barrel_mask, barrel_res, resolution)
        resolution = torch.where(endcap_mask, endcap_res, resolution)
        resolution = torch.where(hf_mask, hf_res, resolution)
        
        return resolution
    
    @staticmethod
    def _hcal_cms_resolution(energy, eta):
        """
        CMS HCAL energy resolution formula (placeholder).
        """
        # Simple parametrization for HCAL
        return torch.sqrt(energy**2 * 0.1**2 + energy * 0.5**2 + 1.0**2)
    
    @staticmethod
    def log_normal_sample(mean, sigma):
        """
        Sample from log-normal distribution to ensure positive energies.
        Same as used in MomentumSmearing module.
        """
        # Avoid log(0) issues
        valid_mask = mean > 0.0
        
        result = torch.zeros_like(mean)
        
        if valid_mask.any():
            mean_valid = mean[valid_mask]
            sigma_valid = sigma[valid_mask]
            
            b = torch.sqrt(torch.log(1.0 + (sigma_valid**2) / (mean_valid**2)))
            a = torch.log(mean_valid) - 0.5 * b**2
            
            result[valid_mask] = torch.exp(a + b * torch.randn_like(a))
        
        return result
    
    def get_energy_fraction(self, pid):
        """
        Get energy fraction for a given PDG ID.
        """
        abs_pid = torch.abs(pid).long()
        
        # Create output tensor
        fractions = torch.ones_like(pid, dtype=torch.float64)
        
        # Apply fractions based on PDG ID
        for pdg_id, fraction in self.energy_fractions.items():
            if pdg_id == 0:  # default
                continue
            mask = abs_pid == abs(pdg_id)
            fractions[mask] = fraction
        
        return fractions
    
    def forward(self, genevent_tensors):
        """
        Apply SimpleCalorimeter to generate calorimeter towers and energy flow objects.
        
        Args:
            genevent_tensors: (N_events, N_particles, D) tensor with masks
            
        Returns:
            genevent_tensors_out: (N_events, N_particles + N_towers, D+3) 
            outputs: Dict with 'ecalTowers', 'eflowTracks', 'eflowPhotons' lists
        """
        genevent_tensors = genevent_tensors.to(self.device)
        n_events, n_particles, n_dim = genevent_tensors.shape
        
        # Initialize output lists
        all_towers = []
        all_eflow_tracks = []
        all_eflow_photons = []
        
        # Process each event independently (towers are event-specific)
        for event_idx in range(n_events):
            event_tensor = genevent_tensors[event_idx]  # (N_particles, D)
            
            # Process this event
            towers, eflow_tracks, eflow_photons = self._process_event(event_tensor)
            
            all_towers.append(towers)
            all_eflow_tracks.append(eflow_tracks)
            all_eflow_photons.append(eflow_photons)
        
        # Now we need to append towers to genevent_tensors and add masks
        # Pad towers to max_towers_per_event and concatenate
        genevent_tensors_out = self._append_towers_to_events(
            genevent_tensors, all_towers, all_eflow_tracks, all_eflow_photons
        )
        
        outputs = {
            'ecalTowers': all_towers,
            'eflowTracks': all_eflow_tracks,
            'eflowPhotons': all_eflow_photons
        }
        
        return genevent_tensors_out, outputs
    
    def _process_event(self, event_tensor):
        """
        Process a single event to create towers and energy flow objects.
        
        Args:
            event_tensor: (N_particles, D) tensor for one event
            
        Returns:
            towers: (N_towers, D) tensor of tower objects
            eflow_tracks: (N_eflow_tracks, D) tensor of eflow track objects
            eflow_photons: (N_eflow_photons, D) tensor of eflow photon objects
        """
        # Extract valid particles (propagated) and tracks (merged)
        valid_particles_mask = (
            event_tensor[:, CMAP["IS_NOT_PAD"]] * 
            event_tensor[:, CMAP["PASS_PROP"]]
        ).bool()
        
        valid_tracks_mask = (
            event_tensor[:, CMAP["IS_NOT_PAD"]] *
            event_tensor[:, CMAP["PASS_PROP"]] *
            event_tensor[:, CMAP["PASS_MERGER"]]
        ).bool()
        
        # Get particle and track subsets
        particles = event_tensor[valid_particles_mask]  # All particles after propagation
        tracks = event_tensor[valid_tracks_mask]  # Tracks (charged particles that passed all filters)
        
        if particles.shape[0] == 0:
            # No particles in this event
            empty = torch.zeros((0, event_tensor.shape[1]), dtype=torch.float64, device=self.device)
            return empty, empty, empty
        
        # Bin particles into towers
        particle_eta = particles[:, CMAP["ETA"]]
        particle_phi = particles[:, CMAP["PHI"]]
        particle_energy = particles[:, CMAP["E"]]
        particle_pid = particles[:, CMAP["PID"]]
        particle_position = particles[:, CMAP["X"]:CMAP["Z"]+1]  # X, Y, Z
        particle_time = particles[:, CMAP["T"]]
        
        # Find bin indices for particles
        particle_eta_bin = torch.searchsorted(self.eta_bins, particle_eta, right=False) - 1
        particle_phi_bin = torch.searchsorted(self.phi_bins, particle_phi, right=False) - 1
        
        # Filter out particles outside bins
        valid_bin_mask = (
            (particle_eta_bin >= 0) & (particle_eta_bin < len(self.eta_bins) - 1) &
            (particle_phi_bin >= 0) & (particle_phi_bin < len(self.phi_bins) - 1)
        )
        
        if not valid_bin_mask.any():
            empty = torch.zeros((0, event_tensor.shape[1]), dtype=torch.float64, device=self.device)
            return empty, empty, empty
        
        # Apply bin filter
        particles = particles[valid_bin_mask]
        particle_eta_bin = particle_eta_bin[valid_bin_mask]
        particle_phi_bin = particle_phi_bin[valid_bin_mask]
        particle_energy = particle_energy[valid_bin_mask]
        particle_pid = particle_pid[valid_bin_mask]
        particle_position = particle_position[valid_bin_mask]
        particle_time = particle_time[valid_bin_mask]
        
        # Get energy fractions
        energy_fractions = self.get_energy_fraction(particle_pid)
        particle_energy_deposited = particle_energy * energy_fractions
        
        # Create unique tower IDs (eta_bin * n_phi_bins + phi_bin)
        n_phi_bins = len(self.phi_bins) - 1
        tower_ids = particle_eta_bin * n_phi_bins + particle_phi_bin
        
        # Get unique tower IDs
        unique_tower_ids = torch.unique(tower_ids)
        n_towers = len(unique_tower_ids)
        
        # Aggregate energy per tower using scatter_add
        tower_energies = torch.zeros(n_towers, dtype=torch.float64, device=self.device)
        tower_times = torch.zeros(n_towers, dtype=torch.float64, device=self.device)
        tower_time_weights = torch.zeros(n_towers, dtype=torch.float64, device=self.device)
        
        # Map tower_ids to indices
        tower_id_to_idx = {tid.item(): idx for idx, tid in enumerate(unique_tower_ids)}
        particle_tower_idx = torch.tensor(
            [tower_id_to_idx[tid.item()] for tid in tower_ids],
            dtype=torch.long, device=self.device
        )
        
        # Accumulate energy and time
        tower_energies.scatter_add_(0, particle_tower_idx, particle_energy_deposited)
        
        # Time weighted by E^2 (sigma_t ~ 1/E)
        time_contribution = particle_energy_deposited**2 * particle_time
        tower_times.scatter_add_(0, particle_tower_idx, time_contribution)
        tower_time_weights.scatter_add_(0, particle_tower_idx, particle_energy_deposited**2)
        
        # Compute average time per tower
        tower_times = torch.where(
            tower_time_weights > 1e-9,
            tower_times / tower_time_weights,
            torch.zeros_like(tower_times)
        )
        
        # Get tower eta/phi from bin indices
        tower_eta_bins = unique_tower_ids // n_phi_bins
        tower_phi_bins = unique_tower_ids % n_phi_bins
        
        # Tower center positions
        tower_eta_center = 0.5 * (self.eta_bins[tower_eta_bins] + self.eta_bins[tower_eta_bins + 1])
        tower_phi_center = 0.5 * (self.phi_bins[tower_phi_bins] + self.phi_bins[tower_phi_bins + 1])
        
        # Optionally smear tower centers
        if self.smear_tower_center:
            tower_eta = torch.rand(n_towers, dtype=torch.float64, device=self.device) * \
                        (self.eta_bins[tower_eta_bins + 1] - self.eta_bins[tower_eta_bins]) + \
                        self.eta_bins[tower_eta_bins]
            tower_phi = torch.rand(n_towers, dtype=torch.float64, device=self.device) * \
                        (self.phi_bins[tower_phi_bins + 1] - self.phi_bins[tower_phi_bins]) + \
                        self.phi_bins[tower_phi_bins]
        else:
            tower_eta = tower_eta_center
            tower_phi = tower_phi_center
        
        # Apply energy smearing
        tower_sigma = self.resolution_fn(tower_energies, tower_eta)
        tower_energies_smeared = self.log_normal_sample(tower_energies, tower_sigma)
        
        # Recompute sigma after smearing
        tower_sigma = self.resolution_fn(tower_energies_smeared, tower_eta)
        
        # Apply energy threshold (for towers, only require energy_min)
        # Significance threshold is only for eflow photons
        significant_mask = tower_energies_smeared >= self.energy_min
        
        # Filter towers
        if not significant_mask.any():
            empty = torch.zeros((0, event_tensor.shape[1]), dtype=torch.float64, device=self.device)
            return empty, empty, empty
        
        tower_energies_smeared = tower_energies_smeared[significant_mask]
        tower_eta = tower_eta[significant_mask]
        tower_phi = tower_phi[significant_mask]
        tower_times = tower_times[significant_mask]
        tower_sigma = tower_sigma[significant_mask]
        unique_tower_ids = unique_tower_ids[significant_mask]
        n_towers = len(tower_energies_smeared)
        
        # Now handle track-tower matching for energy flow
        # For each tower, find tracks in the same tower
        eflow_tracks_list = []
        eflow_photons_list = []
        towers_list = []
        
        for tower_idx in range(n_towers):
            tid = unique_tower_ids[tower_idx]
            tower_energy = tower_energies_smeared[tower_idx]
            tower_sigma_val = tower_sigma[tower_idx]
            
            # Find tracks in this tower
            track_mask = valid_tracks_mask.clone()
            if tracks.shape[0] > 0:
                track_eta = tracks[:, CMAP["ETA"]]
                track_phi = tracks[:, CMAP["PHI"]]
                track_eta_bin = torch.searchsorted(self.eta_bins, track_eta, right=False) - 1
                track_phi_bin = torch.searchsorted(self.phi_bins, track_phi, right=False) - 1
                track_tower_ids = track_eta_bin * n_phi_bins + track_phi_bin
                
                tower_tracks_mask = track_tower_ids == tid
                tower_tracks = tracks[tower_tracks_mask]
            else:
                tower_tracks = torch.zeros((0, event_tensor.shape[1]), dtype=torch.float64, device=self.device)
            
            # Compute total track energy and track resolution
            if tower_tracks.shape[0] > 0:
                track_energies = tower_tracks[:, CMAP["E"]]
                track_resolutions = tower_tracks[:, CMAP["PT"]] * 0.01  # Approximate track resolution
                
                # Get energy fractions for tracks
                track_pids = tower_tracks[:, CMAP["PID"]]
                track_energy_fractions = self.get_energy_fraction(track_pids)
                track_energies_deposited = track_energies * track_energy_fractions
                
                total_track_energy = track_energies_deposited.sum()
                total_track_sigma = torch.sqrt((track_resolutions * track_energies)**2).sum()
            else:
                total_track_energy = 0.0
                total_track_sigma = 0.0
            
            # Compute neutral energy
            neutral_energy = max(tower_energy - total_track_energy, 0.0)
            
            # Compute neutral significance
            if total_track_sigma**2 + tower_sigma_val**2 > 0:
                neutral_sigma = neutral_energy / torch.sqrt(
                    torch.tensor(total_track_sigma**2 + tower_sigma_val**2, device=self.device)
                )
            else:
                neutral_sigma = 0.0
            
            # Always create tower for ECalTower output if energy > threshold
            if tower_energy > self.energy_min:
                tower_obj = self._create_tower_object(
                    tower_eta[tower_idx], tower_phi[tower_idx],
                    tower_energy, tower_times[tower_idx],
                    event_tensor.shape[1]
                )
                towers_list.append(tower_obj)
            
            # Energy flow logic for creating eflow objects
            if neutral_energy > self.energy_min and neutral_sigma > self.energy_sig_min:
                # Significant neutral energy - create eflow photon
                neutral_tower = self._create_tower_object(
                    tower_eta[tower_idx], tower_phi[tower_idx],
                    neutral_energy, tower_times[tower_idx],
                    event_tensor.shape[1]
                )
                eflow_photons_list.append(neutral_tower)
                
                # Pass tracks through unchanged (they coexist with neutral energy)
                for track in tower_tracks:
                    eflow_tracks_list.append(track.unsqueeze(0))
            
            elif total_track_energy > 0.0:
                # No significant neutral energy, but we have tracks
                # Rescale tracks to match best energy estimate
                weight_track = 1.0 / total_track_sigma**2 if total_track_sigma > 0 else 0.0
                weight_calo = 1.0 / tower_sigma_val**2 if tower_sigma_val > 0 else 0.0
                
                if weight_track + weight_calo > 0:
                    best_energy = (weight_track * total_track_energy + weight_calo * tower_energy) / \
                                  (weight_track + weight_calo)
                    rescale_factor = best_energy / total_track_energy
                else:
                    rescale_factor = 1.0
                
                # Rescale and add tracks as eflow tracks
                for track in tower_tracks:
                    rescaled_track = track.clone()
                    rescaled_track[CMAP["PT"]] *= rescale_factor
                    rescaled_track[CMAP["E"]] *= rescale_factor
                    rescaled_track[CMAP["PX"]] *= rescale_factor
                    rescaled_track[CMAP["PY"]] *= rescale_factor
                    rescaled_track[CMAP["PZ"]] *= rescale_factor
                    eflow_tracks_list.append(rescaled_track.unsqueeze(0))
        
        # Concatenate results
        if len(towers_list) > 0:
            towers = torch.cat(towers_list, dim=0)
        else:
            towers = torch.zeros((0, event_tensor.shape[1]), dtype=torch.float64, device=self.device)
        
        if len(eflow_tracks_list) > 0:
            eflow_tracks = torch.cat(eflow_tracks_list, dim=0)
        else:
            eflow_tracks = torch.zeros((0, event_tensor.shape[1]), dtype=torch.float64, device=self.device)
        
        if len(eflow_photons_list) > 0:
            eflow_photons = torch.cat(eflow_photons_list, dim=0)
        else:
            eflow_photons = torch.zeros((0, event_tensor.shape[1]), dtype=torch.float64, device=self.device)
        
        return towers, eflow_tracks, eflow_photons
    
    def _create_tower_object(self, eta, phi, energy, time, n_features):
        """
        Create a tower object as a particle-like tensor.
        """
        tower = torch.zeros(1, n_features, dtype=torch.float64, device=self.device)
        
        # Set tower properties
        tower[0, CMAP["PID"]] = 22 if self.is_ecal else 0  # Photon for ECAL, neutral for HCAL
        tower[0, CMAP["STATUS"]] = 1
        tower[0, CMAP["CHARGE"]] = 0.0
        tower[0, CMAP["E"]] = energy
        
        # Momentum (massless)
        pt = energy / torch.cosh(eta)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        
        tower[0, CMAP["PT"]] = pt
        tower[0, CMAP["ETA"]] = eta
        tower[0, CMAP["PHI"]] = phi
        tower[0, CMAP["PX"]] = px
        tower[0, CMAP["PY"]] = py
        tower[0, CMAP["PZ"]] = pz
        
        # Position (approximate from eta/phi at calorimeter surface)
        # For simplicity, use R=1.29m (tracker radius) as reference
        r = 1.29  # meters
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)
        z = r * torch.sinh(eta)
        
        tower[0, CMAP["X"]] = x * 1000  # convert to mm
        tower[0, CMAP["Y"]] = y * 1000
        tower[0, CMAP["Z"]] = z * 1000
        tower[0, CMAP["T"]] = time
        tower[0, CMAP["MASS"]] = 0.0
        
        return tower
    
    def _append_towers_to_events(self, genevent_tensors, all_towers, 
                                 all_eflow_tracks, all_eflow_photons):
        """
        Append towers to genevent_tensors and add masks.
        
        Pad to (N_particles + max_towers_per_event) per event.
        """
        n_events, n_particles, n_dim = genevent_tensors.shape
        
        # New dimension includes 3 new mask columns
        n_dim_out = n_dim + 3
        max_size = n_particles + self.max_towers_per_event
        
        # Initialize output tensor
        genevent_tensors_out = torch.zeros(
            (n_events, max_size, n_dim_out),
            dtype=torch.float64, device=self.device
        )
        
        # Copy original particle data (first n_particles entries per event)
        genevent_tensors_out[:, :n_particles, :n_dim] = genevent_tensors
        
        # Process each event
        for event_idx in range(n_events):
            towers = all_towers[event_idx]
            eflow_tracks = all_eflow_tracks[event_idx]
            eflow_photons = all_eflow_photons[event_idx]
            
            n_towers = towers.shape[0]
            
            # Append towers after particles
            if n_towers > 0:
                tower_start = n_particles
                tower_end = min(n_particles + n_towers, max_size)
                actual_towers = tower_end - tower_start
                
                genevent_tensors_out[event_idx, tower_start:tower_end, :n_dim] = \
                    towers[:actual_towers, :n_dim]
                
                # Set tower masks
                genevent_tensors_out[event_idx, tower_start:tower_end, CMAP["IS_NOT_PAD"]] = 1.0
                genevent_tensors_out[event_idx, tower_start:tower_end, CMAP["PASS_ECAL_TOWER"]] = 1.0
            
            # Mark eflow tracks
            if eflow_tracks.shape[0] > 0:
                # Find matching particles in original tensor and mark them
                # For simplicity, we'll create a new mask based on particle IDs
                # (This is a simplified approach; production code might need particle tracking)
                pass
            
            # Mark eflow photons  
            if eflow_photons.shape[0] > 0:
                # These are actually the towers that had neutral excess
                # Already marked above with PASS_ECAL_TOWER
                pass
        
        return genevent_tensors_out


# Example usage and testing
if __name__ == "__main__":
    print("Testing SimpleCalorimeter PyTorch Module\n")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create example event
    n_events = 2
    n_particles = 50
    n_dim = 19  # After Merger module
    
    genevent_tensors = torch.zeros((n_events, n_particles, n_dim), dtype=torch.float64)
    
    # Fill with example particles
    for event_idx in range(n_events):
        n_real = np.random.randint(20, 40)
        
        # Create mix of particles
        for i in range(n_real):
            # Random kinematics
            pt = np.random.uniform(1.0, 50.0)
            eta = np.random.uniform(-2.5, 2.5)
            phi = np.random.uniform(-np.pi, np.pi)
            
            # Random PID (electrons, photons, charged hadrons)
            pid = np.random.choice([11, 22, 211])
            
            genevent_tensors[event_idx, i, CMAP["PID"]] = pid
            genevent_tensors[event_idx, i, CMAP["CHARGE"]] = 1.0 if abs(pid) == 211 or abs(pid) == 11 else 0.0
            genevent_tensors[event_idx, i, CMAP["PT"]] = pt
            genevent_tensors[event_idx, i, CMAP["ETA"]] = eta
            genevent_tensors[event_idx, i, CMAP["PHI"]] = phi
            genevent_tensors[event_idx, i, CMAP["E"]] = pt * np.cosh(eta)
            
            # Set masks
            genevent_tensors[event_idx, i, CMAP["IS_NOT_PAD"]] = 1.0
            genevent_tensors[event_idx, i, CMAP["PASS_PROP"]] = 1.0
            
            # Some are tracks
            if i < n_real // 2:
                genevent_tensors[event_idx, i, CMAP["PASS_MERGER"]] = 1.0
    
    # Create SimpleCalorimeter module
    # CMS-like binning (simplified)
    eta_bins = np.linspace(-2.5, 2.5, 50)
    phi_bins = np.linspace(-np.pi, np.pi, 50)
    
    ecal = SimpleCalorimeter(
        eta_bins=eta_bins,
        phi_bins=phi_bins,
        energy_min=0.5,
        energy_sig_min=2.0,
        resolution_formula='ecal_cms',
        is_ecal=True,
        max_towers_per_event=100,
        device='cpu'
    )
    
    print("Input shape:", genevent_tensors.shape)
    
    # Process
    genevent_out, outputs = ecal(genevent_tensors)
    
    print("\nOutput shape:", genevent_out.shape)
    print("\nNumber of towers per event:")
    for i, towers in enumerate(outputs['ecalTowers']):
        print(f"  Event {i}: {towers.shape[0]} towers")
    
    print("\nNumber of eflow tracks per event:")
    for i, tracks in enumerate(outputs['eflowTracks']):
        print(f"  Event {i}: {tracks.shape[0]} eflow tracks")
    
    print("\nNumber of eflow photons per event:")
    for i, photons in enumerate(outputs['eflowPhotons']):
        print(f"  Event {i}: {photons.shape[0]} eflow photons")
    
    print("\n✓ SimpleCalorimeter test completed!")
