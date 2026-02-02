"""
PyTorch implementation of Delphes Tower module.

Implements calorimeter tower binning, energy smearing, and energy flow object creation.
This module:
1. Bins particles into eta-phi calorimeter towers
2. Applies energy resolution smearing
3. Creates energy flow objects (eflowTracks and eflowPhotons)
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable, Union, Tuple

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP
from parnassus.torch_delphes.stochastic_utils import log_normal_sample
from parnassus.torch_delphes.SimpleCalorimeter.calorimeter_resolution import ecal_cms_resolution, hcal_cms_resolution


class Tower(nn.Module):
    """
    PyTorch implementation of Delphes SimpleCalorimeter.Tower module.
    
    Simulates electromagnetic or hadronic calorimeter response by:
    - Binning particles into eta-phi towers
    - Summing energy deposits with PDG-based energy fractions
    - Applying energy resolution smearing
    - Creating energy flow objects by comparing track vs calorimeter energy
    
    Input: genevent_tensors with shape (N_events, N_particles, D)
           Must have masks: IS_NOT_PAD, PASS_PROP
    
    Output: genevent_tensors with shape (N_events, N_particles + N_towers, D+3)
            New masks: PASS_ECAL_TOWER, PASS_EFLOW_TRACK, PASS_EFLOW_PHOTON
            
            ecalTowers: List of tower tensors per event
    """
    
    def __init__(
        self,
        eta_bins: List[float],
        phi_bins: List[float], 
        energy_min: float = 0.5,
        energy_sig_min: float = 2.0,
        resolution_formula: Union[str, Callable] = 'ecal_cms',
        is_ecal: bool = True,
        smear_tower_center: bool = True,
        energy_fractions: Optional[Dict[int, float]] = None,
        max_towers_per_event: int = 500,
    ) -> None:
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
        """
        super().__init__()
        self.energy_min = energy_min
        self.energy_sig_min = energy_sig_min
        self.is_ecal = is_ecal
        self.smear_tower_center = smear_tower_center
        self.max_towers_per_event = max_towers_per_event
        
        # Store bin edges as tensors
        self.eta_bins = torch.tensor(eta_bins, dtype=torch.float64)
        self.phi_bins = torch.tensor(phi_bins, dtype=torch.float64)

        self.first_in = False
        
        # Energy fractions: default essentials (e/gamma/pi0=1.0, muons/neutrinos=0.0, hadrons=0.3)
        if energy_fractions is None:
            # default CMS energy fractions
            energy_fractions = {
                0: 0.0,      # default
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
                1000022: 0.0,   # neutralino
                1000023: 0.0,   # neutralino2
                1000025: 0.0,   # neutralino3
                1000035: 0.0,   # neutralino4
                1000045: 0.0,    # neutralino5
                -1000022: 0.0,   # neutralino
                -1000023: 0.0,   # neutralino2
                -1000025: 0.0,   # neutralino3
                -1000035: 0.0,   # neutralino4
                -1000045: 0.0,    # neutralino5
            }
        self.energy_fractions = energy_fractions
        
        # Load resolution formula
        if resolution_formula == 'ecal_cms':
            self.resolution_fn = ecal_cms_resolution
        elif resolution_formula == 'hcal_cms':
            self.resolution_fn = hcal_cms_resolution
        elif callable(resolution_formula):
            self.resolution_fn = resolution_formula
        else:
            raise ValueError(f"Unknown resolution formula: {resolution_formula}")
    
    def forward(self, pap_tensors: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply Tower to generate calorimeter towers and energy flow objects.
        
        Args:
            pap_tensors: (N_events, N_particles, D) tensor with masks
            
        Returns:
            genevent_tensors_out: (N_events, N_particles + N_towers, D+3) 
            outputs: Dict with 'ecalTowers', 'eflowTracks', 'eflowPhotons' lists
        """

        if not self.first_in:
            # Move bin edges to the same device as input tensors
            device = pap_tensors.device
            self.eta_bins = self.eta_bins.to(device)
            self.phi_bins = self.phi_bins.to(device)
            self.first_in = True

        event_numbers = set(pap_tensors[:, CMAP["EVENT_NUMBER"]].cpu().numpy().tolist())
        
        # Initialize output lists
        all_towers = []

        # Process each event independently (towers are event-specific)
        for event_num in event_numbers:
            particles_event = pap_tensors[pap_tensors[:, CMAP["EVENT_NUMBER"]] == event_num]
            towers_event = self._process_event(particles_event)
            all_towers.append(towers_event)

        return all_towers
  
    def _process_event(self, particles: torch.Tensor) -> List[torch.Tensor]:
        """
        Process a single event to create towers and energy flow objects.
        
        Args:
            particles: (N_particles, D) tensor for one event
            
        Returns:
            towers: (N_towers, D) tensor of tower objects
        """
        
        # Bin particles into towers 
        particle_eta = particles[:, CMAP["ETA_OUTER"]]
        particle_phi = particles[:, CMAP["PHI_OUTER"]]
        particle_energy = particles[:, CMAP["E"]]
        particle_pid = particles[:, CMAP["PID"]]
        particle_time = particles[:, CMAP["T"]]
        
        # Find bin indices for particles # <-- SUSPECT
        particle_eta_bin = torch.searchsorted(self.eta_bins, particle_eta, right=False) - 1
        particle_phi_bin = torch.searchsorted(self.phi_bins, particle_phi, right=False) - 1
        
        # Filter out particles outside bins
        valid_bin_mask = (
            (particle_eta_bin >= 0) & (particle_eta_bin < len(self.eta_bins) - 1) &
            (particle_phi_bin >= 0) & (particle_phi_bin < len(self.phi_bins) - 1)
        )
        
        # Apply bin filter
        particle_eta_bin = particle_eta_bin[valid_bin_mask]
        particle_phi_bin = particle_phi_bin[valid_bin_mask]
        particle_energy = particle_energy[valid_bin_mask]
        particle_pid = particle_pid[valid_bin_mask]
        particle_time = particle_time[valid_bin_mask]
        
        # Get energy fractions
        energy_fractions = self.get_energy_fraction(particle_pid)
        particle_energy_deposited = particle_energy * energy_fractions
        
        # Filter out particles that deposit zero energy (muons, neutrinos, etc.)
        nonzero_energy_mask = particle_energy_deposited > 0.0
        particle_eta_bin = particle_eta_bin[nonzero_energy_mask]
        particle_phi_bin = particle_phi_bin[nonzero_energy_mask]
        particle_energy = particle_energy[nonzero_energy_mask]
        particle_pid = particle_pid[nonzero_energy_mask]
        particle_time = particle_time[nonzero_energy_mask]
        particle_energy_deposited = particle_energy_deposited[nonzero_energy_mask]
        
        # Create unique tower IDs (eta_bin * n_phi_bins + phi_bin)
        n_phi_bins = len(self.phi_bins) - 1
        tower_ids = particle_eta_bin * n_phi_bins + particle_phi_bin
        
        # Get unique tower IDs
        unique_tower_ids = torch.unique(tower_ids)
        n_towers = len(unique_tower_ids)
        
        # Aggregate energy per tower using scatter_add
        tower_energies = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        tower_times = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)
        tower_time_weights = torch.zeros(n_towers, dtype=torch.float64, device=particles.device)

        # Map tower_ids to indices
        tower_id_to_idx = {tid.item(): idx for idx, tid in enumerate(unique_tower_ids)}
        particle_tower_idx = torch.tensor(
            [tower_id_to_idx[tid.item()] for tid in tower_ids],
            dtype=torch.long, device=particles.device
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
        tower_phi_bins = unique_tower_ids % n_phi_bins # SUSPECT
        
        # Tower center positions
        tower_eta_center = 0.5 * (self.eta_bins[tower_eta_bins] + self.eta_bins[tower_eta_bins + 1])
        tower_phi_center = 0.5 * (self.phi_bins[tower_phi_bins] + self.phi_bins[tower_phi_bins + 1]) # SUSPECT
        
        # Optionally smear tower centers (TODO: Make optional)
        tower_eta = torch.rand(n_towers, dtype=torch.float64, device=particles.device) * \
                    (self.eta_bins[tower_eta_bins + 1] - self.eta_bins[tower_eta_bins]) + \
                    self.eta_bins[tower_eta_bins]
        tower_phi = torch.rand(n_towers, dtype=torch.float64, device=particles.device) * \
                    (self.phi_bins[tower_phi_bins + 1] - self.phi_bins[tower_phi_bins]) + \
                    self.phi_bins[tower_phi_bins] # SUSPECT

        
        # Apply energy smearing
        tower_sigma = self.resolution_fn(tower_energies, tower_eta)
        tower_energies_smeared = log_normal_sample(tower_energies, tower_sigma)
        
        # Recompute sigma after smearing
        tower_sigma = self.resolution_fn(tower_energies_smeared, tower_eta)
                
        # Now handle track-tower matching for energy flow
        # For each tower, find tracks in the same tower
        towers_list = []
        for tower_idx in range(n_towers):
            tower_energy = tower_energies_smeared[tower_idx]
            tower_sigma_val = tower_sigma[tower_idx]
            
            # Apply significance threshold to tower energy (C++ line 436)
            # if(energy < fEnergyMin || energy < fEnergySignificanceMin * sigma) energy = 0.0;
            if tower_energy < self.energy_min or tower_energy < self.energy_sig_min * tower_sigma_val:
                tower_energy = 0.0  # Zero the tower energy like C++ does
            
            # Create tower object for ECalTower output if energy > 0
            if tower_energy > 0.0: # DEF SUSPECT
                tower_obj = self._create_tower_object(
                    tower_eta[tower_idx], tower_phi[tower_idx],
                    tower_energy, tower_times[tower_idx],
                    particles.shape[1]
                )
                towers_list.append(tower_obj)       
        
        # Concatenate results
        towers = torch.cat(towers_list, dim=0)
        
        return towers
   
    def _create_tower_object(
        self, 
        eta: float, 
        phi: float, 
        energy: float, 
        time: float, 
        n_features: int
    ) -> torch.Tensor:
        """
        Create a tower object as a particle-like tensor.
        """
        tower = torch.zeros(1, n_features, dtype=torch.float64, device=eta.device)
        
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

    def get_energy_fraction(self, pid: int) -> float:
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
     
