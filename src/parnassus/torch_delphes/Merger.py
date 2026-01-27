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
from parnassus.torch_delphes.Efficiency import Efficiency


class Merger(nn.Module):
    """
    PyTorch implementation of Delphes Merger module.
    
    This module:
    1. Filters particles based on PID (which particle types to include)
    2. Creates a PASS_MERGER mask column (AND of previous masks + PID filter)
    3. Optionally computes aggregate statistics per event
    
    For TrackMerger, this combines charged hadrons, electrons, and muons
    that have passed propagation, efficiency, and momentum smearing.
    
    Input shape: (N_events, N_particles, N_FEATURES)
        Must contain IS_NOT_PAD, PASS_PROP, PASS_EFF mask columns
    
    Output shape: (N_events, N_particles, D) with filled-in PASS_MERGER column
    """
    
    def __init__(
        self, 
        particle_types: List[str] = ['charged_hadron', 'electron', 'muon'],
        device: str = 'cpu'
    ) -> None:
        """
        Args:
            particle_types: List of particle types to include in merger
                           Options: 'charged_hadron', 'electron', 'muon', 'neutral'
            device: torch device ('cpu' or 'cuda')
        """
        super().__init__()
        self.device = device
        self.particle_types = particle_types
        
        # Map particle types to their PDG filter functions
        self.pdg_filters = {
            'charged_hadron': Efficiency._charged_hadron_pdg_filter,
            'electron': Efficiency._electron_pdg_filter,
            'muon': Efficiency._muon_pdg_filter,
            'neutral': Efficiency._neutral_pdg_filter
        }
    
    def forward(self, genevent_tensors: torch.Tensor) -> torch.Tensor:
        """
        Apply merger to create unified output with PASS_MERGER mask.
        
        Args:
            genevent_tensors: tensor of shape (N_events, N_particles, D)
                Must have columns: IS_NOT_PAD, PASS_PROP, PASS_EFF
                
        Returns:
            genevent_tensors: tensor of shape (N_events, N_particles, D+1)
                with new PASS_MERGER mask column appended
        """
        # Move to device
        genevent_tensors = genevent_tensors.to(self.device)
        
        # Extract dimensions
        n_events, n_particles, n_dim = genevent_tensors.shape
        
        # Get existing masks - particles that passed all previous stages
        valid_mask = (
            genevent_tensors[:, :, CMAP["IS_NOT_PAD"]] *
            genevent_tensors[:, :, CMAP["PASS_PROP"]] *
            genevent_tensors[:, :, CMAP["PASS_EFF"]]
        )
        
        # Apply PID filters to select particle types for this merger
        combined_pid_mask = torch.zeros(n_events, n_particles, device=self.device)
        
        for particle_type in self.particle_types:
            if particle_type in self.pdg_filters:
                # Apply PDG filter for this particle type
                # Note: PDG filters expect (N, D) or (B, N, D) shaped input
                pid_mask = self.pdg_filters[particle_type](genevent_tensors)
                combined_pid_mask = combined_pid_mask + pid_mask
        
        # Clamp to 0-1 range (in case particle matches multiple filters)
        combined_pid_mask = combined_pid_mask.clamp(max=1.0)
        
        # Create PASS_MERGER mask: valid particles that match PID filter
        pass_merger_mask = valid_mask * combined_pid_mask
        
        # Fill in on PASS_MERGER column
        genevent_tensors[:, :, CMAP["PASS_MERGER"]] = pass_merger_mask

        return genevent_tensors
    
    def compute_aggregate_stats(self, genevent_tensors: torch.Tensor) -> torch.Tensor:
        """
        Compute per-event aggregate statistics for particles that pass merger.
        
        This matches the C++ Delphes Merger outputs:
        - MomentumOutputArray: vector sum of 4-momenta
        - EnergyOutputArray: scalar sums of PT and E
        
        Args:
            genevent_tensors: tensor with PASS_MERGER column
            
        Returns:
            List of dicts, one per event, containing:
                - sum_px, sum_py, sum_pz, sum_e: vector sum of 4-momentum
                - sum_pt: scalar sum of transverse momentum
                - scalar_sum_e: scalar sum of energy
                - n_tracks: number of particles in merged output
        """
        genevent_tensors = genevent_tensors.to(self.device)
        n_events = genevent_tensors.shape[0]
        
        stats = []
        
        for event_idx in range(n_events):
            # Get mask for this event
            merger_mask = genevent_tensors[event_idx, :, CMAP["PASS_MERGER"]] > 0.5
            
            # Extract particles that passed merger
            event_particles = genevent_tensors[event_idx]
            
            if merger_mask.sum() > 0:
                # Get momentum components
                px = event_particles[merger_mask, CMAP["PX"]]
                py = event_particles[merger_mask, CMAP["PY"]]
                pz = event_particles[merger_mask, CMAP["PZ"]]
                e = event_particles[merger_mask, CMAP["E"]]
                pt = event_particles[merger_mask, CMAP["PT"]]
                
                # Compute sums
                event_stats = {
                    'sum_px': px.sum().item(),
                    'sum_py': py.sum().item(),
                    'sum_pz': pz.sum().item(),
                    'sum_e': e.sum().item(),
                    'sum_pt': pt.sum().item(),
                    'n_tracks': merger_mask.sum().item()
                }
            else:
                # Empty event
                event_stats = {
                    'sum_px': 0.0,
                    'sum_py': 0.0,
                    'sum_pz': 0.0,
                    'sum_e': 0.0,
                    'sum_pt': 0.0,
                    'n_tracks': 0
                }
            
            stats.append(event_stats)
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("Testing Delphes Merger PyTorch Module\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create example events with multiple particle types
    n_events = 5
    n_particles = 20
    n_dim = 18  # After Efficiency module (includes IS_NOT_PAD, PASS_PROP, PASS_EFF)
    
    genevent_tensors = torch.zeros((n_events, n_particles, n_dim), dtype=torch.float64)
    
    # Fill with example particles
    for event_idx in range(n_events):
        n_real_particles = np.random.randint(10, 18)
        
        for i in range(n_real_particles):
            # Randomly assign particle types
            particle_type = np.random.choice(['charged_hadron', 'electron', 'muon', 'photon'])
            
            if particle_type == 'charged_hadron':
                genevent_tensors[event_idx, i, CMAP["PID"]] = 211  # pi+
                genevent_tensors[event_idx, i, CMAP["CHARGE"]] = 1
            elif particle_type == 'electron':
                genevent_tensors[event_idx, i, CMAP["PID"]] = 11  # e-
                genevent_tensors[event_idx, i, CMAP["CHARGE"]] = -1
            elif particle_type == 'muon':
                genevent_tensors[event_idx, i, CMAP["PID"]] = 13  # mu-
                genevent_tensors[event_idx, i, CMAP["CHARGE"]] = -1
            else:  # photon
                genevent_tensors[event_idx, i, CMAP["PID"]] = 22
                genevent_tensors[event_idx, i, CMAP["CHARGE"]] = 0
            
            # Set kinematics
            pt = np.random.uniform(1, 50)
            eta = np.random.uniform(-2.5, 2.5)
            phi = np.random.uniform(-np.pi, np.pi)
            
            genevent_tensors[event_idx, i, CMAP["PT"]] = pt
            genevent_tensors[event_idx, i, CMAP["ETA"]] = eta
            genevent_tensors[event_idx, i, CMAP["PHI"]] = phi
            genevent_tensors[event_idx, i, CMAP["PX"]] = pt * np.cos(phi)
            genevent_tensors[event_idx, i, CMAP["PY"]] = pt * np.sin(phi)
            genevent_tensors[event_idx, i, CMAP["PZ"]] = pt * np.sinh(eta)
            genevent_tensors[event_idx, i, CMAP["E"]] = np.sqrt(
                genevent_tensors[event_idx, i, CMAP["PX"]]**2 +
                genevent_tensors[event_idx, i, CMAP["PY"]]**2 +
                genevent_tensors[event_idx, i, CMAP["PZ"]]**2
            )
            
            # Set masks (all particles passed previous stages)
            genevent_tensors[event_idx, i, CMAP["IS_NOT_PAD"]] = 1.0
            genevent_tensors[event_idx, i, CMAP["PASS_PROP"]] = 1.0
            genevent_tensors[event_idx, i, CMAP["PASS_EFF"]] = 1.0
    
    print(f"Input shape: {genevent_tensors.shape}")
    print(f"Number of events: {n_events}")
    print(f"Max particles per event: {n_particles}")
    
    # Count particles by type before merger
    print("\nParticles before merger:")
    for event_idx in range(n_events):
        valid_mask = genevent_tensors[event_idx, :, CMAP["IS_NOT_PAD"]] > 0.5
        pids = genevent_tensors[event_idx, valid_mask, CMAP["PID"]]
        n_ch = (torch.abs(pids) == 211).sum().item()
        n_el = (torch.abs(pids) == 11).sum().item()
        n_mu = (torch.abs(pids) == 13).sum().item()
        n_photon = (pids == 22).sum().item()
        print(f"  Event {event_idx}: CH={n_ch}, El={n_el}, Mu={n_mu}, Photon={n_photon}")
    
    # Create TrackMerger (charged hadrons, electrons, muons only)
    print("\n" + "="*70)
    print("Applying TrackMerger")
    print("="*70)
    
    merger = Merger(
        particle_types=['charged_hadron', 'electron', 'muon'],
        device='cpu'
    )
    
    # Apply merger
    genevent_tensors_merged = merger(genevent_tensors)
    
    print(f"\nOutput shape: {genevent_tensors_merged.shape}")
    print(f"New dimension added: PASS_MERGER at column {CMAP['PASS_MERGER']}")
    
    # Count particles after merger
    print("\nParticles after merger (tracks only):")
    for event_idx in range(n_events):
        merger_mask = genevent_tensors_merged[event_idx, :, CMAP["PASS_MERGER"]] > 0.5
        pids = genevent_tensors_merged[event_idx, merger_mask, CMAP["PID"]]
        n_ch = (torch.abs(pids) == 211).sum().item()
        n_el = (torch.abs(pids) == 11).sum().item()
        n_mu = (torch.abs(pids) == 13).sum().item()
        n_total = merger_mask.sum().item()
        print(f"  Event {event_idx}: Total tracks={n_total} (CH={n_ch}, El={n_el}, Mu={n_mu})")
    
    # Compute aggregate statistics
    print("\n" + "="*70)
    print("Computing aggregate statistics")
    print("="*70)
    
    stats = merger.compute_aggregate_stats(genevent_tensors_merged)
    
    for event_idx, event_stats in enumerate(stats):
        print(f"\nEvent {event_idx}:")
        print(f"  N tracks: {event_stats['n_tracks']}")
        print(f"  Sum PT: {event_stats['sum_pt']:.2f} GeV")
        print(f"  Sum E: {event_stats['sum_e']:.2f} GeV")
        print(f"  Sum 4-momentum: ({event_stats['sum_px']:.2f}, "
              f"{event_stats['sum_py']:.2f}, {event_stats['sum_pz']:.2f}, "
              f"{event_stats['sum_e']:.2f})")
    
    # Verify photons are excluded
    print("\n" + "="*70)
    print("Verification: Photons should be excluded from tracks")
    print("="*70)
    
    for event_idx in range(n_events):
        valid_mask = genevent_tensors[event_idx, :, CMAP["IS_NOT_PAD"]] > 0.5
        merger_mask = genevent_tensors_merged[event_idx, :, CMAP["PASS_MERGER"]] > 0.5
        
        # Check if any photons passed merger
        pids_merged = genevent_tensors_merged[event_idx, merger_mask, CMAP["PID"]]
        n_photons_merged = (pids_merged == 22).sum().item()
        
        if n_photons_merged == 0:
            print(f"  Event {event_idx}: ✓ No photons in merged tracks")
        else:
            print(f"  Event {event_idx}: ✗ WARNING: {n_photons_merged} photons in merged tracks!")
    
    print("\n" + "="*70)
    print("✓ Merger test completed successfully!")
    print("="*70)
