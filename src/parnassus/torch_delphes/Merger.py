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
    
# Example usage and testing
if __name__ == "__main__":
    print("Testing Delphes Merger PyTorch Module\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create example events with multiple particle types
    n_events = 5
    n_particles = 20
    n_dim = 18  # After Efficiency module (includes IS_NOT_PAD, PASS_PROP)
    
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
    )
    
    # Apply merger
    genevent_tensors_merged = merger(genevent_tensors)
    
    print(f"\nOutput shape: {genevent_tensors_merged.shape}")

