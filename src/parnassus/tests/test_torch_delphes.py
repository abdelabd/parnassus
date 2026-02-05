"""
Apply PyTorch Delphes modules (ParticlePropagator, Efficiency, MomentumSmearing, Merger, SimpleCalorimeter) to HepMC file and save outputs.

This is a redesigned version that uses pure tensor operations:
- HepMC→ Tensor conversion happens once at the beginning
- All processing happens in tensor space
- Tensor → ROOT conversion happens once per output file

Compares against C++ Delphes with delphes_card_CMS_5_0.tcl (includes ECal/SimpleCalorimeter).

Usage:
    python test_torch_delphes.py [--input FILE] [--output FILE] [--benchmark FILE]
    
Default:
    Input:  delphes_data/HZZ4l/HZZ4l_0.hepmc
    Output: delphes_data/HZZ4l/HZZ4l_5_0_torch.root
    Benchmark: delphes_data/HZZ4l/HZZ4l_5_0.root
"""
from typing import List, Tuple, Dict, Optional
import sys
import os
import random
import time
from pathlib import Path
from tqdm import tqdm
import argparse 

import torch
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd

# Set PyTorch to use maximum precision (double precision / float64)
torch.set_default_dtype(torch.float64)

# Seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from parnassus.torch_delphes import Efficiency, Merger, MomentumSmearing, ParticlePropagator, SimpleCalorimeter
from parnassus.torch_delphes.tensor_utils import (
    hepmc_to_tensor,
    tensor_to_root_dict,
    write_root_file,
    COLUMN_MAP as CMAP
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def process_particle_propagator(
    genevent_tensors: torch.Tensor, 
    batch_size: int = 100
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Apply ParticlePropagator to GenEvent tensors using batched processing.
    
    Args:
        genevent_tensors: Tensor of shape N_EVENT x N_PARTICLES x 15
                - Should be zero-padded such that all tensors have the same shape (number of particles)
        batch_size: Number of events to process in each batch
        
    Returns:
        genevent_tensors: Tensor of shape N_EVENT x N_PARTICLES x 16
        pbp_tensors: List of tensors for all particles BEFORE propagation
        pap_tensors: List of tensors for all particles AFTER propagation
        ch_tensors: List of charged hadron tensors
        el_tensors: List of electron tensors  
        mu_tensors: List of muon tensors
    """

    n_event, n_part, n_dim = genevent_tensors.shape
    
    # Initialize ParticlePropagator module
    propagator = ParticlePropagator(
        radius=1.29,        # CMS tracker radius in meters
        half_length=3.0,    # CMS tracker half-length in meters  
        bz=3.8,             # Magnetic field in Tesla
    ).to(DEVICE)
    
    print(f"\nParticlePropagator (batch_size={batch_size})...")
    print(f"genevent_tensors.shape: {genevent_tensors.shape}")

    genevent_tensors_propagated = torch.zeros(genevent_tensors.shape, dtype=genevent_tensors.dtype).to(DEVICE)
    # Collect particle_after_prop, charged_hadron, electron, muon tensors after propagation (for intermediate testing and validation)
    
    pbp_tensors = [] # pbp = particles_before_prop
    pap_tensors = [] # pap = particles_after_prop
    ch_tensors = []
    el_tensors = []
    mu_tensors = []
    # Process in batches
    for batch_start in tqdm(range(0, n_event, batch_size)):
        batch_end = min(batch_start + batch_size, n_event)

        # Flatten to (B*N, N_FEATURES)
        batch_events = genevent_tensors[batch_start:batch_end]
        batch_size_actual = batch_events.shape[0]
        
        # Propagate particles (batched)
        particles = batch_events.reshape(-1, n_dim) # Flatten to (B*N, N_FEATURES)
        particles_before_prop_batch = particles.clone()
        particles_after_prop_batch, _, charged_hadrons_batch, electrons_batch, muons_batch = propagator(particles)

        # Update genevnt_tensors
        genevent_tensors_propagated[batch_start:batch_end] = particles_after_prop_batch.reshape(batch_size_actual, n_part, n_dim)
        
        # For debugging: Collect ParticleBeforeProp, ParticleAfterProp, ChargedHadron, Electron, and Muon tensors after propagation
        
        # ParticleBeforeProp
        pbp_mask = particles_before_prop_batch[:, CMAP["IS_NOT_PAD"]].float()
        pbp_tensors.append(particles_before_prop_batch[pbp_mask > 0.5].clone().to(torch.float32))

        # ParticleAfterProp
        pap_mask = particles_after_prop_batch[:, CMAP["IS_NOT_PAD"]].float() * particles_after_prop_batch[:, CMAP["PASS_PROP"]].float()
        pap_tensors.append(particles_after_prop_batch[pap_mask > 0.5].to(torch.float32))

        # ChargedHadron
        ch_tensors.append(charged_hadrons_batch)

        # Electron
        el_tensors.append(electrons_batch)

        # Muon
        mu_tensors.append(muons_batch)

    return genevent_tensors_propagated, pbp_tensors, pap_tensors, ch_tensors, el_tensors, mu_tensors

def process_efficiency_pipeline(
    ch_tensors: List[torch.Tensor],
    el_tensors: List[torch.Tensor],
    mu_tensors: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Apply tracking efficiency to all three particle types using batched processing.
    
    Args:
        charged_hadron_tensors: List of tensors (one per event)
        electron_tensors: List of tensors (one per event)
        muon_tensors: List of tensors (one per event)
        batch_size: Number of events to process in each batch
        
    Returns:
        Tuple of (ch_filtered, el_filtered, mu_filtered)
        Each is a list of tensors (one per event)
    """
    
    # Initialize efficiency modules
    ch_eff_module = Efficiency(
        efficiency_formula='charged_hadron_cms',
    ).to(DEVICE)
    el_eff_module = Efficiency(
        efficiency_formula='electron_cms',
    ).to(DEVICE)
    mu_eff_module = Efficiency(
        efficiency_formula='muon_cms',
    ).to(DEVICE)

    # Collect charged_hadron, electron, muon tensors after Efficiency
    ch_tensors_eff = []
    el_tensors_eff = []
    mu_tensors_eff = []

    # Process in batches
    for ch_batch_in, el_batch_in, mu_batch_in in tqdm(zip(ch_tensors, el_tensors, mu_tensors), total=len(ch_tensors)):
        ch_batch_out = ch_eff_module(ch_batch_in)
        el_batch_out = el_eff_module(el_batch_in)
        mu_batch_out = mu_eff_module(mu_batch_in)

        ch_tensors_eff.append(ch_batch_out.to(torch.float32))
        el_tensors_eff.append(el_batch_out.to(torch.float32))
        mu_tensors_eff.append(mu_batch_out.to(torch.float32))

    return ch_tensors_eff, el_tensors_eff, mu_tensors_eff

def process_smearing_pipeline(
    ch_tensors: List[torch.Tensor],
    el_tensors: List[torch.Tensor],
    mu_tensors: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Apply momentum smearing to all three particle types using batched processing.
    
    Args:
        ch_filtered: List of filtered charged hadron tensors
        el_filtered: List of filtered electron tensors
        mu_filtered: List of filtered muon tensors
        batch_size: Number of events to process in each batch
        
    Returns:
        Tuple of (ch_smeared, el_smeared, mu_smeared)
        Each is a list of tensors (one per event)
    """
    
    # Initialize smearing modules
    ch_smear_module = MomentumSmearing(
        resolution_formula='charged_hadron_cms',
    ).to(DEVICE)
    
    el_smear_module = MomentumSmearing(
        resolution_formula='electron_cms',
    ).to(DEVICE)
    
    mu_smear_module = MomentumSmearing(
        resolution_formula='muon_cms',
    ).to(DEVICE)

    ch_tensors_smeared = []
    el_tensors_smeared = []
    mu_tensors_smeared = []
    # Process in batches
    for ch_batch_in, el_batch_in, mu_batch_in in tqdm(zip(ch_tensors, el_tensors, mu_tensors), total=len(ch_tensors)):
        ch_batch_out = ch_smear_module(ch_batch_in)
        el_batch_out = el_smear_module(el_batch_in)
        mu_batch_out = mu_smear_module(mu_batch_in)

        ch_tensors_smeared.append(ch_batch_out.to(torch.float32))
        el_tensors_smeared.append(el_batch_out.to(torch.float32))
        mu_tensors_smeared.append(mu_batch_out.to(torch.float32))

    return ch_tensors_smeared, el_tensors_smeared, mu_tensors_smeared

def process_merger_pipeline(
    ch_tensors: List[torch.Tensor],
    el_tensors: List[torch.Tensor],
    mu_tensors: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Apply TrackMerger to combine charged hadrons, electrons, and muons.
    
    Args:
        genevent_tensors: Tensor of shape (N_events, N_particles, D)
        batch_size: Number of events to process in each batch
        
    Returns:
        genevent_tensors: Tensor of shape (N_events, N_particles, D)
        track_tensors: List of track tensors (for validation)
    """
    
    # Initialize TrackMerger module
    merger = Merger().to(DEVICE)
    
    
    track_tensors = []
    # Process in batches
    for ch_batch_in, el_batch_in, mu_batch_in in tqdm(zip(ch_tensors, el_tensors, mu_tensors), total=len(ch_tensors)):
        tracks_batch_out = merger([ch_batch_in, el_batch_in, mu_batch_in])
        track_tensors.append(tracks_batch_out.to(torch.float32))
    
    return track_tensors


def process_ecal_pipeline(
    pap_tensors: List[torch.Tensor],
    merged_tracks: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Apply SimpleCalorimeter Step 1: Compute energy fractions based on PDG ID.
    
    This step validates against fTowerFractions and fTrackFractions in C++ Delphes.
    
    Args:
        pap_tensors: List of particle tensors after propagation (one per event)
        merged_tracks: List of merged track tensors (one per event)
        
    Returns:
        particle_fractions: List of (N_particles,) tensors with energy fractions
        track_fractions: List of (N_tracks,) tensors with energy fractions
    """
    
    energy_fractions = {
        0: 0.0,        # default (hadrons) - no ECAL response
        11: 1.0,       # electrons
        22: 1.0,       # photons
        111: 1.0,      # pi0
        12: 0.0,       # neutrino (electron)
        13: 0.0,       # muon
        14: 0.0,       # neutrino (muon)
        16: 0.0,       # neutrino (tau)
        1000022: 0.0,  # neutralino
        1000023: 0.0,  # neutralino
        1000025: 0.0,  # neutralino
        1000035: 0.0,  # neutralino
        1000045: 0.0,  # neutralino
        310: 0.3,      # K0short
        3122: 0.3,     # Lambda
    }
    
    # Create eta and phi bins from CMS card (delphes_card_CMS_5_0.tcl)
    # The card builds a map: eta_value -> set of phi bins
    # We need to replicate this exactly
    
    # Fine phi bins for barrel and endcap (361 bins, -pi to pi in 1 degree steps)
    phi_bins_fine = [i * np.pi / 180.0 for i in range(-180, 181)]
    
    # Coarse phi bins for HF (37 bins, -pi to pi in 10 degree steps)
    phi_bins_coarse = [i * np.pi / 18.0 for i in range(-18, 19)]
    
    # Build the eta bins and corresponding phi bins exactly as C++ Delphes does
    # The C++ code uses a map<double, set<double>> which gets sorted by eta
    # Then converts to parallel vectors: fEtaBins and fPhiBins[etaBin]
    
    eta_phi_map = {}  # eta -> set of phi bin edges
    
    # Barrel: 0.02 unit in eta from -85*0.0174 to 86*0.0174
    for i in range(-85, 87):
        eta = i * 0.0174
        if eta not in eta_phi_map:
            eta_phi_map[eta] = set()
        eta_phi_map[eta].update(phi_bins_fine)
    
    # Endcap negative: -2.958 + i*0.0174 for i in 1..84
    for i in range(1, 85):
        eta = -2.958 + i * 0.0174
        if eta not in eta_phi_map:
            eta_phi_map[eta] = set()
        eta_phi_map[eta].update(phi_bins_fine)
    
    # Endcap positive: 1.4964 + i*0.0174 for i in 1..84
    for i in range(1, 85):
        eta = 1.4964 + i * 0.0174
        if eta not in eta_phi_map:
            eta_phi_map[eta] = set()
        eta_phi_map[eta].update(phi_bins_fine)
    
    # HF: specific eta values with coarse phi binning
    hf_etas = [-5, -4.7, -4.525, -4.35, -4.175, -4, -3.825, -3.65, -3.475, -3.3, -3.125, -2.958,
               3.125, 3.3, 3.475, 3.65, 3.825, 4, 4.175, 4.35, 4.525, 4.7, 5]
    for eta in hf_etas:
        if eta not in eta_phi_map:
            eta_phi_map[eta] = set()
        eta_phi_map[eta].update(phi_bins_coarse)
    
    # Convert to sorted lists (matching C++ behavior)
    eta_bins = sorted(eta_phi_map.keys())
    phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

    calo = SimpleCalorimeter(
        eta_bins=eta_bins,
        phi_bins=phi_bins_per_eta,
        energy_min=0.5,
        energy_sig_min=2.0,
        energy_fractions=energy_fractions,
        resolution_formula='ecal_cms',
        is_ecal=True
    ).to(DEVICE)
    
    particle_fractions_list = []
    track_fractions_list = []
    particle_eta_bins_list = []
    particle_phi_bins_list = []
    particle_valid_list = []
    track_eta_bins_list = []
    track_phi_bins_list = []
    track_valid_list = []
    tower_results_list = []
    
    # Debug: Check track ETA_OUTER/PHI_OUTER values before processing
    print("\nDEBUG: Checking track ETA_OUTER/PHI_OUTER values...")
    all_tracks = torch.cat(merged_tracks, dim=0)
    track_eta_outer = all_tracks[:, CMAP["ETA_OUTER"]]
    track_phi_outer = all_tracks[:, CMAP["PHI_OUTER"]]
    print(f"  Total tracks: {len(all_tracks)}")
    print(f"  Track ETA_OUTER range: [{track_eta_outer.min().item():.4f}, {track_eta_outer.max().item():.4f}]")
    print(f"  Track PHI_OUTER range: [{track_phi_outer.min().item():.4f}, {track_phi_outer.max().item():.4f}]")
    print(f"  Tracks with ETA_OUTER == 0: {(track_eta_outer == 0).sum().item()}")
    print(f"  Tracks with PHI_OUTER == 0: {(track_phi_outer == 0).sum().item()}")
    print(f"  Tracks with |ETA_OUTER| < 5.0: {(track_eta_outer.abs() < 5.0).sum().item()}")
    print(f"  Tracks with |ETA_OUTER| < 3.0: {(track_eta_outer.abs() < 3.0).sum().item()}")
    
    print("\nSimpleCalorimeter: Computing energy fractions and binning...")
    for batch_particles, batch_tracks in zip(pap_tensors, merged_tracks):
        event_numbers = torch.unique(batch_particles[:, CMAP["EVENT_NUMBER"]]).cpu().numpy()

        # forward method takes a single event, for now
        for event_num in tqdm(event_numbers):
            event_mask_particles = (batch_particles[:, CMAP["EVENT_NUMBER"]] == event_num)
            event_mask_tracks = (batch_tracks[:, CMAP["EVENT_NUMBER"]] == event_num)
            particles = batch_particles[event_mask_particles]
            tracks = batch_tracks[event_mask_tracks]
            
            # Run forward pass
            result = calo(particles, tracks)
            
            # Collect Step 1 results (energy fractions)
            particle_fractions_list.append(result['particle_energy_fractions'].cpu())
            track_fractions_list.append(result['track_energy_fractions'].cpu())
            
            # Collect Step 2 results (binning)
            particle_eta_bins_list.append(result['particle_eta_bin'].cpu())
            particle_phi_bins_list.append(result['particle_phi_bin'].cpu())
            particle_valid_list.append(result['particle_valid'].cpu())
            track_eta_bins_list.append(result['track_eta_bin'].cpu())
            track_phi_bins_list.append(result['track_phi_bin'].cpu())
            track_valid_list.append(result['track_valid'].cpu())
            
            # Collect Step 4 results (tower aggregation)
            tower_results_list.append({
                'n_towers': result['n_towers'],
                'tower_eta_bin': result['tower_eta_bin'].cpu(),
                'tower_phi_bin': result['tower_phi_bin'].cpu(),
                'tower_energy': result['tower_energy'].cpu(),
                'tower_track_energy': result['tower_track_energy'].cpu(),
                'max_phi_bins': result['max_phi_bins'],
                # Step 5: Tower centers and edges
                'tower_eta': result['tower_eta'].cpu(),
                'tower_phi': result['tower_phi'].cpu(),
                'tower_eta_lo': result['tower_eta_lo'].cpu(),
                'tower_eta_hi': result['tower_eta_hi'].cpu(),
                'tower_phi_lo': result['tower_phi_lo'].cpu(),
                'tower_phi_hi': result['tower_phi_hi'].cpu(),
            })
    
    return {
        'particle_fractions': particle_fractions_list,
        'track_fractions': track_fractions_list,
        'particle_eta_bins': particle_eta_bins_list,
        'particle_phi_bins': particle_phi_bins_list,
        'particle_valid': particle_valid_list,
        'track_eta_bins': track_eta_bins_list,
        'track_phi_bins': track_phi_bins_list,
        'track_valid': track_valid_list,
        'tower_results': tower_results_list,
    }

def validate_simple_cal(
    ecal_results: Dict,
    cpp_fractions_file: str,
    cpp_towerhits_file: str,
    cpp_towerenergy_file: str,
    output_dir: str,
) -> None:
    """
    Validate SimpleCalorimeter Steps 1, 2, & 4 against C++ Delphes debug output.
    
    Args:
        ecal_results: Dict with keys from process_ecal()
        cpp_fractions_file: Path to CSV file from C++ Delphes (simplecalo_debug_fractions.csv)
        cpp_towerhits_file: Path to CSV file from C++ Delphes (simplecalo_debug_towerhits.csv)
        output_dir: Directory to save validation plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Step 1: Energy Fractions ==========
    print(f"\n{'='*70}")
    print("Validating SimpleCalorimeter Step 1: Energy Fractions")
    print(f"{'='*70}")
    
    particle_fractions = ecal_results['particle_fractions']
    track_fractions = ecal_results['track_fractions']
    
    # Load C++ debug output
    if not Path(cpp_fractions_file).exists():
        print(f"  ⚠ C++ debug file not found: {cpp_fractions_file}")
        print("  Run C++ Delphes with modified SimpleCalorimeter.cc to generate this file.")
    else:
        cpp_df = pd.read_csv(cpp_fractions_file)
        print(f"  Loaded {len(cpp_df)} entries from C++ debug file")
        
        # Separate particle and track fractions
        cpp_particle_df = cpp_df[cpp_df['type'] == 'particle']
        cpp_track_df = cpp_df[cpp_df['type'] == 'track']
        
        # Flatten TorchDelphes fractions
        torch_particle_fracs = torch.cat(particle_fractions).numpy()
        torch_track_fracs = torch.cat(track_fractions).numpy()
        
        cpp_particle_fracs = cpp_particle_df['fraction'].values
        cpp_track_fracs = cpp_track_df['fraction'].values
        
        print(f"  TorchDelphes particles: {len(torch_particle_fracs)}")
        print(f"  C++ Delphes particles:  {len(cpp_particle_fracs)}")
        print(f"  TorchDelphes tracks:    {len(torch_track_fracs)}")
        print(f"  C++ Delphes tracks:     {len(cpp_track_fracs)}")
        
        # Check if counts match
        if len(torch_particle_fracs) != len(cpp_particle_fracs):
            print(f"  ⚠ Particle count mismatch! TorchDelphes has {len(torch_particle_fracs)}, C++ has {len(cpp_particle_fracs)}")
        if len(torch_track_fracs) != len(cpp_track_fracs):
            print(f"  ⚠ Track count mismatch! TorchDelphes has {len(torch_track_fracs)}, C++ has {len(cpp_track_fracs)}")
        
        # Create comparison plots
        for frac_type, torch_fracs, cpp_fracs in [
            ('Particle', torch_particle_fracs, cpp_particle_fracs),
            ('Track', torch_track_fracs, cpp_track_fracs),
        ]:
            # Create figure with histogram comparison
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Left: Overlaid histograms
            ax = axes[0]
            bins = np.linspace(-0.1, 1.1, 25)
            
            ax.hist(cpp_fracs, bins=bins, histtype='stepfilled', color='orange', alpha=0.5,
                    linewidth=2, label='C++ Delphes', density=False)
            ax.hist(torch_fracs, bins=bins, histtype='step', color='blue',
                    linewidth=2, label='Parnassus.TorchDelphes', density=False)

            ax.set_xlabel('Energy Fraction', fontsize=12)
            ax.set_ylabel('Counts', fontsize=12)
            ax.set_title(f'SimpleCalorimeter Step 1: {frac_type} Energy Fractions', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'TorchDelphes: {len(torch_fracs)} {frac_type.lower()}s\nC++ Delphes: {len(cpp_fracs)} {frac_type.lower()}s'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Right: Per-fraction value comparison (bar chart)
            ax2 = axes[1]
            unique_fracs = sorted(set(cpp_fracs) | set(torch_fracs))
            
            cpp_counts = [np.sum(cpp_fracs == f) for f in unique_fracs]
            torch_counts = [np.sum(torch_fracs == f) for f in unique_fracs]
            
            x = np.arange(len(unique_fracs))
            width = 0.35
            
            ax2.bar(x - width/2, cpp_counts, width, label='C++ Delphes', color='orange', alpha=0.7)
            ax2.bar(x + width/2, torch_counts, width, label='Parnassus.TorchDelphes', color='blue', alpha=0.7)
            
            ax2.set_xlabel('Energy Fraction Value', fontsize=12)
            ax2.set_ylabel('Counts', fontsize=12)
            ax2.set_title(f'{frac_type} Fractions by Value', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{f:.2f}' for f in unique_fracs])
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plot_file = output_dir / f"{frac_type.lower()}_fractions.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")
        
        # Check exact match (if counts are equal)
        if len(torch_particle_fracs) == len(cpp_particle_fracs):
            particle_match = np.allclose(torch_particle_fracs, cpp_particle_fracs, rtol=1e-9)
            print(f"  Particle fractions exact match: {'✓ YES' if particle_match else '✗ NO'}")
        
        if len(torch_track_fracs) == len(cpp_track_fracs):
            track_match = np.allclose(torch_track_fracs, cpp_track_fracs, rtol=1e-9)
            print(f"  Track fractions exact match: {'✓ YES' if track_match else '✗ NO'}")
        
        print(f"  ✓ Step 1 validation complete.")
    
    # ========== Step 2: Binning (Tower Hits) ==========
    print(f"\n{'='*70}")
    print("Validating SimpleCalorimeter Step 2: Binning (Tower Hits)")
    print(f"{'='*70}")
    
    if not Path(cpp_towerhits_file).exists():
        print(f"  ⚠ C++ debug file not found: {cpp_towerhits_file}")
        print("  Run C++ Delphes with modified SimpleCalorimeter.cc to generate this file.")
    else:
        cpp_hits_df = pd.read_csv(cpp_towerhits_file)
        print(f"  Loaded {len(cpp_hits_df)} tower hits from C++ debug file")
        
        # Separate particle and track hits
        cpp_particle_hits = cpp_hits_df[cpp_hits_df['type'] == 'particle']
        cpp_track_hits = cpp_hits_df[cpp_hits_df['type'] == 'track']
        
        # Get TorchDelphes valid particles/tracks (those that passed binning filter)
        particle_valid = torch.cat(ecal_results['particle_valid'])
        track_valid = torch.cat(ecal_results['track_valid'])
        particle_eta_bins = torch.cat(ecal_results['particle_eta_bins'])
        particle_phi_bins = torch.cat(ecal_results['particle_phi_bins'])
        track_eta_bins = torch.cat(ecal_results['track_eta_bins'])
        track_phi_bins = torch.cat(ecal_results['track_phi_bins'])
        
        # Count valid particles/tracks
        torch_valid_particles = particle_valid.sum().item()
        torch_valid_tracks = track_valid.sum().item()
        cpp_valid_particles = len(cpp_particle_hits)
        cpp_valid_tracks = len(cpp_track_hits)
        
        print(f"  TorchDelphes valid particles: {torch_valid_particles}")
        print(f"  C++ Delphes valid particles:  {cpp_valid_particles}")
        print(f"  TorchDelphes valid tracks:    {torch_valid_tracks}")
        print(f"  C++ Delphes valid tracks:     {cpp_valid_tracks}")
        
        # Debug: Print track eta/phi bin ranges and distributions
        print(f"\n  DEBUG Track binning:")
        print(f"    Track eta_bin range: [{track_eta_bins.min().item():.0f}, {track_eta_bins.max().item():.0f}]")
        print(f"    Track phi_bin range: [{track_phi_bins.min().item():.0f}, {track_phi_bins.max().item():.0f}]")
        print(f"    Total tracks: {len(track_valid)}")
        print(f"    Tracks with valid eta bin: {((track_eta_bins > 0) & (track_eta_bins < 260)).sum().item()}")
        print(f"    Tracks with valid phi bin: {((track_phi_bins > 0) & (track_phi_bins < 361)).sum().item()}")
        
        # Check if counts match
        if torch_valid_particles != cpp_valid_particles:
            print(f"  ⚠ Valid particle count mismatch!")
        else:
            print(f"  ✓ Valid particle counts match")
            
        if torch_valid_tracks != cpp_valid_tracks:
            print(f"  ⚠ Valid track count mismatch!")
        else:
            print(f"  ✓ Valid track counts match")
        
        # Compare eta/phi bin distributions for valid hits
        for hit_type, valid_mask, eta_bins, phi_bins, cpp_hits in [
            ('Particle', particle_valid, particle_eta_bins, particle_phi_bins, cpp_particle_hits),
            ('Track', track_valid, track_eta_bins, track_phi_bins, cpp_track_hits),
        ]:
            torch_eta = eta_bins[valid_mask].numpy()
            torch_phi = phi_bins[valid_mask].numpy()
            cpp_eta = cpp_hits['eta_bin'].values
            cpp_phi = cpp_hits['phi_bin'].values
            
            # Create figure with 2x2 grid: top row histograms, bottom row ratios
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)
            
            # === Left column: Eta bins ===
            ax_eta = fig.add_subplot(gs[0, 0])
            ax_eta_ratio = fig.add_subplot(gs[1, 0], sharex=ax_eta)
            
            if len(torch_eta) > 0 and len(cpp_eta) > 0:
                all_eta = np.concatenate([torch_eta, cpp_eta])
                eta_bin_edges = np.arange(all_eta.min() - 0.5, all_eta.max() + 1.5, 1)
            else:
                eta_bin_edges = 50
            
            cpp_eta_counts, eta_bin_edges, _ = ax_eta.hist(
                cpp_eta, bins=eta_bin_edges, histtype='stepfilled', color='orange', alpha=0.5,
                linewidth=2, label=f'C++ Delphes: {len(cpp_eta)} hits', density=False)
            torch_eta_counts, _, _ = ax_eta.hist(
                torch_eta, bins=eta_bin_edges, histtype='step', color='blue',
                linewidth=2, label=f'Parnassus.TorchDelphes: {len(torch_eta)} hits', density=False)

            ax_eta.set_ylabel('Counts', fontsize=12)
            ax_eta.set_title(f'SimpleCalorimeter Step 2: {hit_type} Eta Bins', fontsize=14)
            ax_eta.legend(fontsize=11)
            ax_eta.grid(True, alpha=0.3)
            ax_eta.tick_params(labelbottom=False)
            
            # Eta ratio plot
            eta_bin_centers = (eta_bin_edges[:-1] + eta_bin_edges[1:]) / 2
            eta_ratio = np.divide(
                torch_eta_counts, cpp_eta_counts,
                out=np.ones_like(torch_eta_counts),
                where=cpp_eta_counts > 0
            )
            ax_eta_ratio.axhline(y=1.0, color='orange', linewidth=2)
            ax_eta_ratio.plot(eta_bin_centers, eta_ratio, color='blue', linewidth=2)
            ax_eta_ratio.set_xlabel('Eta Bin Index', fontsize=12)
            ax_eta_ratio.set_ylabel('Ratio', fontsize=10)
            ax_eta_ratio.set_ylim([0.9*min(eta_ratio), 1.1*max(eta_ratio)])
            ax_eta_ratio.grid(True, alpha=0.3)
            
            # === Right column: Phi bins ===
            ax_phi = fig.add_subplot(gs[0, 1])
            ax_phi_ratio = fig.add_subplot(gs[1, 1], sharex=ax_phi)
            
            if len(torch_phi) > 0 and len(cpp_phi) > 0:
                all_phi = np.concatenate([torch_phi, cpp_phi])
                phi_bin_edges = np.arange(all_phi.min() - 0.5, all_phi.max() + 1.5, 1)
            else:
                phi_bin_edges = 50

            cpp_phi_counts, phi_bin_edges, _ = ax_phi.hist(
                cpp_phi, bins=phi_bin_edges, histtype='stepfilled', color='orange', alpha=0.5,
                linewidth=2, label=f'C++ Delphes: {len(cpp_phi)} hits', density=False)
            torch_phi_counts, _, _ = ax_phi.hist(
                torch_phi, bins=phi_bin_edges, histtype='step', color='blue',
                linewidth=2, label=f'Parnassus.TorchDelphes: {len(torch_phi)} hits', density=False)

            ax_phi.set_ylabel('Counts', fontsize=12)
            ax_phi.set_title(f'SimpleCalorimeter Step 2: {hit_type} Phi Bins', fontsize=14)
            ax_phi.legend(fontsize=11)
            ax_phi.grid(True, alpha=0.3)
            ax_phi.tick_params(labelbottom=False)
            
            # Phi ratio plot
            phi_bin_centers = (phi_bin_edges[:-1] + phi_bin_edges[1:]) / 2
            phi_ratio = np.divide(
                torch_phi_counts, cpp_phi_counts,
                out=np.ones_like(torch_phi_counts),
                where=cpp_phi_counts > 0
            )
            ax_phi_ratio.axhline(y=1.0, color='orange', linewidth=2)
            ax_phi_ratio.plot(phi_bin_centers, phi_ratio, color='blue', linewidth=2)
            ax_phi_ratio.set_xlabel('Phi Bin Index', fontsize=12)
            ax_phi_ratio.set_ylabel('Ratio', fontsize=10)
            ax_phi_ratio.set_ylim([0.9*min(phi_ratio), 1.1*max(phi_ratio)])
            ax_phi_ratio.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / f"{hit_type.lower()}_bins.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")
        
        print(f"  ✓ Step 2 validation complete.")
    
    # ========== Step 4: Tower Energy Aggregation ==========
    print(f"\n{'='*70}")
    print("Validating SimpleCalorimeter Step 4: Tower Energy Aggregation")
    print(f"{'='*70}")
    
    if not Path(cpp_towerenergy_file).exists():
        print(f"  ⚠ C++ debug file not found: {cpp_towerenergy_file}")
        print("  Run C++ Delphes with modified SimpleCalorimeter.cc to generate this file.")
    else:
        cpp_towers_df = pd.read_csv(cpp_towerenergy_file)
        print(f"  Loaded {len(cpp_towers_df)} towers from C++ debug file")
        
        # Get TorchDelphes tower results
        tower_results = ecal_results['tower_results']
        
        # Aggregate all events - for comparison, we need to match towers by (eta, phi)
        # C++ outputs tower_eta, tower_phi (center coordinates)
        # We'll compare by event and validate aggregate statistics
        
        # Aggregate statistics across all events
        total_torch_towers = sum(r['n_towers'] for r in tower_results)
        total_cpp_towers = len(cpp_towers_df)
        
        print(f"  TorchDelphes total towers: {total_torch_towers}")
        print(f"  C++ Delphes total towers:  {total_cpp_towers}")
        
        if total_torch_towers != total_cpp_towers:
            print(f"  ⚠ Tower count mismatch!")
        else:
            print(f"  ✓ Tower counts match")
        
        # Aggregate all tower energies
        torch_tower_energies = torch.cat([r['tower_energy'] for r in tower_results]).numpy()
        torch_track_energies = torch.cat([r['tower_track_energy'] for r in tower_results]).numpy()
        cpp_tower_energies = cpp_towers_df['tower_energy'].values
        cpp_track_energies = cpp_towers_df['track_energy'].values
        
        print(f"\n  Tower Energy Statistics:")
        print(f"    TorchDelphes: sum={torch_tower_energies.sum():.2f}, mean={torch_tower_energies.mean():.4f}, max={torch_tower_energies.max():.4f}")
        print(f"    C++ Delphes:  sum={cpp_tower_energies.sum():.2f}, mean={cpp_tower_energies.mean():.4f}, max={cpp_tower_energies.max():.4f}")
        
        print(f"\n  Track Energy Statistics:")
        print(f"    TorchDelphes: sum={torch_track_energies.sum():.2f}, mean={torch_track_energies.mean():.4f}, max={torch_track_energies.max():.4f}")
        print(f"    C++ Delphes:  sum={cpp_track_energies.sum():.2f}, mean={cpp_track_energies.mean():.4f}, max={cpp_track_energies.max():.4f}")
        
        # Create comparison plots with ratio subplots
        fig = plt.figure(figsize=(16, 14))
        
        # ===== Plot 1: Tower energy distribution with ratio =====
        ax1_main = fig.add_axes([0.05, 0.75, 0.4, 0.2])  # Main histogram
        ax1_ratio = fig.add_axes([0.05, 0.55, 0.4, 0.1])  # Ratio subplot
        
        # Use log-spaced bins for energy
        e_min = max(min(torch_tower_energies.min(), cpp_tower_energies.min()), 1e-6)
        e_max = max(torch_tower_energies.max(), cpp_tower_energies.max())
        energy_bins = np.logspace(np.log10(e_min), np.log10(e_max * 1.1), 50)
        
        cpp_counts, _ = np.histogram(cpp_tower_energies, bins=energy_bins)
        torch_counts, _ = np.histogram(torch_tower_energies, bins=energy_bins)
        
        ax1_main.hist(cpp_tower_energies, bins=energy_bins, histtype='stepfilled', color='orange', alpha=0.5,
                label=f'C++ Delphes ({len(cpp_tower_energies)} towers)')
        ax1_main.hist(torch_tower_energies, bins=energy_bins, histtype='step', color='blue', linewidth=2,
                label=f'TorchDelphes ({len(torch_tower_energies)} towers)')
        ax1_main.set_xscale('log')
        ax1_main.set_ylabel('Counts', fontsize=12)
        ax1_main.set_title('Step 4: Tower Energy (fTowerEnergy)', fontsize=14)
        ax1_main.legend(fontsize=10)
        ax1_main.grid(True, alpha=0.3)
        ax1_main.set_xticklabels([])
        
        # Ratio subplot
        bin_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(cpp_counts > 0, torch_counts / cpp_counts, np.nan)
        ax1_ratio.scatter(bin_centers, ratio, s=15, c='purple', alpha=0.7)
        ax1_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
        ax1_ratio.set_xscale('log')
        ax1_ratio.set_xlabel('Tower Energy (GeV)', fontsize=12)
        ax1_ratio.set_ylabel('Torch/C++', fontsize=10)
        ax1_ratio.set_ylim(0.9*min(ratio[~np.isnan(ratio)]), 1.1*max(ratio[~np.isnan(ratio)]))
        ax1_ratio.grid(True, alpha=0.3)
        
        # ===== Plot 2: Track energy distribution with ratio =====
        ax2_main = fig.add_axes([0.55, 0.75, 0.4, 0.2])
        ax2_ratio = fig.add_axes([0.55, 0.55, 0.4, 0.1])
        
        nonzero_torch_track = torch_track_energies[torch_track_energies > 0]
        nonzero_cpp_track = cpp_track_energies[cpp_track_energies > 0]
        
        if len(nonzero_torch_track) > 0 or len(nonzero_cpp_track) > 0:
            t_min = max(min(nonzero_torch_track.min() if len(nonzero_torch_track) > 0 else 1e-6,
                           nonzero_cpp_track.min() if len(nonzero_cpp_track) > 0 else 1e-6), 1e-6)
            t_max = max(nonzero_torch_track.max() if len(nonzero_torch_track) > 0 else 1,
                       nonzero_cpp_track.max() if len(nonzero_cpp_track) > 0 else 1)
            track_bins = np.logspace(np.log10(t_min), np.log10(t_max * 1.1), 50)
            
            cpp_track_counts, _ = np.histogram(nonzero_cpp_track, bins=track_bins)
            torch_track_counts, _ = np.histogram(nonzero_torch_track, bins=track_bins)
            
            ax2_main.hist(nonzero_cpp_track, bins=track_bins, histtype='stepfilled', color='orange', alpha=0.5,
                    label=f'C++ Delphes ({len(nonzero_cpp_track)} non-zero)')
            ax2_main.hist(nonzero_torch_track, bins=track_bins, histtype='step', color='blue', linewidth=2,
                    label=f'TorchDelphes ({len(nonzero_torch_track)} non-zero)')
            ax2_main.set_xscale('log')
            
            # Ratio subplot
            track_bin_centers = np.sqrt(track_bins[:-1] * track_bins[1:])
            with np.errstate(divide='ignore', invalid='ignore'):
                track_ratio = np.where(cpp_track_counts > 0, torch_track_counts / cpp_track_counts, np.nan)
            ax2_ratio.scatter(track_bin_centers, track_ratio, s=15, c='purple', alpha=0.7)
            ax2_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
            ax2_ratio.set_xscale('log')
            ax2_ratio.set_ylim(0.9*min(track_ratio), 1.1*max(track_ratio))
        else:
            ax2_main.text(0.5, 0.5, 'No non-zero track energies', transform=ax2_main.transAxes,
                   ha='center', va='center', fontsize=14)
        
        ax2_main.set_ylabel('Counts', fontsize=12)
        ax2_main.set_title('Step 4: Track Energy (fTrackEnergy)', fontsize=14)
        ax2_main.legend(fontsize=10)
        ax2_main.grid(True, alpha=0.3)
        ax2_main.set_xticklabels([])
        ax2_ratio.set_xlabel('Track Energy in Tower (GeV)', fontsize=12)
        ax2_ratio.set_ylabel('Torch/C++', fontsize=10)
        ax2_ratio.grid(True, alpha=0.3)
        
        # ===== Plot 3: Scatter plot comparing tower energies (sorted) =====
        ax3 = fig.add_axes([0.05, 0.08, 0.4, 0.35])
        
        torch_sorted = np.sort(torch_tower_energies)[::-1]
        cpp_sorted = np.sort(cpp_tower_energies)[::-1]
        
        # Pad shorter array with zeros for comparison
        max_len = max(len(torch_sorted), len(cpp_sorted))
        torch_padded = np.pad(torch_sorted, (0, max_len - len(torch_sorted)))
        cpp_padded = np.pad(cpp_sorted, (0, max_len - len(cpp_sorted)))
        
        ax3.scatter(cpp_padded, torch_padded, alpha=0.5, s=10, c='purple')
        
        # Add y=x line
        max_val = max(cpp_padded.max(), torch_padded.max())
        ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
        
        ax3.set_xlabel('C++ Tower Energy (sorted)', fontsize=12)
        ax3.set_ylabel('TorchDelphes Tower Energy (sorted)', fontsize=12)
        ax3.set_title('Tower Energy Comparison (Sorted)', fontsize=14)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
        
        # ===== Plot 4: Number of towers per event with ratio =====
        ax4_main = fig.add_axes([0.55, 0.20, 0.4, 0.23])
        ax4_ratio = fig.add_axes([0.55, 0.08, 0.4, 0.10])
        
        torch_towers_per_event = [r['n_towers'] for r in tower_results]
        cpp_events = cpp_towers_df['event'].unique()
        cpp_towers_per_event = [len(cpp_towers_df[cpp_towers_df['event'] == e]) for e in cpp_events]
        
        x = np.arange(len(torch_towers_per_event))
        width = 0.35
        
        ax4_main.bar(x - width/2, cpp_towers_per_event[:len(x)], width, label='C++ Delphes', color='orange', alpha=0.7)
        ax4_main.bar(x + width/2, torch_towers_per_event, width, label='TorchDelphes', color='blue', alpha=0.7)
        
        ax4_main.set_ylabel('Number of Towers', fontsize=12)
        ax4_main.set_title('Towers per Event', fontsize=14)
        ax4_main.legend(fontsize=10)
        ax4_main.grid(True, alpha=0.3, axis='y')
        ax4_main.set_xticklabels([])
        
        # Ratio subplot
        cpp_arr = np.array(cpp_towers_per_event[:len(x)])
        torch_arr = np.array(torch_towers_per_event)
        with np.errstate(divide='ignore', invalid='ignore'):
            towers_ratio = np.where(cpp_arr > 0, torch_arr / cpp_arr, np.nan)
        ax4_ratio.scatter(x, towers_ratio, s=20, c='purple', alpha=0.7)
        ax4_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
        ax4_ratio.set_xlabel('Event', fontsize=12)
        ax4_ratio.set_ylabel('Torch/C++', fontsize=10)
        ax4_ratio.set_ylim(0.9*min(towers_ratio), 1.1*max(towers_ratio))
        ax4_ratio.grid(True, alpha=0.3)
        
        # Add summary statistics
        total_torch = sum(torch_towers_per_event)
        total_cpp = sum(cpp_towers_per_event[:len(x)])
        fig.text(0.55, 0.45, f'Total towers: TorchDelphes={total_torch}, C++={total_cpp} (ratio={total_torch/total_cpp:.3f})',
                fontsize=11, ha='left')
        
        plot_file = output_dir / "tower_energies.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {plot_file}")
        
        # Per-event detailed comparison
        print(f"\n  Per-event tower comparison:")
        for event_idx, (torch_res, cpp_event) in enumerate(zip(tower_results, cpp_events)):
            cpp_event_df = cpp_towers_df[cpp_towers_df['event'] == cpp_event]
            
            torch_n = torch_res['n_towers']
            cpp_n = len(cpp_event_df)
            torch_sum_e = torch_res['tower_energy'].sum().item()
            cpp_sum_e = cpp_event_df['tower_energy'].sum()
            torch_sum_t = torch_res['tower_track_energy'].sum().item()
            cpp_sum_t = cpp_event_df['track_energy'].sum()
            
            match_str = "✓" if torch_n == cpp_n else "✗"
            print(f"    Event {event_idx}: {match_str} towers: Torch={torch_n}, C++={cpp_n} | "
                  f"E_tower: Torch={torch_sum_e:.2f}, C++={cpp_sum_e:.2f} | "
                  f"E_track: Torch={torch_sum_t:.2f}, C++={cpp_sum_t:.2f}")
            
            if event_idx >= 9:  # Limit output
                print(f"    ... (showing first 10 events)")
                break
        
        print(f"  ✓ Step 4 validation complete.")
    
    # ============ Step 5: Validate Tower Centers ============
    # Compare tower_eta, tower_phi, and tower edges against C++ debug output
    print(f"\n  Step 5: Validating Tower Centers...")
    
    # Collect all torch tower positions for comparison
    torch_tower_eta = np.concatenate([r['tower_eta'].numpy() for r in tower_results])
    torch_tower_phi = np.concatenate([r['tower_phi'].numpy() for r in tower_results])
    torch_eta_lo = np.concatenate([r['tower_eta_lo'].numpy() for r in tower_results])
    torch_eta_hi = np.concatenate([r['tower_eta_hi'].numpy() for r in tower_results])
    torch_phi_lo = np.concatenate([r['tower_phi_lo'].numpy() for r in tower_results])
    torch_phi_hi = np.concatenate([r['tower_phi_hi'].numpy() for r in tower_results])
    
    # C++ tower positions from the tower energy file
    cpp_tower_eta = cpp_towers_df['tower_eta'].values
    cpp_tower_phi = cpp_towers_df['tower_phi'].values
    cpp_eta_lo = cpp_towers_df['eta_lo'].values
    cpp_eta_hi = cpp_towers_df['eta_hi'].values
    cpp_phi_lo = cpp_towers_df['phi_lo'].values
    cpp_phi_hi = cpp_towers_df['phi_hi'].values
    
    # Create comparison plots with ratio subplots
    fig = plt.figure(figsize=(16, 12))
    
    # ===== Tower eta comparison with ratio =====
    ax_eta_main = fig.add_axes([0.05, 0.72, 0.28, 0.22])
    ax_eta_ratio = fig.add_axes([0.05, 0.58, 0.28, 0.10])
    
    eta_bins = np.linspace(-5, 5, 100)
    cpp_eta_counts, _ = np.histogram(cpp_tower_eta, bins=eta_bins)
    torch_eta_counts, _ = np.histogram(torch_tower_eta, bins=eta_bins)
    
    ax_eta_main.hist(cpp_tower_eta, bins=eta_bins, histtype='stepfilled', color='orange', alpha=0.5,
            label=f'C++ Delphes ({len(cpp_tower_eta)})')
    ax_eta_main.hist(torch_tower_eta, bins=eta_bins, histtype='step', color='blue', linewidth=2,
            label=f'TorchDelphes ({len(torch_tower_eta)})')
    ax_eta_main.set_ylabel('Counts', fontsize=12)
    ax_eta_main.set_title('Tower Eta Distribution', fontsize=14)
    ax_eta_main.legend(fontsize=9)
    ax_eta_main.grid(True, alpha=0.3)
    ax_eta_main.set_xticklabels([])
    
    # Ratio subplot
    eta_bin_centers = 0.5 * (eta_bins[:-1] + eta_bins[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        eta_ratio = np.where(cpp_eta_counts > 0, torch_eta_counts / cpp_eta_counts, np.nan)
    valid_eta_ratio = ~np.isnan(eta_ratio)
    ax_eta_ratio.scatter(eta_bin_centers[valid_eta_ratio], eta_ratio[valid_eta_ratio], s=10, c='purple', alpha=0.7)
    ax_eta_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
    ax_eta_ratio.set_xlabel('Tower Eta', fontsize=12)
    ax_eta_ratio.set_ylabel('Torch/C++', fontsize=10)
    ax_eta_ratio.set_ylim(0.5, 1.5)
    ax_eta_ratio.grid(True, alpha=0.3)
    
    # ===== Tower phi comparison with ratio =====
    ax_phi_main = fig.add_axes([0.38, 0.72, 0.28, 0.22])
    ax_phi_ratio = fig.add_axes([0.38, 0.58, 0.28, 0.10])
    
    phi_bins = np.linspace(-np.pi, np.pi, 100)
    cpp_phi_counts, _ = np.histogram(cpp_tower_phi, bins=phi_bins)
    torch_phi_counts, _ = np.histogram(torch_tower_phi, bins=phi_bins)
    
    ax_phi_main.hist(cpp_tower_phi, bins=phi_bins, histtype='stepfilled', color='orange', alpha=0.5,
            label=f'C++ Delphes')
    ax_phi_main.hist(torch_tower_phi, bins=phi_bins, histtype='step', color='blue', linewidth=2,
            label=f'TorchDelphes')
    ax_phi_main.set_ylabel('Counts', fontsize=12)
    ax_phi_main.set_title('Tower Phi Distribution', fontsize=14)
    ax_phi_main.legend(fontsize=9)
    ax_phi_main.grid(True, alpha=0.3)
    ax_phi_main.set_xticklabels([])
    
    # Ratio subplot
    phi_bin_centers = 0.5 * (phi_bins[:-1] + phi_bins[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        phi_ratio = np.where(cpp_phi_counts > 0, torch_phi_counts / cpp_phi_counts, np.nan)
    valid_phi_ratio = ~np.isnan(phi_ratio)
    ax_phi_ratio.scatter(phi_bin_centers[valid_phi_ratio], phi_ratio[valid_phi_ratio], s=10, c='purple', alpha=0.7)
    ax_phi_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
    ax_phi_ratio.set_xlabel('Tower Phi', fontsize=12)
    ax_phi_ratio.set_ylabel('Torch/C++', fontsize=10)
    ax_phi_ratio.set_ylim(0.5, 1.5)
    ax_phi_ratio.grid(True, alpha=0.3)
    
    # ===== 2D eta-phi scatter (sample to avoid too many points) =====
    ax_2d = fig.add_axes([0.71, 0.58, 0.26, 0.36])
    n_sample = min(5000, len(cpp_tower_eta), len(torch_tower_eta))
    cpp_idx = np.random.choice(len(cpp_tower_eta), n_sample, replace=False) if len(cpp_tower_eta) > n_sample else np.arange(len(cpp_tower_eta))
    torch_idx = np.random.choice(len(torch_tower_eta), n_sample, replace=False) if len(torch_tower_eta) > n_sample else np.arange(len(torch_tower_eta))
    ax_2d.scatter(cpp_tower_eta[cpp_idx], cpp_tower_phi[cpp_idx], s=5, alpha=0.3, c='orange', label='C++')
    ax_2d.scatter(torch_tower_eta[torch_idx], torch_tower_phi[torch_idx], s=5, alpha=0.3, c='blue', label='Torch')
    ax_2d.set_xlabel('Tower Eta', fontsize=12)
    ax_2d.set_ylabel('Tower Phi', fontsize=12)
    ax_2d.set_title('Tower Eta-Phi Distribution', fontsize=14)
    ax_2d.legend(fontsize=9)
    ax_2d.grid(True, alpha=0.3)
    
    # ===== Eta comparison scatter (sorted values) =====
    ax_eta_scatter = fig.add_axes([0.05, 0.08, 0.26, 0.38])
    torch_eta_sorted = np.sort(torch_tower_eta)
    cpp_eta_sorted = np.sort(cpp_tower_eta)
    min_len = min(len(torch_eta_sorted), len(cpp_eta_sorted))
    ax_eta_scatter.scatter(cpp_eta_sorted[:min_len], torch_eta_sorted[:min_len], s=5, alpha=0.5, c='purple')
    ax_eta_scatter.plot([-5, 5], [-5, 5], 'r--', linewidth=2, label='y=x')
    ax_eta_scatter.set_xlabel('C++ Tower Eta (sorted)', fontsize=12)
    ax_eta_scatter.set_ylabel('TorchDelphes Tower Eta (sorted)', fontsize=12)
    ax_eta_scatter.set_title('Tower Eta Comparison', fontsize=14)
    ax_eta_scatter.legend(fontsize=9)
    ax_eta_scatter.grid(True, alpha=0.3)
    ax_eta_scatter.set_aspect('equal', adjustable='box')
    
    # ===== Phi comparison scatter (sorted values) =====
    ax_phi_scatter = fig.add_axes([0.38, 0.08, 0.26, 0.38])
    torch_phi_sorted = np.sort(torch_tower_phi)
    cpp_phi_sorted = np.sort(cpp_tower_phi)
    min_len = min(len(torch_phi_sorted), len(cpp_phi_sorted))
    ax_phi_scatter.scatter(cpp_phi_sorted[:min_len], torch_phi_sorted[:min_len], s=5, alpha=0.5, c='purple')
    ax_phi_scatter.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', linewidth=2, label='y=x')
    ax_phi_scatter.set_xlabel('C++ Tower Phi (sorted)', fontsize=12)
    ax_phi_scatter.set_ylabel('TorchDelphes Tower Phi (sorted)', fontsize=12)
    ax_phi_scatter.set_title('Tower Phi Comparison', fontsize=14)
    ax_phi_scatter.legend(fontsize=9)
    ax_phi_scatter.grid(True, alpha=0.3)
    ax_phi_scatter.set_aspect('equal', adjustable='box')
    
    # ===== Unique tower center positions summary =====
    ax_summary = fig.add_axes([0.71, 0.08, 0.26, 0.38])
    torch_unique_centers = set(zip(np.round(torch_tower_eta, 4), np.round(torch_tower_phi, 4)))
    cpp_unique_centers = set(zip(np.round(cpp_tower_eta, 4), np.round(cpp_tower_phi, 4)))
    common = torch_unique_centers & cpp_unique_centers
    torch_only = torch_unique_centers - cpp_unique_centers
    cpp_only = cpp_unique_centers - torch_unique_centers
    
    ax_summary.text(0.5, 0.8, f'Unique tower positions:', transform=ax_summary.transAxes, fontsize=12, ha='center', fontweight='bold')
    ax_summary.text(0.5, 0.65, f'C++ Delphes: {len(cpp_unique_centers)}', transform=ax_summary.transAxes, fontsize=11, ha='center', color='orange')
    ax_summary.text(0.5, 0.52, f'TorchDelphes: {len(torch_unique_centers)}', transform=ax_summary.transAxes, fontsize=11, ha='center', color='blue')
    ax_summary.text(0.5, 0.39, f'Common: {len(common)}', transform=ax_summary.transAxes, fontsize=11, ha='center', color='green')
    ax_summary.text(0.5, 0.26, f'C++ only: {len(cpp_only)}', transform=ax_summary.transAxes, fontsize=11, ha='center', color='red')
    ax_summary.text(0.5, 0.13, f'Torch only: {len(torch_only)}', transform=ax_summary.transAxes, fontsize=11, ha='center', color='purple')
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis('off')
    ax_summary.set_title('Tower Position Summary', fontsize=14)
    plot_file = output_dir / "tower_centers.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {plot_file}")
    
    # Print statistics
    print(f"\n  Tower Center Statistics:")
    print(f"    TorchDelphes: eta range=[{torch_tower_eta.min():.4f}, {torch_tower_eta.max():.4f}], "
          f"phi range=[{torch_tower_phi.min():.4f}, {torch_tower_phi.max():.4f}]")
    print(f"    C++ Delphes:  eta range=[{cpp_tower_eta.min():.4f}, {cpp_tower_eta.max():.4f}], "
          f"phi range=[{cpp_tower_phi.min():.4f}, {cpp_tower_phi.max():.4f}]")
    
    print(f"  ✓ Step 5 validation complete.")
    
    print(f"\n  ✓ All SimpleCalorimeter validation complete. Plots saved to {output_dir}")


def validate_against_benchmark(
    torch_output_file: str, 
    benchmark_file: str, 
    output_dir: str, 
    debug: bool = False
) -> None:
    """
    Validate PyTorch Delphes implementation against C++ Delphes benchmark.
    
    Args:
        torch_output_file: Path to PyTorch output ROOT file (e.g., HZZ4l_3_2_torch.root)
        benchmark_file: Path to benchmark ROOT file from C++ Delphes
        output_dir: Directory to save validation plots
        debug: If True, print histogram bin counts and edges
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading PyTorch output: {torch_output_file}")
    torch_root = uproot.open(torch_output_file)
    torch_tree = torch_root["Delphes"]
    
    print(f"Loading C++ Delphes benchmark: {benchmark_file}")
    benchmark_root = uproot.open(benchmark_file)
    benchmark_tree = benchmark_root["Delphes"]
    
    # Kinematic variables to compare
    # Track objects: PID, Charge, P, PT, Eta, Phi
    # Tower objects: E, ET, Eta, Phi, Eem, Ehad (no PID - towers are aggregated)
    track_kinematic_vars = ['PID', 'Charge', 'P', 'PT', 'Eta', 'EtaOuter', 'Phi', 'T', 'X', 'Y', 'Z']
    tower_kinematic_vars = ['E', 'ET', 'Eta', 'Phi', 'T']
    
    # Branches to validate (branch_name, variable_list)
    branches = [
        # ('ParticleBeforeProp', track_kinematic_vars),
        # ('ParticleAfterProp', track_kinematic_vars),
        # ('ChargedHadron', track_kinematic_vars),
        # ('Electron', track_kinematic_vars),
        # ('Muon', track_kinematic_vars),
        # ('ChargedHadronEfficiency', track_kinematic_vars),
        # ('ElectronEfficiency', track_kinematic_vars),
        # ('MuonEfficiency', track_kinematic_vars),
        # ('ChargedHadronSmeared', track_kinematic_vars),
        # ('ElectronSmeared', track_kinematic_vars),
        # ('MuonSmeared', track_kinematic_vars),
        # ('MergedTracks', track_kinematic_vars),
    ]
    
    print(f"\nValidating branches: {', '.join([b[0] for b in branches])}")
    
    for branch_name, kinematic_vars in branches:
        print(f"\n{'='*70}")
        print(f"Validating {branch_name}...")
        print(f"{'='*70}")
        
        # Create branch-specific directory
        branch_dir = output_dir / branch_name
        branch_dir.mkdir(exist_ok=True)
        
        # Check if branch exists in PyTorch output
        torch_branch_keys = [k for k in torch_tree.keys() if k.startswith(f"{branch_name}/")]
        if not torch_branch_keys:
            print(f"  ⚠ {branch_name} not found in PyTorch output, skipping...")
            continue
        
        ### 1. Standalone plots for each kinematic variable
        for var in kinematic_vars:
            # Check if variable exists in both datasets
            torch_key = f"{branch_name}/{branch_name}.{var}"
            benchmark_key = f"{branch_name}/{branch_name}.{var}"
            
            if torch_key not in torch_tree.keys():
                print(f"  ⚠ {var} not found in PyTorch {branch_name}, skipping...")
                continue
            
            if benchmark_key not in benchmark_tree.keys():
                print(f"  ⚠ {var} not found in C++ {branch_name}, skipping...")
                continue
            
            try:
                # Load data from both sources
                torch_data = torch_tree[torch_key].array()
                torch_data = ak.flatten(torch_data)
                
                benchmark_data = benchmark_tree[benchmark_key].array()
                benchmark_data = ak.flatten(benchmark_data)
                
                # Convert to numpy for plotting
                torch_np = np.asarray(torch_data)
                benchmark_np = np.asarray(benchmark_data)
                
                # Create figure with two subplots: histogram on top, ratio below
                fig = plt.figure(figsize=(10, 8))
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax_hist = fig.add_subplot(gs[0])
                ax_ratio = fig.add_subplot(gs[1], sharex=ax_hist)
                
                # Special handling for PID: use discrete bins
                if var == 'PID':
                    # Get unique PIDs across both datasets
                    unique_pids = np.unique(np.concatenate([torch_np, benchmark_np]))
                    
                    # Count occurrences of each PID
                    torch_counts = np.array([np.sum(torch_np == pid) for pid in unique_pids])
                    benchmark_counts = np.array([np.sum(benchmark_np == pid) for pid in unique_pids])
                    
                    # Create bar positions
                    x = np.arange(len(unique_pids))
                    width = 0.35
                    
                    # Plot bars
                    ax_hist.bar(x - width/2, benchmark_counts, width, label='C++ Delphes', 
                               color='orange', alpha=0.7)
                    ax_hist.bar(x + width/2, torch_counts, width, label='Parnassus.TorchDelphes', 
                               color='blue', alpha=0.7)
                    
                    ax_hist.set_xticks(x)
                    ax_hist.set_xticklabels([f'{int(pid)}' for pid in unique_pids], rotation=45, ha='right')
                    ax_hist.tick_params(labelbottom=False)
                    
                    # For ratio plot
                    bin_centers = x
                    ratio = np.divide(
                        torch_counts, benchmark_counts,
                        out=np.ones_like(torch_counts, dtype=float),
                        where=benchmark_counts > 0
                    )
                    
                    # Print PID counts if debug mode
                    if debug:
                        print(f"\n  PID Counts for {branch_name}:")
                        for pid, torch_count, bench_count in zip(unique_pids, torch_counts, benchmark_counts):
                            ratio_val = torch_count / bench_count if bench_count > 0 else np.inf
                            print(f"    PID {int(pid):6d}: PyTorch={torch_count:5d}, C++={bench_count:5d}, Ratio={ratio_val:.4f}")
                    
                else:
                    # Standard continuous histogram
                    # Determine bin range
                    all_data = np.concatenate([torch_np, benchmark_np])
                    if len(all_data) > 0:
                        bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 50)
                    else:
                        bins = 50
                    
                    # Plot histograms
                    benchmark_counts, bin_edges, _ = ax_hist.hist(
                        benchmark_np, bins=bins, histtype='stepfilled', color='orange', alpha=0.5,
                        linewidth=2, label='C++ Delphes', density=False
                    )
                    torch_counts, _, _ = ax_hist.hist(
                        torch_np, bins=bins, histtype='step', color='blue', 
                        linewidth=2, label='Parnassus.TorchDelphes', density=False
                    )
                    
                    # For ratio plot
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ratio = np.divide(
                        torch_counts, benchmark_counts, 
                        out=np.ones_like(torch_counts), 
                        where=benchmark_counts > 0
                    )
                
                # Debug: print histogram statistics
                if debug and var != 'PID':
                    print(f"\n{branch_name}.{var} bins:")
                    print(f"  Bin edges: {bin_edges}, len(bin_edges)={len(bin_edges)}")
                    print(f"  C++ counts: {benchmark_counts}")
                    print(f"  TorchDelphes counts: {torch_counts}")
                    print(f"  Total C++ counts: {np.sum(benchmark_counts):.0f}")
                    print(f"  Total TorchDelphes counts: {np.sum(torch_counts):.0f}")
                    print(f"  Ratio (Torch/C++): {np.sum(torch_counts) / np.sum(benchmark_counts):.4f}")
                    
                    # Compute and print ratio statistics
                    valid_ratio = ratio[benchmark_counts > 0]
                    if len(valid_ratio) > 0:
                        print(f"  Ratio mean: {np.mean(valid_ratio):.4f}")
                        print(f"  Ratio std: {np.std(valid_ratio):.4f}")
                        print(f"  Ratio min: {np.min(valid_ratio):.4f}")
                        print(f"  Ratio max: {np.max(valid_ratio):.4f}")
                    print(f"  --- END DEBUG ---\n")
                
                ax_hist.set_ylabel('Counts', fontsize=12)
                ax_hist.set_title(f'{branch_name}: {var}', fontsize=14, fontweight='bold')
                ax_hist.legend(fontsize=11)
                ax_hist.grid(True, alpha=0.3)
                if var != 'PID':
                    ax_hist.tick_params(labelbottom=False)  # Hide x-axis labels for top plot
                
                # Add statistics text
                stats_text = f'PyTorch: {len(torch_np)} particles\nC++ Delphes: {len(benchmark_np)} particles'
                ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Plot ratio: TorchDelphes / C++ Delphes
                if var == 'PID':
                    # Bar plot for PID
                    ax_ratio.axhline(y=1.0, color='orange', linewidth=2)
                    ax_ratio.bar(bin_centers, ratio, width*2, color='blue', alpha=0.7)
                    ax_ratio.set_xticks(bin_centers)
                    ax_ratio.set_xticklabels([f'{int(pid)}' for pid in unique_pids], rotation=45, ha='right')
                    ax_ratio.set_ylim([0.9*min(ratio), 1.1*max(ratio)])  # Focus on ±10% range
                else:
                    # Line plot for continuous variables
                    ax_ratio.axhline(y=1.0, color='orange', linewidth=2)
                    ax_ratio.plot(bin_centers, ratio, color='blue', markersize=4, linewidth=2)
                    ax_ratio.set_ylim([0.9*min(ratio), 1.1*max(ratio)])  # Focus on ±20% range
                
                ax_ratio.set_xlabel(var, fontsize=12)
                ax_ratio.set_ylabel('Torch / C++', fontsize=10)
                ax_ratio.grid(True, alpha=0.3)
                
                # Save plot
                plot_file = branch_dir / f"{var}.png"
                plt.tight_layout()
                plt.savefig(plot_file, dpi=150)
                plt.close()
                
                print(f"  ✓ {var}: PyTorch={len(torch_np)}, C++={len(benchmark_np)} → {plot_file.name}")
                
            except Exception as e:
                print(f"  ✗ {var}: Error - {e}")
                continue
        
        ### 2. Combined plot with key kinematic variables
        # For tracks: Eta, Phi, PT, P
        # For towers: Eta, Phi, E, ET
        if 'P' in kinematic_vars:
            combined_vars = ['Eta', 'Phi', 'PT', 'P']
            print(f"\n  Creating combined kinematic plot (Eta, Phi, PT, P)...")
        elif 'E' in kinematic_vars:
            combined_vars = ['Eta', 'Phi', 'E', 'ET']
            print(f"\n  Creating combined kinematic plot (Eta, Phi, E, ET)...")
        else:
            combined_vars = kinematic_vars[:4]  # Take first 4 variables
            print(f"\n  Creating combined kinematic plot ({', '.join(combined_vars)})...")
        
        # Create figure with 2 rows (histogram + ratio) and 4 columns (one per variable)
        fig = plt.figure(figsize=(30, 6))
        
        for idx, var in enumerate(combined_vars):
            torch_key = f"{branch_name}/{branch_name}.{var}"
            benchmark_key = f"{branch_name}/{branch_name}.{var}"
            
            if torch_key not in torch_tree.keys() or benchmark_key not in benchmark_tree.keys():
                continue
            
            try:
                # Load data
                torch_data = torch_tree[torch_key].array()
                torch_data = ak.flatten(torch_data)
                benchmark_data = benchmark_tree[benchmark_key].array()
                benchmark_data = ak.flatten(benchmark_data)
                
                # Convert to numpy
                torch_np = np.asarray(torch_data)
                benchmark_np = np.asarray(benchmark_data)
                
                # Create subplot with histogram on top, ratio below
                # Use 4 rows to match the 3:1 height ratio, columns for each variable
                gs = plt.GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.3, 
                                  height_ratios=[3, 1, 0, 0])
                
                # Column position (0-3 for Eta, Phi, PT, P)
                col = idx
                
                # Histogram subplot (row 0, takes 3 units of height)
                ax_hist = fig.add_subplot(gs[0, col])
                # Ratio subplot (row 1, takes 1 unit of height)
                ax_ratio = fig.add_subplot(gs[1, col], sharex=ax_hist)
                
                # Determine bin range
                all_data = np.concatenate([torch_np, benchmark_np])
                if len(all_data) > 0:
                    bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 40)
                else:
                    bins = 40
                
                # Plot histograms
                benchmark_counts, bin_edges, _ = ax_hist.hist(
                    benchmark_np, bins=bins, histtype='stepfilled', color='orange', alpha=0.5,
                    linewidth=2, label=f'C++ Delphes: {len(benchmark_np)} particles', density=False
                )
                torch_counts, _, _ = ax_hist.hist(
                    torch_np, bins=bins, histtype='step', color='blue', 
                    linewidth=2, label=f'Parnassus.TorchDelphes: {len(torch_np)} particles', density=False
                )
                
                ax_hist.set_ylabel('Counts', fontsize=11)
                ax_hist.set_title(f'{var}', fontsize=13, fontweight='bold')
                if idx == 0:  # Only show legend on first subplot
                    ax_hist.legend(fontsize=10)
                ax_hist.grid(True, alpha=0.3)
                ax_hist.tick_params(labelbottom=False)
                
                # Plot ratio
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ratio = np.divide(
                    torch_counts, benchmark_counts, 
                    out=np.ones_like(torch_counts), 
                    where=benchmark_counts > 0
                )
                
                ax_ratio.axhline(y=1.0, color='orange', linewidth=2)
                ax_ratio.plot(bin_centers, ratio, color='blue', markersize=3, linewidth=2)
                ax_ratio.set_xlabel(var, fontsize=11)
                ax_ratio.set_ylabel('Torch/C++', fontsize=9)
                ax_ratio.set_ylim([0.9*min(ratio), 1.1*max(ratio)])
                ax_ratio.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"    ✗ Error plotting {var} in combined plot: {e}")
                continue
        
        # Add overall title
        fig.suptitle(f'{branch_name}', fontsize=16, fontweight='bold', y=0.98)
        
        # Save combined figure
        combined_plot_file = branch_dir / "all.png"
        plt.savefig(combined_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Combined plot saved → {combined_plot_file.name}")
        
        ### 3. PID-specific combined plots (only for branches with PID field)
        torch_pid_key = f"{branch_name}/{branch_name}.PID"
        benchmark_pid_key = f"{branch_name}/{branch_name}.PID"
        
        if torch_pid_key in torch_tree.keys() and benchmark_pid_key in benchmark_tree.keys():
            
            # Load PID data to get unique PIDs
            torch_pids = torch_tree[torch_pid_key].array()
            benchmark_pids = benchmark_tree[benchmark_pid_key].array()
            
            # Get unique PIDs across both datasets
            torch_pids_flat = ak.flatten(torch_pids)
            benchmark_pids_flat = ak.flatten(benchmark_pids)
            unique_pids = np.unique(np.concatenate([
                np.asarray(torch_pids_flat),
                np.asarray(benchmark_pids_flat)
            ]))
            
            # For each unique PID, create a combined plot
            for pid in unique_pids:
                pid_int = int(pid)
                
                # Create figure with 2 rows (histogram + ratio) and 4 columns (one per variable)
                fig = plt.figure(figsize=(30, 6))
                
                for idx, var in enumerate(combined_vars):
                    torch_key = f"{branch_name}/{branch_name}.{var}"
                    benchmark_key = f"{branch_name}/{branch_name}.{var}"
                    
                    if torch_key not in torch_tree.keys() or benchmark_key not in benchmark_tree.keys():
                        continue
                    
                    # Load data (event-wise, not flattened yet)
                    torch_data_events = torch_tree[torch_key].array()
                    benchmark_data_events = benchmark_tree[benchmark_key].array()
                    
                    # Filter by PID: for each event, select only particles with matching PID
                    torch_pid_events = torch_tree[torch_pid_key].array()
                    benchmark_pid_events = benchmark_tree[benchmark_pid_key].array()
                    
                    # Apply PID mask and flatten
                    torch_data_filtered = ak.flatten(torch_data_events[torch_pid_events == pid])
                    benchmark_data_filtered = ak.flatten(benchmark_data_events[benchmark_pid_events == pid])
                    
                    # Convert to numpy
                    torch_np = np.asarray(torch_data_filtered)
                    benchmark_np = np.asarray(benchmark_data_filtered)
                    
                    # Skip if no data for this PID
                    if len(torch_np) == 0 and len(benchmark_np) == 0:
                        continue
                    
                    # Create subplot with histogram on top, ratio below
                    gs = plt.GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.3, 
                                        height_ratios=[3, 1, 0, 0])
                    
                    # Column position (0-3 for variables)
                    col = idx
                    
                    # Histogram subplot (row 0, takes 3 units of height)
                    ax_hist = fig.add_subplot(gs[0, col])
                    # Ratio subplot (row 1, takes 1 unit of height)
                    ax_ratio = fig.add_subplot(gs[1, col], sharex=ax_hist)
                    
                    # Determine bin range
                    all_data = np.concatenate([torch_np, benchmark_np])
                    if len(all_data) > 0:
                        bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 40)
                    else:
                        bins = 40
                    
                    # Plot histograms
                    benchmark_counts, bin_edges, _ = ax_hist.hist(
                        benchmark_np, bins=bins, histtype='stepfilled', color='orange', alpha=0.5,
                        linewidth=2, label=f'C++ Delphes, {len(benchmark_np)} particles', density=False
                    )
                    torch_counts, _, _ = ax_hist.hist(
                        torch_np, bins=bins, histtype='step', color='blue', 
                        linewidth=2, label=f'Parnassus.TorchDelphes, {len(torch_np)} particles', density=False
                    )
                    
                    ax_hist.set_ylabel('Counts', fontsize=11)
                    ax_hist.set_title(f'{var}', fontsize=13, fontweight='bold')
                    if idx == 0:  # Only show legend on first subplot
                        ax_hist.legend(fontsize=10)
                    ax_hist.grid(True, alpha=0.3)
                    ax_hist.tick_params(labelbottom=False)
                    
                    # Plot ratio
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ratio = np.divide(
                        torch_counts, benchmark_counts, 
                        out=np.ones_like(torch_counts), 
                        where=benchmark_counts > 0
                    )
                    
                    ax_ratio.axhline(y=1.0, color='orange', linewidth=2)
                    ax_ratio.plot(bin_centers, ratio, color='blue', markersize=3, linewidth=2)
                    ax_ratio.set_xlabel(var, fontsize=11)
                    ax_ratio.set_ylabel('Torch/C++', fontsize=9)
                    ax_ratio.set_ylim([0.9*min(ratio), 1.1*max(ratio)])
                    ax_ratio.grid(True, alpha=0.3)

                    if debug and var != 'PID':
                        print(f"\n{branch_name}.{var}, PID={pid_int} bins:")
                        print(f"  Bin edges: {bin_edges}, len(bin_edges)={len(bin_edges)}")
                        print(f"  C++ counts: {benchmark_counts}")
                        print(f"  TorchDelphes counts: {torch_counts}")
                        print(f"  Total C++ counts: {np.sum(benchmark_counts):.0f}")
                        print(f"  Total TorchDelphes counts: {np.sum(torch_counts):.0f}")
                        print(f"  Ratio (Torch/C++): {np.sum(torch_counts) / np.sum(benchmark_counts):.4f}")
                        
                        # Compute and print ratio statistics
                        valid_ratio = ratio[benchmark_counts > 0]
                        if len(valid_ratio) > 0:
                            print(f"  Ratio mean: {np.mean(valid_ratio):.4f}")
                            print(f"  Ratio std: {np.std(valid_ratio):.4f}")
                            print(f"  Ratio min: {np.min(valid_ratio):.4f}")
                            print(f"  Ratio max: {np.max(valid_ratio):.4f}")
                            
                
                # Add overall title with PID
                fig.suptitle(f'{branch_name} (PID={pid_int})', fontsize=16, fontweight='bold', y=0.98)
                
                # Save PID-specific combined figure
                pid_plot_file = branch_dir / f"pid_{pid_int}.png"
                plt.savefig(pid_plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                    
        else:
            print(f"  ℹ No PID field - skipping PID-specific plots (normal for Tower objects)")
            
    print(f"\n{'='*70}")
    print(f"✓ Validation complete! Plots saved to {output_dir}")
    print(f"{'='*70}")

def main(
    input_file: str, 
    output_file: str, 
    benchmark_file: str, 
    max_events: Optional[int] = None, 
    batch_size: int = 100, 
    debug: bool = False
) -> None:
    """Main processing function.
    
    Args:
        input_file: Path to input HepMC file
        output_file: Path to output ROOT file
        benchmark_file: Path to benchmark ROOT file
        max_events: Maximum number of events to process (None = all)
        batch_size: Number of events to process per batch (for GPU acceleration)
        debug: If True, print histogram bin counts and edges for debugging
    """
    
    print("\n" + "="*80)
    print("Parnassus.TorchDelphes Processing")
    print("="*80)
    print(f"\nInput:  {input_file}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {DEVICE}")

    # Set up dict for ROOT branches
    branches_torch_root = {}
    
    # ========================================================================
    # STEP 1: Load HepMC and convert to tensors
    # ========================================================================
    print("\n" + "="*80)
    print(f"STEP 1: Loading HepMC file and converting to tensors: {input_file}")
    print("="*80)
    
    genevent_tensors = hepmc_to_tensor(input_file, max_events).to(DEVICE)
    n_events = len(genevent_tensors)
    print(f"Loaded {n_events} events from HepMC")
    print(f"  Total stable particles: {sum(t.shape[0] for t in genevent_tensors)}")


    # ========================================================================
    # STEP 2: Apply ParticlePropagator
    # ========================================================================

    tic_torch = time.time()

    print("\n" + "="*80)
    print("STEP 2: Applying ParticlePropagator (batched)")
    print("="*80)

    genevent_tensors, pbp_tensors, pap_tensors, ch_tensors, el_tensors, mu_tensors = process_particle_propagator(genevent_tensors, batch_size=batch_size)
    
    # Extract expected event numbers from pap_tensors (most complete set of particles)
    # This ensures all branches have the same number of events
    all_pap = torch.cat([t for t in pap_tensors if t.shape[0] > 0], dim=0)
    expected_event_nums = sorted(set(all_pap[:, CMAP["EVENT_NUMBER"]].cpu().numpy().tolist()))
    
    branches_torch_root.update({
        'ParticleBeforeProp': tensor_to_root_dict([i.cpu() for i in pbp_tensors], 'ParticleBeforeProp', expected_event_nums),
        'ParticleAfterProp': tensor_to_root_dict([i.cpu() for i in pap_tensors], 'ParticleAfterProp', expected_event_nums),
        'ChargedHadron': tensor_to_root_dict([i.cpu() for i in ch_tensors], 'ChargedHadron', expected_event_nums),
        'Electron': tensor_to_root_dict([i.cpu() for i in el_tensors], 'Electron', expected_event_nums),
        'Muon': tensor_to_root_dict([i.cpu() for i in mu_tensors], 'Muon', expected_event_nums),
    })

    print(f"\nAfter ParticlePropagator: {len(genevent_tensors)} events")
    print(f"  Total ParticleAfterProp: {sum(t.shape[0] for t in pap_tensors)}")

    # ========================================================================
    # STEP 3: Apply (Tracking)Efficiency
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: Applying Efficiency modules (batched)")
    print("="*80)

    ch_filtered, el_filtered, mu_filtered = process_efficiency_pipeline(
        ch_tensors, el_tensors, mu_tensors
    )
    branches_torch_root.update({
        'ChargedHadronEfficiency': tensor_to_root_dict([i.cpu() for i in ch_filtered], 'ChargedHadronEfficiency', expected_event_nums),
        'ElectronEfficiency': tensor_to_root_dict([i.cpu() for i in el_filtered], 'ElectronEfficiency', expected_event_nums),
        'MuonEfficiency': tensor_to_root_dict([i.cpu() for i in mu_filtered], 'MuonEfficiency', expected_event_nums),
    })

    print("\n✓ Efficiency applied")

    # ========================================================================
    # STEP 4: Apply MomentumSmearing
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: Applying MomentumSmearing modules (batched)")
    print("="*80)

    ch_smeared, el_smeared, mu_smeared = process_smearing_pipeline(
        ch_filtered, el_filtered, mu_filtered
    )
    branches_torch_root.update({
        'ChargedHadronSmeared': tensor_to_root_dict([i.cpu() for i in ch_smeared], 'ChargedHadronSmeared', expected_event_nums),
        'ElectronSmeared': tensor_to_root_dict([i.cpu() for i in el_smeared], 'ElectronSmeared', expected_event_nums),
        'MuonSmeared': tensor_to_root_dict([i.cpu() for i in mu_smeared], 'MuonSmeared', expected_event_nums),
    })
    
    print("\n✓ MomentumSmearing applied")

    # ========================================================================
    # STEP 5: Apply TrackMerger
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 5: Applying TrackMerger (batched)")
    print("="*80)

    merged_tracks = process_merger_pipeline(
        ch_smeared, el_smeared, mu_smeared
    )
    branches_torch_root.update({
        'MergedTracks': tensor_to_root_dict([i.cpu() for i in merged_tracks], 'MergedTracks', expected_event_nums),
    })
    
    print("\n✓ TrackMerger applied")
    
    # ========================================================================
    # STEP 6: Apply SimpleCalorimeter
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 6: Applying SimpleCalorimeter (Steps 1 & 2)")
    print("="*80)
    
    ecal_results = process_ecal_pipeline(
        pap_tensors, merged_tracks
    )
    
    print("\n✓ SimpleCalorimeter Steps 1, 2, & 4 complete")
    
    # Validate intermediate outputs against C++
    script_dir = Path(__file__).parent
    if len(expected_event_nums) == 100:
        cpp_fractions_file = script_dir / "torch_delphes_validation" / "SimpleCalorimeter_CPP" / "energy_fractions_100.csv"
        cpp_towerhits_file = script_dir / "torch_delphes_validation" / "SimpleCalorimeter_CPP" / "tower_hits_100.csv"
        cpp_towerenergy_file = script_dir / "torch_delphes_validation" / "SimpleCalorimeter_CPP" / "tower_energy_100.csv"
    elif len(expected_event_nums) == 1000:
        cpp_fractions_file = script_dir / "torch_delphes_validation" / "SimpleCalorimeter_CPP" / "energy_fractions_1000.csv"
        cpp_towerhits_file = script_dir / "torch_delphes_validation" / "SimpleCalorimeter_CPP" / "tower_hits_1000.csv"
        cpp_towerenergy_file = script_dir / "torch_delphes_validation" / "SimpleCalorimeter_CPP" / "tower_energy_1000.csv"
    else: raise FileNotFoundError("expected_event_nums must be 100 or 1000 for validation")
    validation_dir = script_dir / "torch_delphes_validation" / "SimpleCalorimeter"
    validate_simple_cal(
        ecal_results, 
        str(cpp_fractions_file),
        str(cpp_towerhits_file),
        str(cpp_towerenergy_file),
        str(validation_dir)
    )
    
    # ========================================================================
    # STEP 7: Write final output
    # ========================================================================

    print(f"Writing {output_file}...")
    write_root_file(output_file, branches_torch_root)

    # ========================================================================
    # STEP 8: Print summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_ch_input = sum(t.shape[0] for t in ch_tensors)
    total_el_input = sum(t.shape[0] for t in el_tensors)
    total_mu_input = sum(t.shape[0] for t in mu_tensors)
    
    print(f"\nChargedHadrons:")
    print(f"  Input:      {total_ch_input}")

    
    print(f"\nElectrons:")
    print(f"  Input:      {total_el_input}")

    
    print(f"\nMuons:")
    print(f"  Input:      {total_mu_input}")

    print("\n" + "="*80)
    print("✓ ALL PROCESSING COMPLETE!")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 9: Validate Against C++ Delphes (Final ROOT branches)
    # ========================================================================
    
    # Determine benchmark file location
    script_dir = Path(__file__).parent
    validation_dir = script_dir / "torch_delphes_validation"
    
    if Path(benchmark_file).exists():
        print(f"\nBenchmark file: {benchmark_file}")
        print(f"Validation directory: {validation_dir}")
        validate_against_benchmark(output_file, benchmark_file, validation_dir, debug=debug)
    else:
        print(f"\n⚠ Benchmark file not found: {benchmark_file}")
        print("  Skipping validation. To enable validation, provide HZZ4l_4_0.root")
        print("  (Generated by C++ Delphes with delphes_card_CMS_4_0.tcl)")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parnassus TorchDelphes HepMC Processing")
    parser.add_argument(
        "--input", "-i", type=str, default="delphes_data/HZZ4l/HZZ4l_0.hepmc",
        help="Input HepMC file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="delphes_data/HZZ4l/HZZ4l_4_0_torch.root",
        help="Output ROOT file"
    )
    parser.add_argument(
        "--benchmark", "-bm", type=str, default="delphes_data/HZZ4l/HZZ4l_4_0.root",
        help="Benchmark ROOT file from C++ Delphes for validation (CMS_4_0 card with ECal)"
    )
    parser.add_argument(
        "--max-events", "-n", type=int, default=1000,
        help="Maximum number of events to process (default: 1000)"
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=100,
        help="Batch size for processing (default: 1000)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print histogram bin counts and edges for debugging"
    )
    return parser.parse_args()

if __name__ == "__main__":
    tic = time.time()
    args = parse_args()

    main(args.input, args.output, args.benchmark, max_events=args.max_events, batch_size=args.batch_size, debug=args.debug)

    toc = time.time()
    dur = toc - tic
    print(f"\n{'='*80}")
    print(f"Total execution time on {DEVICE}: {dur//60:.0f} minutes, {dur%60:.2f} seconds")
    print(f"{'='*80}\n")
