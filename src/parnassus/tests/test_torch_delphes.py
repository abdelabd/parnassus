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
        device=DEVICE
    )
    
    print(f"\nParticlePropagator (batch_size={batch_size})...")
    print(f"genevent_tensors.shape: {genevent_tensors.shape}")

    genevent_tensors_propagated = torch.zeros(genevent_tensors.shape, dtype=genevent_tensors.dtype)
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
        batch_events = genevent_tensors[batch_start:batch_end].to(DEVICE)
        batch_size_actual = batch_events.shape[0]
        
        # Propagate particles (batched)
        particles = batch_events.reshape(-1, n_dim) # Flatten to (B*N, N_FEATURES)
        particles_before_prop_batch = particles.clone()
        particles_after_prop_batch, _, charged_hadrons_batch, electrons_batch, muons_batch = propagator(particles)

        # Update genevnt_tensors
        genevent_tensors_propagated[batch_start:batch_end] = particles_after_prop_batch.reshape(batch_size_actual, n_part, n_dim).cpu()
        
        # For debugging: Collect ParticleBeforeProp, ParticleAfterProp, ChargedHadron, Electron, and Muon tensors after propagation
        mask = particles_after_prop_batch[:, CMAP["IS_NOT_PAD"]] * particles_after_prop_batch[:, CMAP["PASS_PROP"]]
        
        # ParticleBeforeProp
        pbp_mask = particles_before_prop_batch[:, CMAP["IS_NOT_PAD"]].float()
        pbp_tensors.append(particles_before_prop_batch[pbp_mask > 0.5].clone().to(torch.float32).cpu())

        # ParticleAfterProp
        pap_mask = particles_after_prop_batch[:, CMAP["IS_NOT_PAD"]].float() * particles_after_prop_batch[:, CMAP["PASS_PROP"]].float()
        pap_tensors.append(particles_after_prop_batch[pap_mask > 0.5].to(torch.float32).cpu())

        # ChargedHadron
        ch_tensors.append(charged_hadrons_batch)

        # Electron
        el_tensors.append(electrons_batch)

        # Muon
        mu_tensors.append(muons_batch)

    return genevent_tensors_propagated, pbp_tensors, pap_tensors, ch_tensors, el_tensors, mu_tensors

def process_efficiency_pipeline(
    genevent_tensors: torch.Tensor, 
    batch_size: int = 100
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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

    n_event, n_part, n_dim = genevent_tensors.shape
    
    # Initialize efficiency modules
    ch_eff_module = Efficiency(
        efficiency_formula='charged_hadron_cms',
        device=DEVICE
    )
    el_eff_module = Efficiency(
        efficiency_formula='electron_cms',
        device=DEVICE
    )
    mu_eff_module = Efficiency(
        efficiency_formula='muon_cms',
        device=DEVICE
    )

    genevent_tensors_eff = []
    # Collect charged_hadron, electron, muon tensors after propagation (for intermediate testing and validation)
    ch_tensors_eff = []
    el_tensors_eff = []
    mu_tensors_eff = []
    # Process in batches
    for batch_start in tqdm(range(0, n_event, batch_size)):
        batch_end = min(batch_start + batch_size, n_event)

        # Flatten to (B*N, 15)
        batch_events = genevent_tensors[batch_start:batch_end].to(DEVICE)
        batch_size = batch_events.shape[0]

        # Send through all 3 efficiency modules (batched)
        particles = batch_events.reshape(-1, n_dim) # Flatten to (B*N, 15)
        particles = ch_eff_module(particles)
        particles = el_eff_module(particles)
        particles = mu_eff_module(particles)

        genevent_tensors_eff.append(particles.reshape(batch_size, n_part, n_dim).cpu())

        mask = particles[:, CMAP["IS_NOT_PAD"]] * particles[:, CMAP["PASS_PROP"]] * particles[:, CMAP["PASS_EFF"]]
        charged_hadron_pid_mask = mask * Efficiency()._charged_hadron_pdg_filter(particles).float()
        electron_pid_mask = mask * Efficiency()._electron_pdg_filter(particles).float()
        muon_pid_mask = mask * Efficiency()._muon_pdg_filter(particles).float()

        ch_tensors_eff.append(particles[charged_hadron_pid_mask > 0.5].to(torch.float32).cpu())
        el_tensors_eff.append(particles[electron_pid_mask > 0.5].to(torch.float32).cpu())
        mu_tensors_eff.append(particles[muon_pid_mask > 0.5].to(torch.float32).cpu())

    # Stack all event tensors into a single tensor
    genevent_tensors_eff = torch.cat(genevent_tensors_eff, dim=0)

    return genevent_tensors_eff, ch_tensors_eff, el_tensors_eff, mu_tensors_eff

def process_smearing_pipeline(
    genevent_tensors: torch.Tensor, 
    batch_size: int = 100
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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

    n_event, n_part, n_dim = genevent_tensors.shape
    
    # Initialize smearing modules
    ch_smear_module = MomentumSmearing(
        resolution_formula='charged_hadron_cms',
        device=DEVICE
    )
    
    el_smear_module = MomentumSmearing(
        resolution_formula='electron_cms',
        device=DEVICE
    )
    
    mu_smear_module = MomentumSmearing(
        resolution_formula='muon_cms',
        device=DEVICE
    )

    genevent_tensors_smeared = []
    ch_tensors_smeared = []
    el_tensors_smeared = []
    mu_tensors_smeared = []
    for batch_start in tqdm(range(0, n_event, batch_size)):
        batch_end = min(batch_start + batch_size, n_event)

        # Flatten to (B*N, 15)
        batch_events = genevent_tensors[batch_start:batch_end].to(DEVICE)
        batch_size = batch_events.shape[0]

        # Send through all 3 smearing modules (batched)
        particles = batch_events.reshape(-1, n_dim) # Flatten to (B*N, 15)
        particles = ch_smear_module(particles)
        particles = el_smear_module(particles)
        particles = mu_smear_module(particles)
        n_dim_new = particles.shape[1]

        genevent_tensors_smeared.append(particles.reshape(batch_size, n_part, n_dim_new).cpu())

        mask = particles[:, CMAP["IS_NOT_PAD"]] * particles[:, CMAP["PASS_PROP"]] * particles[:, CMAP["PASS_EFF"]]
        charged_hadron_pid_mask = mask * Efficiency()._charged_hadron_pdg_filter(particles).float()
        electron_pid_mask = mask * Efficiency()._electron_pdg_filter(particles).float()
        muon_pid_mask = mask * Efficiency()._muon_pdg_filter(particles).float()

        ch_tensors_smeared.append(particles[charged_hadron_pid_mask > 0.5].to(torch.float32).cpu())
        el_tensors_smeared.append(particles[electron_pid_mask > 0.5].to(torch.float32).cpu())
        mu_tensors_smeared.append(particles[muon_pid_mask > 0.5].to(torch.float32).cpu())

    # Stack all event tensors into a single tensor
    genevent_tensors_smeared = torch.cat(genevent_tensors_smeared, dim=0)

    return genevent_tensors_smeared, ch_tensors_smeared, el_tensors_smeared, mu_tensors_smeared

def process_merger_pipeline(
    genevent_tensors: torch.Tensor, 
    batch_size: int = 100
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Apply TrackMerger to combine charged hadrons, electrons, and muons.
    
    Args:
        genevent_tensors: Tensor of shape (N_events, N_particles, D)
        batch_size: Number of events to process in each batch
        
    Returns:
        genevent_tensors: Tensor of shape (N_events, N_particles, D+1) with PASS_MERGER column
        track_tensors: List of track tensors (for validation)
    """
    
    n_event, n_part, n_dim = genevent_tensors.shape
    
    # Initialize TrackMerger module
    merger = Merger(
        particle_types=['charged_hadron', 'electron', 'muon'],
        device=DEVICE
    )
    
    print(f"\nTrackMerger (batch_size={batch_size})...")
    print(f"genevent_tensors.shape: {genevent_tensors.shape}")
    
    genevent_tensors_merged = []
    track_tensors = []  # For validation
    
    # Process in batches
    for batch_start in tqdm(range(0, n_event, batch_size)):
        batch_end = min(batch_start + batch_size, n_event)
        
        # Extract batch
        batch_events = genevent_tensors[batch_start:batch_end].to(DEVICE)
        batch_size_actual = batch_events.shape[0]
        
        # Apply merger (operates on batched input directly)
        batch_merged = merger(batch_events)
        
        genevent_tensors_merged.append(batch_merged.cpu())
        
        # Extract valid tracks for this batch (for validation)
        # Flatten batch to (B*N, D)
        particles = batch_merged.reshape(-1, n_dim)
        
        # Mask for particles that passed merger
        track_mask = (
            particles[:, CMAP["IS_NOT_PAD"]] *
            particles[:, CMAP["PASS_PROP"]] *
            particles[:, CMAP["PASS_EFF"]] *
            particles[:, CMAP["PASS_MERGER"]]
        )
        
        track_tensors.append(particles[track_mask > 0.5].to(torch.float32).cpu())
    
    # Stack all event tensors into a single tensor
    genevent_tensors_merged = torch.cat(genevent_tensors_merged, dim=0)
    
    return genevent_tensors_merged, track_tensors

def process_ecal_pipeline(
    genevent_tensors: torch.Tensor, 
    batch_size: int = 100
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Apply SimpleCalorimeter (ECal) module.
    
    Args:
        genevent_tensors: Tensor of shape (N_events, N_particles, D)
        batch_size: Number of events to process in each batch
        
    Returns:
        genevent_tensors: Tensor of shape (N_events, N_particles + N_towers, D+3) with ECAL masks
        ecal_towers: List of tower tensors per event (for validation)
        eflow_tracks: List of eflow track tensors per event (for validation)
        eflow_photons: List of eflow photon tensors per event (for validation)
    """
    
    n_event, n_part, n_dim = genevent_tensors.shape
    
    # Create eta and phi bins from CMS card
    # Barrel: |eta| < 1.5, 0.02 x 0.02 resolution
    eta_bins_barrel = [i * 0.0174 for i in range(-85, 87)]  # -1.479 to 1.496
    
    # Endcap: 1.5 < |eta| < 3.0, 0.02 x 0.02 resolution
    eta_bins_endcap_neg = [-2.958 + i * 0.0174 for i in range(1, 85)]  # -2.941 to 1.479
    eta_bins_endcap_pos = [1.4964 + i * 0.0174 for i in range(1, 85)]  # 1.514 to 2.958
    
    # HF: 3.0 < |eta| < 5.0
    eta_bins_hf = [-5, -4.7, -4.525, -4.35, -4.175, -4, -3.825, -3.65, -3.475, -3.3, -3.125, -2.958,
                   3.125, 3.3, 3.475, 3.65, 3.825, 4, 4.175, 4.35, 4.525, 4.7, 5]
    
    # Combine all eta bins (sorted)
    eta_bins = sorted(set(eta_bins_hf + eta_bins_endcap_neg + eta_bins_barrel + eta_bins_endcap_pos))
    
    # Phi bins: uniform from -pi to pi
    phi_bins = [i * np.pi / 180.0 for i in range(-180, 181)]

    # energy fractions
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
    
    # Initialize ECal module
    eflowtrack_module = SimpleCalorimeter.EFlowTrack(
        eta_bins=eta_bins,
        phi_bins=phi_bins,
        energy_min=0.5,
        energy_sig_min=2.0,
        resolution_formula='ecal_cms',
        is_ecal=True,
        smear_tower_center=True,
        energy_fractions=energy_fractions,
        max_towers_per_event=500,
        device=DEVICE
    )
    tower_module = SimpleCalorimeter.Tower(
        eta_bins=eta_bins,
        phi_bins=phi_bins,
        energy_min=0.5,
        energy_sig_min=2.0,
        resolution_formula='ecal_cms',
        is_ecal=True,
        smear_tower_center=True,
        energy_fractions=energy_fractions,
        max_towers_per_event=500,
        device=DEVICE
    )
    eflowphoton_module = SimpleCalorimeter.EFlowPhoton(
        eta_bins=eta_bins,
        phi_bins=phi_bins,
        energy_min=0.5,
        energy_sig_min=2.0,
        resolution_formula='ecal_cms',
        is_ecal=True,
        smear_tower_center=True,
        energy_fractions=energy_fractions,
        max_towers_per_event=500,
        device=DEVICE
    )
    
    print(f"\nECal (batch_size={batch_size})...")
    print(f"genevent_tensors.shape: {genevent_tensors.shape}")
    print(f"Number of eta bins: {len(eta_bins)}")
    print(f"Number of phi bins: {len(phi_bins)}")
    
    genevent_tensors_ecal = []
    all_ecal_towers = []
    all_eflow_tracks = []
    all_eflow_photons = []
    
    # Process in batches
    for batch_start in tqdm(range(0, n_event, batch_size)):
        batch_end = min(batch_start + batch_size, n_event)
        
        # Extract batch
        batch_events = genevent_tensors[batch_start:batch_end].to(DEVICE)
        
        # Apply ECal

        # 1. EFlowTracks
        batch_eflow_tracks = eflowtrack_module(batch_events)
        batch_tracks = torch.cat(batch_eflow_tracks, dim=0).cpu() if batch_eflow_tracks else torch.empty(0, 5)
        all_eflow_tracks.append(batch_tracks.to(torch.float32))
        
        # 2. EFlowPhotons
        batch_photons = eflowphoton_module(batch_events)
        batch_photons = torch.cat(batch_photons, dim=0).cpu() if batch_photons else torch.empty(0, 5)
        all_eflow_photons.append(batch_photons.to(torch.float32))

        # 3. Towers
        batch_ecal, batch_towers = tower_module(batch_events)
        batch_towers = torch.cat(batch_towers, dim=0).cpu() if batch_towers else torch.empty(0, 5)
        all_ecal_towers.append(batch_towers.to(torch.float32))

        genevent_tensors_ecal.append(batch_ecal.cpu())
        
    # Stack all event tensors into a single tensor
    genevent_tensors_ecal = torch.cat(genevent_tensors_ecal, dim=0)
    
    print(f"\nECal output shape: {genevent_tensors_ecal.shape}")
    print(f"Total towers: {sum(t.shape[0] for t in all_ecal_towers)}")
    print(f"Total eflow tracks: {sum(t.shape[0] for t in all_eflow_tracks)}")
    print(f"Total eflow photons: {sum(t.shape[0] for t in all_eflow_photons)}")
    
    return genevent_tensors_ecal, all_ecal_towers, all_eflow_tracks, all_eflow_photons


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
        ('ParticleBeforeProp', track_kinematic_vars),
        ('ParticleAfterProp', track_kinematic_vars),
        ('ChargedHadron', track_kinematic_vars),
        ('Electron', track_kinematic_vars),
        ('Muon', track_kinematic_vars),
        ('ChargedHadronEfficiency', track_kinematic_vars),
        ('ElectronEfficiency', track_kinematic_vars),
        ('MuonEfficiency', track_kinematic_vars),
        ('ChargedHadronSmeared', track_kinematic_vars),
        ('ElectronSmeared', track_kinematic_vars),
        ('MuonSmeared', track_kinematic_vars),
        ('MergedTracks', track_kinematic_vars),
        ('ECalTower', tower_kinematic_vars),
        ('ECal_EFlowTrack', track_kinematic_vars),
        ('EFlowPhoton', tower_kinematic_vars)
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
                # if debug and (branch_name in {"ECal_EFlowTrack", "ECalTower", "EFlowPhoton"}) and var != 'PID':
                if debug and (branch_name in {"ChargedHadron", "Electron", "Muon"}) and var != 'PID':
                    print(f"\n{branch_name}.{var} bins:")
                    print(f"  Bin edges: {bin_edges}, len(bin_edges)={len(bin_edges)}")
                    print(f"  C++ counts: {benchmark_counts}")
                    print(f"  TorchDelphes counts: {torch_counts}")
                    print(f"  Total C++ counts: {np.sum(benchmark_counts):.0f}")
                    print(f"  Total TorchDelphes counts: {np.sum(torch_counts):.0f}")
                    print(f"  Ratio (Torch/C++): {np.sum(torch_counts) / np.sum(benchmark_counts):.4f}")
                    
                    # Compute and print ratio statistics
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ratio = np.divide(
                        torch_counts, benchmark_counts, 
                        out=np.ones_like(torch_counts), 
                        where=benchmark_counts > 0
                    )
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
                    ax_ratio.set_ylim([0.5, 1.5])  # Focus on reasonable ratio range
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
                        linewidth=2, label='C++ Delphes', density=False
                    )
                    torch_counts, _, _ = ax_hist.hist(
                        torch_np, bins=bins, histtype='step', color='blue', 
                        linewidth=2, label='Parnassus.TorchDelphes', density=False
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
                        
                
                # Add overall title with PID
                fig.suptitle(f'{branch_name} (PID={pid_int})', fontsize=16, fontweight='bold', y=0.98)
                
                # Save PID-specific combined figure
                pid_plot_file = branch_dir / f"pid_{pid_int}.png"
                plt.savefig(pid_plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                    
        else:
            print(f"  ℹ No PID field - skipping PID-specific plots (normal for Tower objects)")
    
        ### 4. Tower energies at eta=phi=0 for ECalTower and EFlowPhoton
        if branch_name in ["ECalTower", "EFlowPhoton"]:
            print(f"\n\nPlotting tower energies at eta=phi=0")

            # Extract data
            key_eta = f"{branch_name}/{branch_name}.Eta"
            key_phi = f"{branch_name}/{branch_name}.Phi"
            key_E = f"{branch_name}/{branch_name}.E"
            eta_max = 0.1
            phi_max = 0.1
            
            # Select towers
            torch_abs_eta = ak.to_numpy(abs(ak.flatten(torch_tree[key_eta].array())))
            torch_abs_phi = ak.to_numpy(abs(ak.flatten(torch_tree[key_phi].array())))
            torch_E = ak.to_numpy(ak.flatten(torch_tree[key_E].array()))
            torch_mask_eta = torch_abs_eta < eta_max
            torch_mask_phi = torch_abs_phi < phi_max
            torch_mask_eta_phi = torch_mask_eta & torch_mask_phi
            torch_E_sel_eta = torch_E[torch_mask_eta]
            torch_E_sel_phi = torch_E[torch_mask_phi]
            torch_E_sel_eta_phi = torch_E[torch_mask_eta_phi]
            print(f"torch_E.shape: {torch_E.shape}")
            print(f"torch_E_sel_eta.shape: {torch_E_sel_eta.shape}")
            print(f"torch_E_sel_phi.shape: {torch_E_sel_phi.shape}")
            print(f"torch_E_sel_eta_phi.shape: {torch_E_sel_eta_phi.shape}")
            print(f"min(torch_E_sel_eta): {np.min(torch_E_sel_eta)}, max(torch_E_sel_eta): {np.max(torch_E_sel_eta)}")
            print(f"min(torch_E_sel_phi): {np.min(torch_E_sel_phi)}, max(torch_E_sel_phi): {np.max(torch_E_sel_phi)}")
            print(f"min(torch_E_sel_eta_phi): {np.min(torch_E_sel_eta_phi)}, max(torch_E_sel_eta_phi): {np.max(torch_E_sel_eta_phi)}")


            benchmark_abs_eta = ak.to_numpy(abs(ak.flatten(benchmark_tree[key_eta].array())))
            benchmark_abs_phi = ak.to_numpy(abs(ak.flatten(benchmark_tree[key_phi].array())))
            benchmark_E = ak.to_numpy(ak.flatten(benchmark_tree[key_E].array()))
            benchmark_mask_eta = benchmark_abs_eta < eta_max
            benchmark_mask_phi = benchmark_abs_phi < phi_max
            benchmark_mask_eta_phi = benchmark_mask_eta & benchmark_mask_phi
            benchmark_E_sel_eta = benchmark_E[benchmark_mask_eta]
            benchmark_E_sel_phi = benchmark_E[benchmark_mask_phi]
            benchmark_E_sel_eta_phi = benchmark_E[benchmark_mask_eta_phi]
            print(f"\nbenchmark_E.shape: {benchmark_E.shape}")
            print(f"benchmark_E_sel_eta.shape: {benchmark_E_sel_eta.shape}")
            print(f"benchmark_E_sel_phi.shape: {benchmark_E_sel_phi.shape}")
            print(f"benchmark_E_sel_eta_phi.shape: {benchmark_E_sel_eta_phi.shape}")
            print(f"min(benchmark_E_sel_eta): {np.min(benchmark_E_sel_eta)}, max(benchmark_E_sel_eta): {np.max(benchmark_E_sel_eta)}")
            print(f"min(benchmark_E_sel_phi): {np.min(benchmark_E_sel_phi)}, max(benchmark_E_sel_phi): {np.max(benchmark_E_sel_phi)}")

            try:
                print(f"min(benchmark_E_sel_eta_phi): {np.min(benchmark_E_sel_eta_phi)}, max(benchmark_E_sel_eta_phi): {np.max(benchmark_E_sel_eta_phi)}")
            except ValueError:
                print("No entries found for benchmark_E_sel_eta_phi")

            for sel in "eta", "phi", "eta_phi":
                if sel == "eta":
                    torch_E_selected = torch_E_sel_eta
                    benchmark_E_selected = benchmark_E_sel_eta
                    print_statement = f"\nSelection: |eta| < {eta_max}"
                    title_str = f"|eta| < {eta_max}"
                    fig_name = f"energy_eta{eta_max}.png"
                elif sel == "phi":
                    torch_E_selected = torch_E_sel_phi
                    benchmark_E_selected = benchmark_E_sel_phi
                    print_statement = f"\nSelection: |phi| < {phi_max}"
                    title_str = f"|phi| < {phi_max}"
                    fig_name = f"energy_phi{phi_max}.png"
                else:  # eta_phi
                    torch_E_selected = torch_E_sel_eta_phi
                    benchmark_E_selected = benchmark_E_sel_eta_phi
                    print_statement = f"\nSelection: |eta| < {eta_max}, |phi| < {phi_max}"
                    title_str = f"|eta| < {eta_max}, |phi| < {phi_max}"
                    fig_name = f"energy_eta{eta_max}_phi{phi_max}.png"

                print(print_statement)
                print(f"  TorchDelphes: {len(torch_E_selected)} towers")
                print(f"  C++ Delphes: {len(benchmark_E_selected)} towers")
                fig = plt.figure(figsize=(10, 8))
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax_hist = fig.add_subplot(gs[0])
                ax_ratio = fig.add_subplot(gs[1], sharex=ax_hist)
                
                # Plot histogram of selected tower energies
                try:
                    if np.min(benchmark_E_selected) < np.min(torch_E_selected):
                        benchmark_counts, bin_edges, _ = ax_hist.hist(benchmark_E_selected, bins=50, histtype='stepfilled', alpha=0.5, color='orange',
                                linewidth=2, label=f'C++ Delphes; {len(benchmark_E_selected)} towers', density=False)
                        torch_counts, _, _ = ax_hist.hist(torch_E_selected, bins=bin_edges, histtype='step', color='blue',
                                linewidth=2, label=f'Parnassus.TorchDelphes; {len(torch_E_selected)} towers', density=False)
                    else:
                        torch_counts, bin_edges, _ = ax_hist.hist(torch_E_selected, bins=50, histtype='step', color='blue',
                                linewidth=2, label=f'Parnassus.TorchDelphes; {len(torch_E_selected)} towers', density=False)
                        benchmark_counts, _, _ = ax_hist.hist(benchmark_E_selected, bins=bin_edges, histtype='stepfilled', alpha=0.5, color='orange',
                                linewidth=2, label=f'C++ Delphes; {len(benchmark_E_selected)} towers', density=False)
                    print(f"benchmark_counts.shape: {benchmark_counts.shape}, torch_counts.shape: {torch_counts.shape}")

                    ax_hist.set_xticks(ticks=[], labels=[])
                    ax_hist.set_title(f'{branch_name}: Tower Energies at {title_str}', fontsize=14, fontweight='bold')
                    ax_hist.set_ylabel('Counts', fontsize=12)
                    ax_hist.legend(fontsize=11)
                    ax_hist.grid(True, alpha=0.3)

                    # Compute ratio
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ratio = torch_counts/benchmark_counts
                    ax_ratio.axhline(y=1.0, color='orange', linewidth=2)
                    ax_ratio.plot(bin_centers, ratio, color='blue', markersize=4, linewidth=2)
                    ax_ratio.set_xlabel('Tower Energy', fontsize=12)
                    ax_ratio.set_ylabel('Torch / C++', fontsize=10)
                    ax_ratio.grid(True, alpha=0.3)

                    fig.savefig("{}/{}_{}".format(branch_dir, branch_name, fig_name), dpi=150)
                    print("\n\n")
                except ValueError:
                    print(f"  ✗ Error plotting tower energies at {title_str}: not enough events pass selection")
                    continue
            
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
    
    # ========================================================================
    # STEP 1: Load HepMC and convert to tensors
    # ========================================================================
    print("\n" + "="*80)
    print(f"STEP 1: Loading HepMC file and converting to tensors: {input_file}")
    print("="*80)
    
    genevent_tensors = hepmc_to_tensor(input_file, max_events)
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

    print(f"\nAfter ParticlePropagator: {len(genevent_tensors)} events")
    print(f"  Total ParticleAfterProp: {sum(t.shape[0] for t in pap_tensors)}")

    # ========================================================================
    # STEP 3: Apply tracking efficiency
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: Applying Efficiency modules (batched)")
    print("="*80)

    genevent_tensors, ch_filtered, el_filtered, mu_filtered = process_efficiency_pipeline(
        genevent_tensors, batch_size=batch_size
    )

    print("\n✓ Efficiency applied")

    # ========================================================================
    # STEP 4: Apply momentum smearing
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: Applying MomentumSmearing modules (batched)")
    print("="*80)

    genevent_tensors, ch_smeared, el_smeared, mu_smeared = process_smearing_pipeline(
        genevent_tensors, batch_size=batch_size
    )
    
    print("\n✓ MomentumSmearing applied")

    # ========================================================================
    # STEP 5: Apply TrackMerger
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 5: Applying TrackMerger (batched)")
    print("="*80)

    genevent_tensors, track_merged = process_merger_pipeline(
        genevent_tensors, batch_size=batch_size
    )
    
    print("\n✓ TrackMerger applied")
    
    # ========================================================================
    # STEP 6: Apply ECal (SimpleCalorimeter)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 6: Applying ECal (SimpleCalorimeter) (batched)")
    print("="*80)
    
    genevent_tensors, ecal_towers, eflow_tracks, eflow_photons = process_ecal_pipeline(
        genevent_tensors, batch_size=batch_size
    )
    
    print("\n✓ ECal applied")
    
    toc_torch = time.time()
    dur_torch = toc_torch - tic_torch
    print(f"\n\nTorch duration: {dur_torch//60:.0f} minutes, {dur_torch%60:.2f} seconds\n\n")

    # ========================================================================
    # STEP 7: Write final output
    # ========================================================================

    print(f"Writing {output_file}...")
    branches_v3_2 = {
        'ParticleBeforeProp': tensor_to_root_dict(pbp_tensors, 'ParticleBeforeProp'),
        'ParticleAfterProp': tensor_to_root_dict(pap_tensors, 'ParticleAfterProp'),
        'ChargedHadron': tensor_to_root_dict(ch_tensors, 'ChargedHadron'),
        'Electron': tensor_to_root_dict(el_tensors, 'Electron'),
        'Muon': tensor_to_root_dict(mu_tensors, 'Muon'),
        'ChargedHadronEfficiency': tensor_to_root_dict(ch_filtered, 'ChargedHadronEfficiency'),
        'ElectronEfficiency': tensor_to_root_dict(el_filtered, 'ElectronEfficiency'),
        'MuonEfficiency': tensor_to_root_dict(mu_filtered, 'MuonEfficiency'),
        'ChargedHadronSmeared': tensor_to_root_dict(ch_smeared, 'ChargedHadronSmeared'),
        'ElectronSmeared': tensor_to_root_dict(el_smeared, 'ElectronSmeared'),
        'MuonSmeared': tensor_to_root_dict(mu_smeared, 'MuonSmeared'),
        'MergedTracks': tensor_to_root_dict(track_merged, 'MergedTracks'),
        'ECalTower': tensor_to_root_dict(ecal_towers, 'ECalTower'),
        'ECal_EFlowTrack': tensor_to_root_dict(eflow_tracks, 'ECal_EFlowTrack'),
        'EFlowPhoton': tensor_to_root_dict(eflow_photons, 'EFlowPhoton')
    }
    write_root_file(output_file, branches_v3_2)

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

    
    print(f"\nECal:")
    print(f"  Towers:        {sum(t.shape[0] for t in ecal_towers)}")
    print(f"  EFlow Tracks:  {sum(t.shape[0] for t in eflow_tracks)}")
    print(f"  EFlow Photons: {sum(t.shape[0] for t in eflow_photons)}")
    
    print("\n" + "="*80)
    print("✓ ALL PROCESSING COMPLETE!")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 8: Validate Against C++ Delphes
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
        print("  Skipping validation. To enable validation, provide HZZ4l_5_0.root")
        print("  (Generated by C++ Delphes with delphes_card_CMS_5_0.tcl)")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parnassus TorchDelphes HepMC Processing")
    parser.add_argument(
        "--input", "-i", type=str, default="delphes_data/HZZ4l/HZZ4l_0.hepmc",
        help="Input HepMC file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="delphes_data/HZZ4l/HZZ4l_5_0_torch.root",
        help="Output ROOT file"
    )
    parser.add_argument(
        "--benchmark", "-bm", type=str, default="delphes_data/HZZ4l/HZZ4l_5_0.root",
        help="Benchmark ROOT file from C++ Delphes for validation (CMS_5_0 card with ECal)"
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
