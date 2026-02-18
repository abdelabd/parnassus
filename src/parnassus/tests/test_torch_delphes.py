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

from parnassus.torch_delphes import Efficiency, EFlowMerger, Merger, MomentumSmearing, ParticlePropagator, SimpleCalorimeter
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
        pbp_tensors.append(particles_before_prop_batch[pbp_mask > 0.5].clone())

        # ParticleAfterProp
        pap_mask = particles_after_prop_batch[:, CMAP["IS_NOT_PAD"]].float() * particles_after_prop_batch[:, CMAP["PASS_PROP"]].float()
        pap_tensors.append(particles_after_prop_batch[pap_mask > 0.5].clone())

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

        ch_tensors_eff.append(ch_batch_out)
        el_tensors_eff.append(el_batch_out)
        mu_tensors_eff.append(mu_batch_out)

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

        ch_tensors_smeared.append(ch_batch_out)
        el_tensors_smeared.append(el_batch_out)
        mu_tensors_smeared.append(mu_batch_out)

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
        track_tensors.append(tracks_batch_out)
    
    return track_tensors


def process_ecal_pipeline(
    pap_tensors: List[torch.Tensor],
    merged_tracks: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
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

    ecal = SimpleCalorimeter(
        eta_bins=eta_bins,
        phi_bins=phi_bins_per_eta,
        energy_min=0.5,
        energy_sig_min=2.0,
        energy_fractions=energy_fractions,
        resolution_formula='ecal_cms',
        is_ecal=True,
        smear_tower_center=True  # Match C++ Delphes: SmearTowerCenter true
    ).to(DEVICE)
    
    # Process in batches
    eflow_tracks = []
    towers = []
    eflow_photons = []
    for batch_particles, batch_tracks in zip(pap_tensors, merged_tracks):
        event_numbers = torch.unique(batch_particles[:, CMAP["EVENT_NUMBER"]])
        eflow_tracks_batch = []
        towers_batch = []
        eflow_photons_batch = []

        # forward method takes a single event, for now
        for event_num in tqdm(event_numbers):
            event_mask_particles = (batch_particles[:, CMAP["EVENT_NUMBER"]] == event_num)
            event_mask_tracks = (batch_tracks[:, CMAP["EVENT_NUMBER"]] == event_num)
            particles = batch_particles[event_mask_particles]
            tracks = batch_tracks[event_mask_tracks]
            
            # Run forward pass
            eflow_tracks_event, towers_event, eflow_photons_event = ecal(particles, tracks)

            eflow_tracks_batch.append(eflow_tracks_event)
            towers_batch.append(towers_event)
            eflow_photons_batch.append(eflow_photons_event)
        
        eflow_tracks.append(torch.cat(eflow_tracks_batch, dim=0))
        towers.append(torch.cat(towers_batch, dim=0))
        eflow_photons.append(torch.cat(eflow_photons_batch, dim=0))

    return eflow_tracks, towers, eflow_photons

def process_hcal_pipeline(
    pap_tensors: List[torch.Tensor],
    ecal_eflow_tracks: List[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Apply HCal SimpleCalorimeter to propagated particles and ECal EFlow tracks.
    
    The HCal takes:
    - ParticleInputArray: ParticlePropagator/stableParticles (same as ECal)
    - TrackInputArray: ECal/eflowTracks (output from ECal)
    
    Args:
        pap_tensors: List of particle tensors after propagation (one per batch)
        ecal_eflow_tracks: List of EFlow track tensors from ECal (one per event)
        
    Returns:
        Dict with tower_tensors, eflow_neutral_hadron_tensors, eflow_track_tensors
        Each is a list of tensors (one per event)
    """
    
    energy_fractions = {
        0: 1.0,        # default (hadrons) - full HCAL response
        11: 0.0,       # electrons (no HCAL response - already absorbed by ECAL)
        22: 0.0,       # photons (no HCAL response)
        111: 0.0,      # pi0 (no HCAL response)
        12: 0.0,       # neutrino (electron)
        13: 0.0,       # muon
        14: 0.0,       # neutrino (muon)
        16: 0.0,       # neutrino (tau)
        1000022: 0.0,  # neutralino
        1000023: 0.0,  # neutralino
        1000025: 0.0,  # neutralino
        1000035: 0.0,  # neutralino
        1000045: 0.0,  # neutralino
        310: 0.7,      # K0short (70% HCAL)
        3122: 0.7,     # Lambda (70% HCAL)
    }
    
    
    eta_phi_map = {}  # eta -> set of phi bin edges
    
    # 5 degrees towers (barrel): phi bins -36 to 36 in steps of pi/36
    phi_bins_5deg = [i * np.pi / 36.0 for i in range(-36, 37)]
    barrel_etas = [-1.566, -1.479, -1.392, -1.305, -1.218, -1.131, -1.044, -0.957, -0.87, -0.783, 
                   -0.696, -0.609, -0.522, -0.435, -0.348, -0.261, -0.174, -0.087, 0, 
                   0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696, 0.783, 0.87, 
                   0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479, 1.566, 1.653]
    for eta in barrel_etas:
        if eta not in eta_phi_map:
            eta_phi_map[eta] = set()
        eta_phi_map[eta].update(phi_bins_5deg)
    
    # 10 degrees towers (endcap): phi bins -18 to 18 in steps of pi/18
    phi_bins_10deg = [i * np.pi / 18.0 for i in range(-18, 19)]
    endcap_etas = [-4.35, -4.175, -4, -3.825, -3.65, -3.475, -3.3, -3.125, -2.95, -2.868, 
                   -2.65, -2.5, -2.322, -2.172, -2.043, -1.93, -1.83, -1.74, -1.653, 
                   1.74, 1.83, 1.93, 2.043, 2.172, 2.322, 2.5, 2.65, 2.868, 2.95, 
                   3.125, 3.3, 3.475, 3.65, 3.825, 4, 4.175, 4.35, 4.525]
    for eta in endcap_etas:
        if eta not in eta_phi_map:
            eta_phi_map[eta] = set()
        eta_phi_map[eta].update(phi_bins_10deg)
    
    # 20 degrees towers (forward): phi bins -9 to 9 in steps of pi/9
    phi_bins_20deg = [i * np.pi / 9.0 for i in range(-9, 10)]
    forward_etas = [-5, -4.7, -4.525, 4.7, 5]
    for eta in forward_etas:
        if eta not in eta_phi_map:
            eta_phi_map[eta] = set()
        eta_phi_map[eta].update(phi_bins_20deg)
    
    # Convert to sorted lists (matching C++ behavior)
    eta_bins = sorted(eta_phi_map.keys())
    phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

    hcal = SimpleCalorimeter(
        eta_bins=eta_bins,
        phi_bins=phi_bins_per_eta,
        energy_min=1.0,          # HCal has higher threshold
        energy_sig_min=1.0,      # HCal has lower significance threshold
        energy_fractions=energy_fractions,
        resolution_formula='hcal_cms',
        is_ecal=False,
        smear_tower_center=True
    ).to(DEVICE)
    
    # Process in batches
    eflow_tracks = []
    towers = []
    eflow_neutral_hadrons = []
    for batch_particles, batch_tracks in zip(pap_tensors, ecal_eflow_tracks):
        event_numbers = torch.unique(batch_particles[:, CMAP["EVENT_NUMBER"]])
        eflow_tracks_batch = []
        towers_batch = []
        eflow_neutral_hadrons_batch = []

        # forward method takes a single event, for now
        for event_num in tqdm(event_numbers):
            event_mask_particles = (batch_particles[:, CMAP["EVENT_NUMBER"]] == event_num)
            event_mask_tracks = (batch_tracks[:, CMAP["EVENT_NUMBER"]] == event_num)
            
            particles = batch_particles[event_mask_particles]
            tracks = batch_tracks[event_mask_tracks]
            
            # Run forward pass
            eflow_tracks_event, towers_event, eflow_neutral_hadrons_event = hcal(particles, tracks)

            eflow_tracks_batch.append(eflow_tracks_event)
            towers_batch.append(towers_event)
            eflow_neutral_hadrons_batch.append(eflow_neutral_hadrons_event)

        eflow_tracks.append(torch.cat(eflow_tracks_batch, dim=0))
        towers.append(torch.cat(towers_batch, dim=0))
        eflow_neutral_hadrons.append(torch.cat(eflow_neutral_hadrons_batch, dim=0))

    return eflow_tracks, towers, eflow_neutral_hadrons


def process_calorimeter_pipeline(
    ecal_tower_tensors: List[torch.Tensor],
    hcal_tower_tensors: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Apply Calorimeter Merger to combine ECal and HCal towers.
    
    The Calorimeter module from delphes_card_CMS_6_0.tcl is a Merger that takes:
    - InputArray: ECal/ecalTowers
    - InputArray: HCal/hcalTowers
    - OutputArray: towers (CalorimeterTower)
    
    Args:
        ecal_tower_tensors: List of ECal tower tensors (one per event)
        hcal_tower_tensors: List of HCal tower tensors (one per event)
        
    Returns:
        List of merged calorimeter tower tensors (one per event)
    """
    
    # Initialize Merger module
    calorimeter = Merger().to(DEVICE)
    
    merged_tower_tensors = []
    
    print("\nCalorimeter Merger: Combining ECal and HCal towers...")
    
    # Process event by event
    for ecal_towers, hcal_towers in tqdm(zip(ecal_tower_tensors, hcal_tower_tensors), total=len(ecal_tower_tensors)):
        # Merge ECal and HCal towers
        merged = calorimeter([ecal_towers, hcal_towers])
        merged_tower_tensors.append(merged)
    
    return merged_tower_tensors

def process_eflow_merger_pipeline(
    hcal_eflow_tracks: List[torch.Tensor],
    eflow_photons: List[torch.Tensor],
    eflow_neutral_hadrons: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Apply EFlowMerger to combine charged tracks and neutral calorimeter deposits.

    The EFlowMerger module from delphes_card_CMS_6_1.tcl merges:
    - InputArray: HCal/eflowTracks (Track objects)
    - InputArray: ECal/eflowPhotons (Tower objects)
    - InputArray: HCal/eflowNeutralHadrons (Tower objects)
    - OutputArray: eflow (ParticleFlowCandidate)

    The EFlowMerger class applies necessary transformations:
    - Tracks: Eta field set to EtaOuter (position eta) for ParticleFlow consistency
    - Photons: PID=22, X/Y/Z=0, Eem=E, Ehad=0
    - Neutral Hadrons: PID=0, X/Y/Z=0, Eem=0, Ehad=E

    Args:
        hcal_eflow_tracks: List of Track tensors from HCal (one per event)
        eflow_photons: List of Tower tensors from ECal (one per event)
        eflow_neutral_hadrons: List of Tower tensors from HCal (one per event)

    Returns:
        List of merged ParticleFlowCandidate tensors (one per event)
    """

    # Initialize EFlowMerger module
    eflow_merger = EFlowMerger().to(DEVICE)

    eflow_tensors = []

    print("\nEFlowMerger: Combining tracks and calorimeter deposits...")

    # Process event by event
    for tracks, photons, neutrals in tqdm(
        zip(hcal_eflow_tracks, eflow_photons, eflow_neutral_hadrons),
        total=len(hcal_eflow_tracks)
    ):
        # Merge all three input arrays with proper transformations
        merged = eflow_merger([tracks, photons, neutrals])
        eflow_tensors.append(merged)

    return eflow_tensors


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
    # ParticleFlowCandidate: Combined Track+Tower fields
    track_kinematic_vars = ['PID', 'Charge', 'P', 'PT', 'Eta', 'EtaOuter', 'Phi', 'T', 'X', 'Y', 'Z']
    tower_kinematic_vars = ['E', 'ET', 'Eta', 'Phi', 'T']
    eflow_kinematic_vars = ['PID', 'Charge', 'E', 'P', 'PT', 'Eta', 'Phi', 'T', 'X', 'Y', 'Z', 'Eem', 'Ehad']

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
        ('EFlowPhoton', tower_kinematic_vars),
        ('HCalTower', tower_kinematic_vars),
        ('HCal_EFlowTrack', track_kinematic_vars),
        ('EFlowNeutralHadron', tower_kinematic_vars),
        ('CalorimeterTower', tower_kinematic_vars),
        ('EFlowObject', eflow_kinematic_vars),
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
                    print(f"  Bin edges: {bin_edges}")
                    print(f"  C++ counts: {benchmark_counts}")
                    print(f"  TorchDelphes counts: {torch_counts}")
                    
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

def validate_specific_event(
    torch_output_file: str,
    benchmark_file: str,
    event_index: int,
    output_dir: str,
    debug: bool = False
) -> None:
    """
    Validate a single event by comparing PyTorch and C++ Delphes outputs.

    Args:
        torch_output_file: Path to PyTorch output ROOT file
        benchmark_file: Path to C++ Delphes benchmark ROOT file
        event_index: Event index (0-based) to validate
        output_dir: Base directory for validation plots
        debug: If True, print detailed statistics
    """
    # Create event-specific output directory
    output_dir = Path(output_dir) / f"event_{event_index}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Validating Event {event_index}")
    print(f"{'='*70}")
    print(f"PyTorch output: {torch_output_file}")
    print(f"C++ Delphes benchmark: {benchmark_file}")
    print(f"Output directory: {output_dir}")

    # Load ROOT files
    torch_root = uproot.open(torch_output_file)
    torch_tree = torch_root["Delphes"]

    benchmark_root = uproot.open(benchmark_file)
    benchmark_tree = benchmark_root["Delphes"]

    # Check event bounds
    # Use a representative branch to check number of events
    sample_branch = "ParticleBeforeProp/ParticleBeforeProp.PID"
    if sample_branch in torch_tree.keys():
        n_events_torch = len(torch_tree[sample_branch].array())
        n_events_benchmark = len(benchmark_tree[sample_branch].array())

        if event_index >= n_events_torch or event_index >= n_events_benchmark:
            raise ValueError(
                f"Event index {event_index} out of range. "
                f"PyTorch has {n_events_torch} events, "
                f"C++ has {n_events_benchmark} events."
            )

        print(f"\nEvent {event_index} / {min(n_events_torch, n_events_benchmark) - 1}")

    # Kinematic variables to compare
    track_kinematic_vars = ['PID', 'Charge', 'P', 'PT', 'Eta', 'EtaOuter', 'Phi', 'T', 'X', 'Y', 'Z']
    tower_kinematic_vars = ['E', 'ET', 'Eta', 'Phi', 'T']
    eflow_kinematic_vars = ['PID', 'Charge', 'E', 'P', 'PT', 'Eta', 'Phi', 'T', 'X', 'Y', 'Z', 'Eem', 'Ehad']

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
        ('EFlowPhoton', tower_kinematic_vars),
        ('HCalTower', tower_kinematic_vars),
        ('HCal_EFlowTrack', track_kinematic_vars),
        ('EFlowNeutralHadron', tower_kinematic_vars),
        ('CalorimeterTower', tower_kinematic_vars),
        ('EFlowObject', eflow_kinematic_vars),
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
                # Load event-wise data
                torch_data_events = torch_tree[torch_key].array()
                benchmark_data_events = benchmark_tree[benchmark_key].array()

                # Check event index bounds
                if event_index >= len(torch_data_events) or event_index >= len(benchmark_data_events):
                    print(f"  ⚠ Event {event_index} out of range for {var}, skipping...")
                    continue

                # Extract single event data
                torch_data = torch_data_events[event_index]
                benchmark_data = benchmark_data_events[event_index]

                # Convert to numpy for plotting
                torch_np = np.asarray(torch_data)
                benchmark_np = np.asarray(benchmark_data)

                # Skip if both are empty
                if len(torch_np) == 0 and len(benchmark_np) == 0:
                    print(f"  ⚠ No particles in {branch_name}.{var} for event {event_index}, skipping...")
                    continue

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
                        print(f"\n  PID Counts for {branch_name} (Event {event_index}):")
                        for pid, torch_count, bench_count in zip(unique_pids, torch_counts, benchmark_counts):
                            ratio_val = torch_count / bench_count if bench_count > 0 else np.inf
                            print(f"    PID {int(pid):6d}: PyTorch={torch_count:5d}, C++={bench_count:5d}, Ratio={ratio_val:.4f}")

                else:
                    # Standard continuous histogram (use fewer bins for single events)
                    # Determine bin range
                    all_data = np.concatenate([torch_np, benchmark_np])
                    if len(all_data) > 0:
                        bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 30)
                    else:
                        bins = 30

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
                    print(f"\n{branch_name}.{var} (Event {event_index}) bins:")
                    print(f"  Total C++ counts: {np.sum(benchmark_counts):.0f}")
                    print(f"  Total TorchDelphes counts: {np.sum(torch_counts):.0f}")
                    if np.sum(benchmark_counts) > 0:
                        print(f"  Ratio (Torch/C++): {np.sum(torch_counts) / np.sum(benchmark_counts):.4f}")

                    # Compute and print ratio statistics
                    valid_ratio = ratio[benchmark_counts > 0]
                    if len(valid_ratio) > 0:
                        print(f"  Ratio mean: {np.mean(valid_ratio):.4f}")
                        print(f"  Ratio std: {np.std(valid_ratio):.4f}")
                        print(f"  Ratio min: {np.min(valid_ratio):.4f}")
                        print(f"  Ratio max: {np.max(valid_ratio):.4f}")

                ax_hist.set_ylabel('Counts', fontsize=12)
                ax_hist.set_title(f'Event {event_index}: {branch_name}.{var}', fontsize=14, fontweight='bold')
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
                    if len(ratio) > 0:
                        ax_ratio.set_ylim([0.9*min(ratio), 1.1*max(ratio)])  # Focus on ±10% range
                else:
                    # Line plot for continuous variables
                    ax_ratio.axhline(y=1.0, color='orange', linewidth=2)
                    ax_ratio.plot(bin_centers, ratio, color='blue', markersize=4, linewidth=2)
                    if len(ratio) > 0:
                        ax_ratio.set_ylim([0.9*min(ratio), 1.1*max(ratio)])  # Focus on ±10% range

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
                # Load event-wise data
                torch_data_events = torch_tree[torch_key].array()
                benchmark_data_events = benchmark_tree[benchmark_key].array()

                # Extract single event
                torch_data = torch_data_events[event_index]
                benchmark_data = benchmark_data_events[event_index]

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
                    bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 25)
                else:
                    bins = 25

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
                if len(ratio) > 0:
                    ax_ratio.set_ylim([0.9*min(ratio), 1.1*max(ratio)])
                ax_ratio.grid(True, alpha=0.3)

            except Exception as e:
                print(f"    ✗ Error plotting {var} in combined plot: {e}")
                continue

        # Add overall title
        fig.suptitle(f'Event {event_index}: {branch_name}', fontsize=16, fontweight='bold', y=0.98)

        # Save combined figure
        combined_plot_file = branch_dir / "all.png"
        plt.savefig(combined_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Combined plot saved → {combined_plot_file.name}")

        ### 3. PID-specific combined plots (only for branches with PID field)
        torch_pid_key = f"{branch_name}/{branch_name}.PID"
        benchmark_pid_key = f"{branch_name}/{branch_name}.PID"

        if torch_pid_key in torch_tree.keys() and benchmark_pid_key in benchmark_tree.keys():

            # Load PID data for the event
            torch_pids_events = torch_tree[torch_pid_key].array()
            benchmark_pids_events = benchmark_tree[benchmark_pid_key].array()

            # Extract event data
            torch_pids = torch_pids_events[event_index]
            benchmark_pids = benchmark_pids_events[event_index]

            # Get unique PIDs in this event
            unique_pids = np.unique(np.concatenate([
                np.asarray(torch_pids),
                np.asarray(benchmark_pids)
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

                    # Load event-wise data
                    torch_data_events = torch_tree[torch_key].array()
                    benchmark_data_events = benchmark_tree[benchmark_key].array()

                    # Extract event data
                    torch_data_event = torch_data_events[event_index]
                    benchmark_data_event = benchmark_data_events[event_index]

                    # Get PID arrays for this event
                    torch_pid_event = torch_pids_events[event_index]
                    benchmark_pid_event = benchmark_pids_events[event_index]

                    # Filter by PID within the event
                    torch_data_filtered = np.asarray(torch_data_event)[np.asarray(torch_pid_event) == pid]
                    benchmark_data_filtered = np.asarray(benchmark_data_event)[np.asarray(benchmark_pid_event) == pid]

                    # Skip if no data for this PID
                    if len(torch_data_filtered) == 0 and len(benchmark_data_filtered) == 0:
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
                    all_data = np.concatenate([torch_data_filtered, benchmark_data_filtered])
                    if len(all_data) > 0:
                        bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 25)
                    else:
                        bins = 25

                    # Plot histograms
                    benchmark_counts, bin_edges, _ = ax_hist.hist(
                        benchmark_data_filtered, bins=bins, histtype='stepfilled', color='orange', alpha=0.5,
                        linewidth=2, label=f'C++ Delphes, {len(benchmark_data_filtered)} particles', density=False
                    )
                    torch_counts, _, _ = ax_hist.hist(
                        torch_data_filtered, bins=bins, histtype='step', color='blue',
                        linewidth=2, label=f'Parnassus.TorchDelphes, {len(torch_data_filtered)} particles', density=False
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
                    if len(ratio) > 0:
                        ax_ratio.set_ylim([0.9*min(ratio), 1.1*max(ratio)])
                    ax_ratio.grid(True, alpha=0.3)

                # Add overall title with PID
                fig.suptitle(f'Event {event_index}: {branch_name} (PID={pid_int})', fontsize=16, fontweight='bold', y=0.98)

                # Save PID-specific combined figure
                pid_plot_file = branch_dir / f"pid_{pid_int}.png"
                plt.savefig(pid_plot_file, dpi=150, bbox_inches='tight')
                plt.close()

        else:
            print(f"  ℹ No PID field - skipping PID-specific plots (normal for Tower objects)")

    print(f"\n{'='*70}")
    print(f"✓ Event {event_index} validation complete! Plots saved to {output_dir}")
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

    print(f"\nAfter ParticlePropagator: {len(genevent_tensors)} events, {sum(t.shape[0] for t in pap_tensors)} particles")

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
    # STEP 6: Apply ECal
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 6: Applying ECal")
    print("="*80)
    
    ecal_eflow_tracks, ecal_towers, eflow_photons = process_ecal_pipeline(
        pap_tensors, merged_tracks
    )
    
    # Add Tower, EFlowPhoton, and EFlowTrack branches to ROOT output
    branches_torch_root.update({
        'ECal_EFlowTrack': tensor_to_root_dict([i.cpu() for i in ecal_eflow_tracks], 'ECal_EFlowTrack', expected_event_nums),
        'ECalTower': tensor_to_root_dict([i.cpu() for i in ecal_towers], 'ECalTower', expected_event_nums),
        'EFlowPhoton': tensor_to_root_dict([i.cpu() for i in eflow_photons], 'EFlowPhoton', expected_event_nums),
    })
    
    print("\n✓ ECal applied")

    # ========================================================================
    # STEP 7: Apply HCal
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 7: Applying HCal")
    print("="*80)

    hcal_eflow_tracks, hcal_towers, eflow_neutral_hadrons = process_hcal_pipeline(
        pap_tensors, ecal_eflow_tracks
    )
    
    # Add Tower, EFlowPhoton, and EFlowTrack branches to ROOT output
    branches_torch_root.update({
        'HCal_EFlowTrack': tensor_to_root_dict([i.cpu() for i in hcal_eflow_tracks], 'HCal_EFlowTrack', expected_event_nums),
        'HCalTower': tensor_to_root_dict([i.cpu() for i in hcal_towers], 'HCalTower', expected_event_nums),
        'EFlowNeutralHadron': tensor_to_root_dict([i.cpu() for i in eflow_neutral_hadrons], 'EFlowNeutralHadron', expected_event_nums),
    })

    print("\n✓ HCal applied")

    # ========================================================================
    # STEP 8: Apply Calorimeter (Merger)
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 8: Applying Calorimeter (Merger)")
    print("="*80)

    merged_towers = process_calorimeter_pipeline(
        ecal_towers, hcal_towers
    )

    # Add CalorimeterTower branch to ROOT output
    branches_torch_root.update({
        'CalorimeterTower': tensor_to_root_dict(merged_towers, 'CalorimeterTower', expected_event_nums),
    })

    print("\n✓ Calorimeter (Merger) applied")

    # ========================================================================
    # STEP 9: Apply EFlowMerger
    # ========================================================================

    print("\n" + "="*80)
    print("STEP 9: Applying EFlowMerger")
    print("="*80)

    eflow_objects = process_eflow_merger_pipeline(
        hcal_eflow_tracks, eflow_photons, eflow_neutral_hadrons
    )

    # Add EFlowObject branch to ROOT output
    branches_torch_root.update({
        'EFlowObject': tensor_to_root_dict(eflow_objects, 'EFlowObject', expected_event_nums),
    })

    print("\n✓ EFlowMerger applied")

    # ========================================================================
    # STEP 10: Write final output
    # ========================================================================

    print(f"Writing {output_file}...")
    write_root_file(output_file, branches_torch_root)

    # ========================================================================
    # STEP 11: Print summary
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
    # STEP 12: Validate Against C++ Delphes (Final ROOT branches)
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
        print("  Skipping validation. To enable validation, provide HZZ4l_6_1.root")
        print("  (Generated by C++ Delphes with delphes_card_CMS_6_1.tcl)")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parnassus TorchDelphes HepMC Processing")
    parser.add_argument(
        "--input", "-i", type=str, default="delphes_data/HZZ4l/HZZ4l_0.hepmc",
        help="Input HepMC file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="delphes_data/HZZ4l/HZZ4l_6_1_torch.root",
        help="Output ROOT file"
    )
    parser.add_argument(
        "--benchmark", "-bm", type=str, default="delphes_data/HZZ4l/HZZ4l_6_1.root",
        help="Benchmark ROOT file from C++ Delphes for validation (CMS_6_1 card with EFlowMerger)"
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
    parser.add_argument(
        "--validate-event", type=int, default=None,
        help="Validate a specific event by index (0-based). If not set, validates all events aggregated."
    )
    return parser.parse_args()

if __name__ == "__main__":
    tic = time.time()
    args = parse_args()

    # Check if validating a specific event
    if args.validate_event is not None:
        # Only run single-event validation (skip full pipeline)
        script_dir = Path(__file__).parent
        validation_dir = script_dir / "torch_delphes_validation"

        if Path(args.output).exists() and Path(args.benchmark).exists():
            validate_specific_event(
                args.output,
                args.benchmark,
                args.validate_event,
                validation_dir,
                debug=args.debug
            )
        else:
            print(f"\n⚠ Error: Output or benchmark file not found!")
            print(f"  Output: {args.output}")
            print(f"  Benchmark: {args.benchmark}")
            print(f"  Please run the full pipeline first (without --validate-event) to generate output files.")
    else:
        # Run full pipeline (includes all-events validation)
        main(args.input, args.output, args.benchmark, max_events=args.max_events, batch_size=args.batch_size, debug=args.debug)

    toc = time.time()
    dur = toc - tic
    print(f"\n{'='*80}")
    print(f"Total execution time on {DEVICE}: {dur//60:.0f} minutes, {dur%60:.2f} seconds")
    print(f"{'='*80}\n")
