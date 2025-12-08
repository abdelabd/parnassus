"""
Apply PyTorch Delphes modules (ParticlePropagator, Efficiency, MomentumSmearing) to HepMC file and save outputs.

This is a redesigned version that uses pure tensor operations:
- HepMC→ Tensor conversion happens once at the beginning
- All processing happens in tensor space
- Tensor → ROOT conversion happens once per output file

Usage:
    python test_torch_delphes_v2.py [input.root] [output.root]
    
Default:
    Input:  delphes_data/HZZ4l/HZZ4l_1.root
    Outputs: delphes_data/HZZ4l/HZZ4l_*_torch.root
"""
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

from parnassus.torch_delphes import Efficiency, Merger, MomentumSmearing, ParticlePropagator
from parnassus.torch_delphes.tensor_utils import (
    hepmc_to_tensor,
    tensor_to_root_dict,
    write_root_file,
    compute_max_particles,
    pad_and_batch,
    unbatch_and_unpad,
    COLUMN_MAP as CMAP
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def process_particle_propagator(genevent_tensors, batch_size=100):
    """
    Apply ParticlePropagator to GenEvent tensors using batched processing.
    
    Args:
        genevent_tensors: Tensor of shape N_EVENT x N_PARTICLES x 15
                - Should be zero-padded such that all tensors have the same shape (number of particles)
        batch_size: Number of events to process in each batch
        
    Returns:
        genevent_tensors: Tensor of shape N_EVENT x N_PARTICLES x 16
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

    genevent_tensors_propagated = torch.zeros((n_event, n_part, n_dim + 1), dtype=genevent_tensors.dtype)
    # Collect charged_hadron, electron, muon tensors after propagation (for intermediate testing and validation)
    ch_tensors = []
    el_tensors = []
    mu_tensors = []
    # Process in batches
    for batch_start in tqdm(range(0, n_event, batch_size)):
        batch_end = min(batch_start + batch_size, n_event)

        # Flatten to (B*N, 15)
        batch_events = genevent_tensors[batch_start:batch_end].to(DEVICE)
        batch_size = batch_events.shape[0]
        
        # Propagate particles (batched)
        particles = batch_events.reshape(-1, n_dim) # Flatten to (B*N, 15)
        particles_propagated = propagator(particles)
        n_dim_new = particles_propagated.shape[1]

        genevent_tensors_propagated[batch_start:batch_end] = particles_propagated.reshape(batch_size, n_part, n_dim_new).cpu()

        mask = particles_propagated[:, CMAP["IS_NOT_PAD"]] * particles_propagated[:, CMAP["PASS_PROP"]]
        charged_hadron_pid_mask = mask * Efficiency()._charged_hadron_pdg_filter(particles_propagated).float()
        electron_pid_mask = mask * Efficiency()._electron_pdg_filter(particles_propagated).float()
        muon_pid_mask = mask * Efficiency()._muon_pdg_filter(particles_propagated).float()

        ch_tensors.append(particles[charged_hadron_pid_mask > 0.5].cpu())
        el_tensors.append(particles[electron_pid_mask > 0.5].cpu())
        mu_tensors.append(particles[muon_pid_mask > 0.5].cpu())

    return genevent_tensors_propagated, ch_tensors, el_tensors, mu_tensors

def process_efficiency_pipeline(genevent_tensors, batch_size=100):
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
        n_dim_new = particles.shape[1]

        genevent_tensors_eff.append(particles.reshape(batch_size, n_part, n_dim_new).cpu())

        mask = particles[:, CMAP["IS_NOT_PAD"]] * particles[:, CMAP["PASS_PROP"]] * particles[:, CMAP["PASS_EFF"]]
        charged_hadron_pid_mask = mask * Efficiency()._charged_hadron_pdg_filter(particles).float()
        electron_pid_mask = mask * Efficiency()._electron_pdg_filter(particles).float()
        muon_pid_mask = mask * Efficiency()._muon_pdg_filter(particles).float()

        ch_tensors_eff.append(particles[charged_hadron_pid_mask > 0.5].cpu())
        el_tensors_eff.append(particles[electron_pid_mask > 0.5].cpu())
        mu_tensors_eff.append(particles[muon_pid_mask > 0.5].cpu())

    # Stack all event tensors into a single tensor
    genevent_tensors_eff = torch.cat(genevent_tensors_eff, dim=0)

    return genevent_tensors_eff, ch_tensors_eff, el_tensors_eff, mu_tensors_eff

def process_smearing_pipeline(genevent_tensors, batch_size=100):
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

        ch_tensors_smeared.append(particles[charged_hadron_pid_mask > 0.5].cpu())
        el_tensors_smeared.append(particles[electron_pid_mask > 0.5].cpu())
        mu_tensors_smeared.append(particles[muon_pid_mask > 0.5].cpu())

    # Stack all event tensors into a single tensor
    genevent_tensors_smeared = torch.cat(genevent_tensors_smeared, dim=0)

    return genevent_tensors_smeared, ch_tensors_smeared, el_tensors_smeared, mu_tensors_smeared

def process_merger_pipeline(genevent_tensors, batch_size=100):
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
        n_dim_new = batch_merged.shape[-1]
        
        genevent_tensors_merged.append(batch_merged.cpu())
        
        # Extract valid tracks for this batch (for validation)
        # Flatten batch to (B*N, D)
        particles = batch_merged.reshape(-1, n_dim_new)
        
        # Mask for particles that passed merger
        track_mask = (
            particles[:, CMAP["IS_NOT_PAD"]] *
            particles[:, CMAP["PASS_PROP"]] *
            particles[:, CMAP["PASS_EFF"]] *
            particles[:, CMAP["PASS_MERGER"]]
        )
        
        track_tensors.append(particles[track_mask > 0.5].cpu())
    
    # Stack all event tensors into a single tensor
    genevent_tensors_merged = torch.cat(genevent_tensors_merged, dim=0)
    
    return genevent_tensors_merged, track_tensors

def validate_against_benchmark(torch_output_file, benchmark_file, output_dir):
    """
    Validate PyTorch Delphes implementation against C++ Delphes benchmark.
    
    Args:
        torch_output_file: Path to PyTorch output ROOT file (e.g., HZZ4l_3_2_torch.root)
        benchmark_file: Path to benchmark ROOT file from C++ Delphes
        output_dir: Directory to save validation plots
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading PyTorch output: {torch_output_file}")
    torch_root = uproot.open(torch_output_file)
    torch_tree = torch_root["Delphes"]
    print(f"torch_tree.keys(): {torch_tree.keys()}")
    
    print(f"Loading C++ Delphes benchmark: {benchmark_file}")
    benchmark_root = uproot.open(benchmark_file)
    benchmark_tree = benchmark_root["Delphes"]
    print(f"benchmark_+tree.keys(): {benchmark_tree.keys()}")
    
    # Kinematic variables to compare (Track objects have these attributes)
    kinematic_vars = ['Charge', 'P', 'PT', 'Eta', 'Phi']
    
    # Branches to validate
    branches = [
        'ChargedHadron', 'Electron', 'Muon',
        'ChargedHadronEfficiency', 'ElectronEfficiency', 'MuonEfficiency',
        'ChargedHadronSmeared', 'ElectronSmeared', 'MuonSmeared',
        'MergedTracks'
    ]
    
    print(f"\nValidating branches: {', '.join(branches)}")
    print(f"Kinematic variables: {', '.join(kinematic_vars)}")
    
    for branch_name in branches:
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
                
                # Determine bin range
                all_data = np.concatenate([torch_np, benchmark_np])
                if len(all_data) > 0:
                    bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 50)
                else:
                    bins = 50
                
                # Plot histograms
                benchmark_counts, bin_edges, _ = ax_hist.hist(
                    benchmark_np, bins=bins, histtype='step', color='orange', 
                    linewidth=2, label='C++ Delphes', density=False
                )
                torch_counts, _, _ = ax_hist.hist(
                    torch_np, bins=bins, histtype='step', color='blue', 
                    linewidth=2, label='Parnassus.TorchDelphes', density=False
                )
                
                ax_hist.set_ylabel('Counts', fontsize=12)
                ax_hist.set_title(f'{branch_name}: {var}', fontsize=14, fontweight='bold')
                ax_hist.legend(fontsize=11)
                ax_hist.grid(True, alpha=0.3)
                ax_hist.tick_params(labelbottom=False)  # Hide x-axis labels for top plot
                
                # Add statistics text
                stats_text = f'PyTorch: {len(torch_np)} particles\nC++ Delphes: {len(benchmark_np)} particles'
                ax_hist.text(0.95, 0.95, stats_text, transform=ax_hist.transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Plot ratio: TorchDelphes / C++ Delphes
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ratio = np.divide(
                    torch_counts, benchmark_counts, 
                    out=np.ones_like(torch_counts), 
                    where=benchmark_counts > 0
                )
                
                ax_ratio.axhline(y=1.0, color='orange', linewidth=2)
                ax_ratio.plot(bin_centers, ratio, color='blue', markersize=4, linewidth=2)
                ax_ratio.set_xlabel(var, fontsize=12)
                ax_ratio.set_ylabel('Torch / C++', fontsize=10)
                ax_ratio.set_ylim([0.9*min(ratio), 1.1*max(ratio)])  # Focus on ±20% range
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
        
        # Create combined plot with all 4 kinematic variables (Eta, Phi, PT, P)
        print(f"\n  Creating combined kinematic plot (Eta, Phi, PT, P)...")
        combined_vars = ['Eta', 'Phi', 'PT', 'P']
        
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
                    benchmark_np, bins=bins, histtype='step', color='orange', 
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
    
    print(f"\n{'='*70}")
    print(f"✓ Validation complete! Plots saved to {output_dir}")
    print(f"{'='*70}")


def main(input_file, output_file, benchmark_file, max_events=None, batch_size=100):
    """Main processing function.
    
    Args:
        input_file: Path to input HepMC file
        output_file: Path to output ROOT file
        benchmark_file: Path to benchmark ROOT file
        max_events: Maximum number of events to process (None = all)
        batch_size: Number of events to process per batch (for GPU acceleration)
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

    genevent_tensors, ch_tensors, el_tensors, mu_tensors = process_particle_propagator(genevent_tensors, batch_size=batch_size)

    print(f"\nAfter ParticlePropagator: {len(genevent_tensors)} events")

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
    
    toc_torch = time.time()
    dur_torch = toc_torch - tic_torch
    print(f"\n\nTorch duration: {dur_torch//60:.0f} minutes, {dur_torch%60:.2f} seconds\n\n")

    # ========================================================================
    # STEP 6: Write final output
    # ========================================================================

    print(f"Writing {output_file}...")
    branches_v3_2 = {
        'ChargedHadron': tensor_to_root_dict(ch_tensors, 'ChargedHadron'),
        'Electron': tensor_to_root_dict(el_tensors, 'Electron'),
        'Muon': tensor_to_root_dict(mu_tensors, 'Muon'),
        'ChargedHadronEfficiency': tensor_to_root_dict(ch_filtered, 'ChargedHadronEfficiency'),
        'ElectronEfficiency': tensor_to_root_dict(el_filtered, 'ElectronEfficiency'),
        'MuonEfficiency': tensor_to_root_dict(mu_filtered, 'MuonEfficiency'),
        'ChargedHadronSmeared': tensor_to_root_dict(ch_smeared, 'ChargedHadronSmeared'),
        'ElectronSmeared': tensor_to_root_dict(el_smeared, 'ElectronSmeared'),
        'MuonSmeared': tensor_to_root_dict(mu_smeared, 'MuonSmeared'),
        'MergedTracks': tensor_to_root_dict(track_merged, 'MergedTracks')
    }
    write_root_file(output_file, branches_v3_2)

    # ========================================================================
    # STEP 7: Print summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_ch_input = sum(t.shape[0] for t in ch_tensors)
    total_el_input = sum(t.shape[0] for t in el_tensors)
    total_mu_input = sum(t.shape[0] for t in mu_tensors)
    
    # total_ch_filtered = sum(t.shape[0] for t in ch_filtered)
    # total_el_filtered = sum(t.shape[0] for t in el_filtered)
    # total_mu_filtered = sum(t.shape[0] for t in mu_filtered)
    
    print(f"\nChargedHadrons:")
    print(f"  Input:      {total_ch_input}")
    # print(f"  After eff:  {total_ch_filtered} ({100*total_ch_filtered/total_ch_input:.1f}%)")
    # print(f"  Smeared:    {sum(t.shape[0] for t in ch_smeared)}")
    
    print(f"\nElectrons:")
    print(f"  Input:      {total_el_input}")
    # print(f"  After eff:  {total_el_filtered} ({100*total_el_filtered/total_el_input:.1f}%)")
    # print(f"  Smeared:    {sum(t.shape[0] for t in el_smeared)}")
    
    print(f"\nMuons:")
    print(f"  Input:      {total_mu_input}")
    # print(f"  After eff:  {total_mu_filtered} ({100*total_mu_filtered/total_mu_input:.1f}%)")
    # print(f"  Smeared:    {sum(t.shape[0] for t in mu_smeared)}")
    
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
        validate_against_benchmark(output_file, benchmark_file, validation_dir)
    else:
        print(f"\n⚠ Benchmark file not found: {benchmark_file}")
        print("  Skipping validation. To enable validation, provide HZZ4l_3_2.root")
        print("  (Generated by C++ Delphes with delphes_card_CMS_3_2.tcl)")

def parse_args():
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
        help="Benchmark ROOT file from C++ Delphes for validation"
    )
    parser.add_argument(
        "--max-events", "-n", type=int, default=1000,
        help="Maximum number of events to process (default: 1000)"
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=100,
        help="Batch size for processing (default: 1000)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    tic = time.time()
    args = parse_args()

    main(args.input, args.output, args.benchmark, max_events=args.max_events, batch_size=args.batch_size)

    toc = time.time()
    dur = toc - tic
    print(f"\n{'='*80}")
    print(f"Total execution time on {DEVICE}: {dur//60:.0f} minutes, {dur%60:.2f} seconds")
    print(f"{'='*80}\n")
