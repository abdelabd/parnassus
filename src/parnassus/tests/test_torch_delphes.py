"""
Apply PyTorch Delphes modules (ParticlePropagator, Efficiency, MomentumSmearing) to HepMC file and save outputs.

This is a redesigned version that uses pure tensor operations:
- HepMC→ Tensor conversion happens once at the beginning
- All processing happens in tensor space
- Tensor → ROOT conversion happens once per output file

This script emulates delphes_card_CMS_3_2.tcl with six output files:
1. v2_0: ChargedHadronEfficiency only
2. v2_1: ChargedHadronEfficiency + ElectronEfficiency
3. v2_2: ChargedHadronEfficiency + ElectronEfficiency + MuonEfficiency
4. v3_0: All efficiency modules + ChargedHadronMomentumSmearing
5. v3_1: All efficiency modules + ChargedHadronSmearing + ElectronSmearing
6. v3_2: All efficiency modules + all momentum smearing

Usage:
    python test_torch_delphes_v2.py [input.root] [output_v2_0.root] ... [output_v3_2.root]
    
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

from parnassus.torch_delphes import Efficiency, MomentumSmearing, ParticlePropagator
from parnassus.torch_delphes.tensor_utils import (
    hepmc_to_tensor,
    tensor_to_root_dict,
    write_root_file,
    compute_max_particles,
    pad_and_batch,
    unbatch_and_unpad
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def process_particle_propagator(genparticle_tensors, batch_size=100):
    """
    Apply ParticlePropagator to GenParticle tensors using batched processing.
    
    Args:
        genparticle_tensors: List of tensors (one per event), each (N, 15)
        batch_size: Number of events to process in each batch
        
    Returns:
        Tuple of (ch_tensors, el_tensors, mu_tensors) after propagation
        Each is a list of tensors (one per event)
    """
    n_events = len(genparticle_tensors)
    
    # Initialize ParticlePropagator module
    propagator = ParticlePropagator(
        radius=1.29,        # CMS tracker radius in meters
        half_length=3.0,    # CMS tracker half-length in meters  
        bz=3.8,             # Magnetic field in Tesla
        device=DEVICE
    )
    
    print(f"\nParticlePropagator (batch_size={batch_size})...")
    
    ch_tensors = []
    el_tensors = []
    mu_tensors = []
    
    # Process in batches
    for batch_start in tqdm(range(0, n_events, batch_size)):
        batch_end = min(batch_start + batch_size, n_events)
        batch_events = genparticle_tensors[batch_start:batch_end]
        
        # Compute max particles for this batch
        max_particles = compute_max_particles(batch_events, scale=1.2)
        
        # Pad and batch
        batched = pad_and_batch(batch_events, max_particles).to(DEVICE)
        
        # Propagate particles (batched)
        outputs = propagator(batched)
        
        # Unbatch each particle type
        ch_batch = unbatch_and_unpad(outputs['ChargedHadron'].cpu(), mask_col=15)
        el_batch = unbatch_and_unpad(outputs['Electron'].cpu(), mask_col=15)
        mu_batch = unbatch_and_unpad(outputs['Muon'].cpu(), mask_col=15)
        
        # Collect results
        ch_tensors.extend(ch_batch)
        el_tensors.extend(el_batch)
        mu_tensors.extend(mu_batch)
    
    return ch_tensors, el_tensors, mu_tensors

def process_efficiency_pipeline(charged_hadron_tensors, electron_tensors, muon_tensors, batch_size=100):
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
    n_events = len(charged_hadron_tensors)
    
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
    
    # Apply efficiency to charged hadrons
    print(f"\nChargedHadronTrackingEfficiency (batch_size={batch_size})...")
    ch_filtered = []
    for batch_start in tqdm(range(0, n_events, batch_size)):
        batch_end = min(batch_start + batch_size, n_events)
        batch_events = charged_hadron_tensors[batch_start:batch_end]
        
        max_particles = compute_max_particles(batch_events, scale=1.2)
        batched = pad_and_batch(batch_events, max_particles).to(DEVICE)
        filtered_batched = ch_eff_module(batched)
        filtered_batch = unbatch_and_unpad(filtered_batched.cpu(), mask_col=15)
        ch_filtered.extend(filtered_batch)
    
    # Apply efficiency to electrons
    print(f"\nElectronTrackingEfficiency (batch_size={batch_size})...")
    el_filtered = []
    for batch_start in tqdm(range(0, n_events, batch_size)):
        batch_end = min(batch_start + batch_size, n_events)
        batch_events = electron_tensors[batch_start:batch_end]
        
        max_particles = compute_max_particles(batch_events, scale=1.2)
        batched = pad_and_batch(batch_events, max_particles).to(DEVICE)
        filtered_batched = el_eff_module(batched)
        filtered_batch = unbatch_and_unpad(filtered_batched.cpu(), mask_col=15)
        el_filtered.extend(filtered_batch)
    
    # Apply efficiency to muons
    print(f"\nMuonTrackingEfficiency (batch_size={batch_size})...")
    mu_filtered = []
    for batch_start in tqdm(range(0, n_events, batch_size)):
        batch_end = min(batch_start + batch_size, n_events)
        batch_events = muon_tensors[batch_start:batch_end]
        
        max_particles = compute_max_particles(batch_events, scale=1.2)
        batched = pad_and_batch(batch_events, max_particles).to(DEVICE)
        filtered_batched = mu_eff_module(batched)
        filtered_batch = unbatch_and_unpad(filtered_batched.cpu(), mask_col=15)
        mu_filtered.extend(filtered_batch)
    
    return ch_filtered, el_filtered, mu_filtered

def process_smearing_pipeline(ch_filtered, el_filtered, mu_filtered, batch_size=100):
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
    n_events = len(ch_filtered)
    
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
    
    # Apply smearing to charged hadrons
    print(f"\nChargedHadronMomentumSmearing (batch_size={batch_size})...")
    ch_smeared = []
    for batch_start in tqdm(range(0, n_events, batch_size)):
        batch_end = min(batch_start + batch_size, n_events)
        batch_events = ch_filtered[batch_start:batch_end]
        
        max_particles = compute_max_particles(batch_events, scale=1.2)
        batched = pad_and_batch(batch_events, max_particles).to(DEVICE)
        smeared_batched = ch_smear_module(batched)
        smeared_batch = unbatch_and_unpad(smeared_batched.cpu(), mask_col=15)
        ch_smeared.extend(smeared_batch)
    
    # Apply smearing to electrons
    print(f"\nElectronMomentumSmearing (batch_size={batch_size})...")
    el_smeared = []
    for batch_start in tqdm(range(0, n_events, batch_size)):
        batch_end = min(batch_start + batch_size, n_events)
        batch_events = el_filtered[batch_start:batch_end]
        
        max_particles = compute_max_particles(batch_events, scale=1.2)
        batched = pad_and_batch(batch_events, max_particles).to(DEVICE)
        smeared_batched = el_smear_module(batched)
        smeared_batch = unbatch_and_unpad(smeared_batched.cpu(), mask_col=15)
        el_smeared.extend(smeared_batch)
    
    # Apply smearing to muons
    print(f"\nMuonMomentumSmearing (batch_size={batch_size})...")
    mu_smeared = []
    for batch_start in tqdm(range(0, n_events, batch_size)):
        batch_end = min(batch_start + batch_size, n_events)
        batch_events = mu_filtered[batch_start:batch_end]
        
        max_particles = compute_max_particles(batch_events, scale=1.2)
        batched = pad_and_batch(batch_events, max_particles).to(DEVICE)
        smeared_batched = mu_smear_module(batched)
        smeared_batch = unbatch_and_unpad(smeared_batched.cpu(), mask_col=15)
        mu_smeared.extend(smeared_batch)
    
    return ch_smeared, el_smeared, mu_smeared

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
    
    print(f"Loading C++ Delphes benchmark: {benchmark_file}")
    benchmark_root = uproot.open(benchmark_file)
    benchmark_tree = benchmark_root["Delphes"]
    
    # Kinematic variables to compare (Track objects have these attributes)
    kinematic_vars = ['Charge', 'P', 'PT', 'Eta', 'Phi']
    
    # Branches to validate
    branches = [
        'ChargedHadron', 'Electron', 'Muon',
        'ChargedHadronEfficiency', 'ElectronEfficiency', 'MuonEfficiency',
        'ChargedHadronSmeared', 'ElectronSmeared', 'MuonSmeared'
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
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Determine bin range
                all_data = np.concatenate([torch_np, benchmark_np])
                if len(all_data) > 0:
                    bins = np.linspace(np.percentile(all_data, 1), np.percentile(all_data, 99), 50)
                else:
                    bins = 50
                
                # Plot histograms
                ax.hist(benchmark_np, bins=bins, histtype='step', color='orange', 
                       linewidth=2, label='C++ Delphes', density=False)
                ax.hist(torch_np, bins=bins, histtype='step', color='blue', 
                       linewidth=2, label='Parnassus.TorchDelphes', density=False)
                
                ax.set_xlabel(var, fontsize=12)
                ax.set_ylabel('Counts', fontsize=12)
                ax.set_title(f'{branch_name}: {var}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f'PyTorch: {len(torch_np)} particles\nC++ Delphes: {len(benchmark_np)} particles'
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Save plot
                plot_file = branch_dir / f"{var}.png"
                plt.tight_layout()
                plt.savefig(plot_file, dpi=150)
                plt.close()
                
                print(f"  ✓ {var}: PyTorch={len(torch_np)}, C++={len(benchmark_np)} → {plot_file.name}")
                
            except Exception as e:
                print(f"  ✗ {var}: Error - {e}")
                continue
    
    print(f"\n{'='*70}")
    print(f"✓ Validation complete! Plots saved to {output_dir}")
    print(f"{'='*70}")


def main(input_file, output_file, max_events=None, batch_size=100):
    """Main processing function.
    
    Args:
        input_file: Path to input HepMC file
        output_file: Path to output ROOT file
        max_events: Maximum number of events to process (None = all)
        batch_size: Number of events to process per batch (for GPU acceleration)
    """
    
    print("\n" + "="*80)
    print("Parnassus.TorchDelphes Processing (BATCHED)")
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
    
    genparticle_tensors = hepmc_to_tensor(input_file, max_events)
    n_events = len(genparticle_tensors)
    print(f"Loaded {n_events} events from HepMC")
    print(f"  Total stable particles: {sum(t.shape[0] for t in genparticle_tensors)}")
    

    # ========================================================================
    # STEP 2: Apply ParticlePropagator
    # ========================================================================

    tic_torch = time.time()

    print("\n" + "="*80)
    print("STEP 2: Applying ParticlePropagator (batched)")
    print("="*80)
    
    ch_tensors, el_tensors, mu_tensors = process_particle_propagator(genparticle_tensors, batch_size=batch_size)
    
    print(f"\nAfter ParticlePropagator: {len(ch_tensors)} events")
    print(f"  ChargedHadrons: {sum(t.shape[0] for t in ch_tensors)} total particles")
    print(f"  Electrons: {sum(t.shape[0] for t in el_tensors)} total particles")
    print(f"  Muons: {sum(t.shape[0] for t in mu_tensors)} total particles")

    # ========================================================================
    # STEP 3: Apply tracking efficiency
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: Applying Efficiency modules (batched)")
    print("="*80)

    ch_filtered, el_filtered, mu_filtered = process_efficiency_pipeline(
        ch_tensors, el_tensors, mu_tensors, batch_size=batch_size
    )

    print("\n✓ Efficiency applied")

    # ========================================================================
    # STEP 4: Apply momentum smearing
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: Applying MomentumSmearing modules (batched)")
    print("="*80)
    
    ch_smeared, el_smeared, mu_smeared = process_smearing_pipeline(
        ch_filtered, el_filtered, mu_filtered, batch_size=batch_size
    )
    
    print("\n✓ MomentumSmearing applied")
    
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
        'MuonSmeared': tensor_to_root_dict(mu_smeared, 'MuonSmeared')
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
    
    total_ch_filtered = sum(t.shape[0] for t in ch_filtered)
    total_el_filtered = sum(t.shape[0] for t in el_filtered)
    total_mu_filtered = sum(t.shape[0] for t in mu_filtered)
    
    print(f"\nChargedHadrons:")
    print(f"  Input:      {total_ch_input}")
    print(f"  After eff:  {total_ch_filtered} ({100*total_ch_filtered/total_ch_input:.1f}%)")
    print(f"  Smeared:    {sum(t.shape[0] for t in ch_smeared)}")
    
    print(f"\nElectrons:")
    print(f"  Input:      {total_el_input}")
    print(f"  After eff:  {total_el_filtered} ({100*total_el_filtered/total_el_input:.1f}%)")
    print(f"  Smeared:    {sum(t.shape[0] for t in el_smeared)}")
    
    print(f"\nMuons:")
    print(f"  Input:      {total_mu_input}")
    print(f"  After eff:  {total_mu_filtered} ({100*total_mu_filtered/total_mu_input:.1f}%)")
    print(f"  Smeared:    {sum(t.shape[0] for t in mu_smeared)}")
    
    print("\n" + "="*80)
    print("✓ ALL PROCESSING COMPLETE!")
    print("="*80 + "\n")
    
    # ========================================================================
    # STEP 8: Validate Against C++ Delphes
    # ========================================================================
    
    # Determine benchmark file location
    script_dir = Path(__file__).parent
    benchmark_file = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_3_2.root"
    validation_dir = script_dir / "torch_delphes_validation"
    
    if benchmark_file.exists():
        print(f"\nBenchmark file: {benchmark_file}")
        print(f"Validation directory: {validation_dir}")
        validate_against_benchmark(output_file, str(benchmark_file), validation_dir)
    else:
        print(f"\n⚠ Benchmark file not found: {benchmark_file}")
        print("  Skipping validation. To enable validation, provide HZZ4l_3_2.root")
        print("  (Generated by C++ Delphes with delphes_card_CMS_3_2.tcl)")


if __name__ == "__main__":
    tic = time.time()
    
    input_file = "delphes_data/HZZ4l/HZZ4l_0.hepmc"
    output_file = "delphes_data/HZZ4l/HZZ4l_3_2_torch.root"
    
    # Batch size: Larger = better GPU utilization, but more memory
    # Typical values: 100-1000 depending on available GPU memory
    batch_size = 1000
    
    main(input_file, output_file, max_events=1000, batch_size=batch_size)
    
    toc = time.time()
    dur = toc - tic
    print(f"\n{'='*80}")
    print(f"Total execution time on {DEVICE}: {dur//60:.0f} minutes, {dur%60:.2f} seconds")
    print(f"{'='*80}\n")
