"""
Apply PyTorch Delphes Efficiency and MomentumSmearing modules to ROOT file and save outputs.

This is a redesigned version that uses pure tensor operations:
- ROOT → Tensor conversion happens once at the beginning
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

from parnassus.torch_delphes import Efficiency, MomentumSmearing
from parnassus.torch_delphes.tensor_utils import (
    load_all_particle_types,
    tensor_to_root_dict,
    write_root_file
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def process_efficiency_pipeline(charged_hadron_tensors, electron_tensors, muon_tensors):
    """
    Apply tracking efficiency to all three particle types.
    
    Args:
        charged_hadron_tensors: List of tensors (one per event)
        electron_tensors: List of tensors (one per event)
        muon_tensors: List of tensors (one per event)
        
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
    
    # Apply efficiency to each event
    print("\nApplying ChargedHadronTrackingEfficiency...")
    ch_filtered = []
    for i in tqdm(range(n_events)):
        filtered = ch_eff_module(charged_hadron_tensors[i])
        ch_filtered.append(filtered)
    
    print("\nApplying ElectronTrackingEfficiency...")
    el_filtered = []
    for i in tqdm(range(n_events)):
        filtered = el_eff_module(electron_tensors[i])
        el_filtered.append(filtered)
    
    print("\nApplying MuonTrackingEfficiency...")
    mu_filtered = []
    for i in tqdm(range(n_events)):
        filtered = mu_eff_module(muon_tensors[i])
        mu_filtered.append(filtered)
    
    return ch_filtered, el_filtered, mu_filtered


def process_smearing_pipeline(ch_filtered, el_filtered, mu_filtered):
    """
    Apply momentum smearing to all three particle types.
    
    Args:
        ch_filtered: List of filtered charged hadron tensors
        el_filtered: List of filtered electron tensors
        mu_filtered: List of filtered muon tensors
        
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
    
    # Apply smearing to each event
    print("\nApplying ChargedHadronMomentumSmearing...")
    ch_smeared = []
    for i in tqdm(range(n_events)):
        smeared = ch_smear_module(ch_filtered[i])
        ch_smeared.append(smeared)
    
    print("\nApplying ElectronMomentumSmearing...")
    el_smeared = []
    for i in tqdm(range(n_events)):
        smeared = el_smear_module(el_filtered[i])
        el_smeared.append(smeared)
    
    print("\nApplying MuonMomentumSmearing...")
    mu_smeared = []
    for i in tqdm(range(n_events)):
        smeared = mu_smear_module(mu_filtered[i])
        mu_smeared.append(smeared)
    
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
    
    # Branches to validate (efficiency modules + momentum smearing)
    branches = ['ChargedHadronEfficiency', 'ElectronEfficiency', 'MuonEfficiency', 
                'ChargedHadronSmeared', 'ElectronSmeared', 'MuonSmeared']
    
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
                       linewidth=2, label='C++ Delphes', density=True)
                ax.hist(torch_np, bins=bins, histtype='step', color='blue', 
                       linewidth=2, label='Parnassus.TorchDelphes', density=True)
                
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


def main(input_file, output_file_v2_0, output_file_v2_1, output_file_v2_2, 
         output_file_v3_0, output_file_v3_1, output_file_v3_2, max_events=None):
    """Main processing function."""
    
    print("\n" + "="*80)
    print("PyTorch Delphes Processing (Redesigned Tensor-Based)")
    print("="*80)
    print(f"\nInput:  {input_file}")
    
    # ==================== STEP 1: ROOT → Tensor ====================
    print("\n" + "="*80)
    print("STEP 1: Loading ROOT data and converting to tensors")
    print("="*80)
    
    input_root = uproot.open(input_file)
    tree = input_root["Delphes"]
    n_events_total = tree.num_entries
    n_events = n_events_total if max_events is None else min(max_events, n_events_total)
    
    print(f"Total events in file: {n_events_total}")
    print(f"Processing: {n_events} events")
    
    # Load all three particle types at once
    print("\nLoading ChargedHadron, Electron, and Muon branches...")
    ch_tensors, el_tensors, mu_tensors = load_all_particle_types(tree, max_events)
    
    print(f"✓ Loaded {len(ch_tensors)} events")
    print(f"  - ChargedHadrons: {sum(t.shape[0] for t in ch_tensors)} total particles")
    print(f"  - Electrons: {sum(t.shape[0] for t in el_tensors)} total particles")
    print(f"  - Muons: {sum(t.shape[0] for t in mu_tensors)} total particles")
    
    # ==================== STEP 2: Apply Efficiency Modules ====================
    print("\n" + "="*80)
    print("STEP 2: Applying tracking efficiency modules")
    print("="*80)
    
    ch_filtered, el_filtered, mu_filtered = process_efficiency_pipeline(
        ch_tensors, el_tensors, mu_tensors
    )
    
    print("\n✓ Tracking efficiency applied")
    print(f"  - ChargedHadrons: {sum(t.shape[0] for t in ch_filtered)} survived")
    print(f"  - Electrons: {sum(t.shape[0] for t in el_filtered)} survived")
    print(f"  - Muons: {sum(t.shape[0] for t in mu_filtered)} survived")
    
    # ==================== STEP 3: Apply Momentum Smearing ====================
    print("\n" + "="*80)
    print("STEP 3: Applying momentum smearing modules")
    print("="*80)
    
    ch_smeared, el_smeared, mu_smeared = process_smearing_pipeline(
        ch_filtered, el_filtered, mu_filtered
    )
    
    print("\n✓ Momentum smearing applied")
    
    # ==================== STEP 4: Tensor → ROOT (Write Outputs) ====================
    print("\n" + "="*80)
    print("STEP 4: Converting tensors to ROOT and writing output files")
    print("="*80)
    
    # Output v2_0: ChargedHadronEfficiency only
    print(f"\nWriting {output_file_v2_0}...")
    branches_v2_0 = {
        "ChargedHadronEfficiency": tensor_to_root_dict(ch_filtered, "ChargedHadronEfficiency")
    }
    write_root_file(output_file_v2_0, branches_v2_0)
    print("✓ Done")
    
    # Output v2_1: ChargedHadronEfficiency + ElectronEfficiency
    print(f"\nWriting {output_file_v2_1}...")
    branches_v2_1 = {
        "ChargedHadronEfficiency": tensor_to_root_dict(ch_filtered, "ChargedHadronEfficiency"),
        "ElectronEfficiency": tensor_to_root_dict(el_filtered, "ElectronEfficiency")
    }
    write_root_file(output_file_v2_1, branches_v2_1)
    print("✓ Done")
    
    # Output v2_2: All three efficiency outputs
    print(f"\nWriting {output_file_v2_2}...")
    branches_v2_2 = {
        "ChargedHadronEfficiency": tensor_to_root_dict(ch_filtered, "ChargedHadronEfficiency"),
        "ElectronEfficiency": tensor_to_root_dict(el_filtered, "ElectronEfficiency"),
        "MuonEfficiency": tensor_to_root_dict(mu_filtered, "MuonEfficiency")
    }
    write_root_file(output_file_v2_2, branches_v2_2)
    print("✓ Done")
    
    # Output v3_0: All efficiency + ChargedHadronSmeared
    print(f"\nWriting {output_file_v3_0}...")
    branches_v3_0 = {
        "ChargedHadronEfficiency": tensor_to_root_dict(ch_filtered, "ChargedHadronEfficiency"),
        "ElectronEfficiency": tensor_to_root_dict(el_filtered, "ElectronEfficiency"),
        "MuonEfficiency": tensor_to_root_dict(mu_filtered, "MuonEfficiency"),
        "ChargedHadronSmeared": tensor_to_root_dict(ch_smeared, "ChargedHadronSmeared")
    }
    write_root_file(output_file_v3_0, branches_v3_0)
    print("✓ Done")
    
    # Output v3_1: All efficiency + ChargedHadron + Electron smearing
    print(f"\nWriting {output_file_v3_1}...")
    branches_v3_1 = {
        "ChargedHadronEfficiency": tensor_to_root_dict(ch_filtered, "ChargedHadronEfficiency"),
        "ElectronEfficiency": tensor_to_root_dict(el_filtered, "ElectronEfficiency"),
        "MuonEfficiency": tensor_to_root_dict(mu_filtered, "MuonEfficiency"),
        "ChargedHadronSmeared": tensor_to_root_dict(ch_smeared, "ChargedHadronSmeared"),
        "ElectronSmeared": tensor_to_root_dict(el_smeared, "ElectronSmeared")
    }
    write_root_file(output_file_v3_1, branches_v3_1)
    print("✓ Done")
    
    # Output v3_2: All efficiency + all smearing
    print(f"\nWriting {output_file_v3_2}...")
    branches_v3_2 = {
        "ChargedHadronEfficiency": tensor_to_root_dict(ch_filtered, "ChargedHadronEfficiency"),
        "ElectronEfficiency": tensor_to_root_dict(el_filtered, "ElectronEfficiency"),
        "MuonEfficiency": tensor_to_root_dict(mu_filtered, "MuonEfficiency"),
        "ChargedHadronSmeared": tensor_to_root_dict(ch_smeared, "ChargedHadronSmeared"),
        "ElectronSmeared": tensor_to_root_dict(el_smeared, "ElectronSmeared"),
        "MuonSmeared": tensor_to_root_dict(mu_smeared, "MuonSmeared")
    }
    write_root_file(output_file_v3_2, branches_v3_2)
    print("✓ Done")
    
    # ==================== Summary ====================
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
    
    # ==================== STEP 5: Validation Against C++ Delphes ====================
    print("\n" + "="*80)
    print("STEP 5: Validation against C++ Delphes benchmark")
    print("="*80)
    
    # Determine benchmark file location
    script_dir = Path(__file__).parent
    benchmark_file = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_3_2.root"
    validation_dir = script_dir / "torch_delphes_validation"
    
    if benchmark_file.exists():
        print(f"\nBenchmark file: {benchmark_file}")
        print(f"Validation directory: {validation_dir}")
        validate_against_benchmark(output_file_v3_2, str(benchmark_file), validation_dir)
    else:
        print(f"\n⚠ Benchmark file not found: {benchmark_file}")
        print("  Skipping validation. To enable validation, provide HZZ4l_3_2.root")
        print("  (Generated by C++ Delphes with delphes_card_CMS_3_2.tcl)")


if __name__ == "__main__":
    tic = time.time()
    
    # Default file paths
    default_input = "delphes_data/HZZ4l/HZZ4l_1.root"
    default_outputs = [
        "delphes_data/HZZ4l/HZZ4l_2_0_torch.root",
        "delphes_data/HZZ4l/HZZ4l_2_1_torch.root",
        "delphes_data/HZZ4l/HZZ4l_2_2_torch.root",
        "delphes_data/HZZ4l/HZZ4l_3_0_torch.root",
        "delphes_data/HZZ4l/HZZ4l_3_1_torch.root",
        "delphes_data/HZZ4l/HZZ4l_3_2_torch.root",
    ]
    
    # Parse command line arguments
    if len(sys.argv) >= 7:
        input_file = sys.argv[1]
        output_files = sys.argv[2:8]
    else:
        input_file = default_input
        output_files = default_outputs
        print(f"Using default files:")
        print(f"  Input: {input_file}")
        for i, out in enumerate(output_files):
            print(f"  Output {i}: {out}")
    
    # Run main processing
    main(input_file, *output_files, max_events=None)
    
    toc = time.time()
    dur = toc - tic
    print(f"\n{'='*80}")
    print(f"Total execution time on {DEVICE}: {dur//60:.0f} minutes, {dur%60:.2f} seconds")
    print(f"{'='*80}\n")
