"""
Apply PyTorch Delphes Efficiency and MomentumSmearing modules to ROOT file and save outputs.

This script emulates six delphes cards:
1. delphes_card_CMS_2_0.tcl: Applies ChargedHadronTrackingEfficiency only
2. delphes_card_CMS_2_1.tcl: Applies ChargedHadronTrackingEfficiency + ElectronTrackingEfficiency
3. delphes_card_CMS_2_2.tcl: Applies ChargedHadronTrackingEfficiency + ElectronTrackingEfficiency + MuonTrackingEfficiency
4. delphes_card_CMS_3_0.tcl: Applies all efficiency modules + ChargedHadronMomentumSmearing
5. delphes_card_CMS_3_1.tcl: Applies all efficiency modules + ChargedHadronMomentumSmearing + ElectronMomentumSmearing
6. delphes_card_CMS_3_2.tcl: Applies all efficiency modules + ChargedHadronMomentumSmearing + ElectronMomentumSmearing + MuonMomentumSmearing

Process:
1. Reads particles from HZZ4l_1.root (output after ParticlePropagator)
2. Applies tracking efficiency modules using PyTorch
3. Applies momentum smearing to charged hadrons, electrons, and muons
4. Writes six output ROOT files with different structures

Usage:
    python test_torch_delphes.py [input.root] [output_v2_0.root] [output_v2_1.root] [output_v2_2.root] [output_v3_0.root] [output_v3_1.root] [output_v3_2.root]
    
Default:
    Input:  delphes_data/HZZ4l/HZZ4l_1.root
    Output: delphes_data/HZZ4l/HZZ4l_2_0_torch.root (ChargedHadron only)
    Output: delphes_data/HZZ4l/HZZ4l_2_1_torch.root (ChargedHadron + Electron)
    Output: delphes_data/HZZ4l/HZZ4l_2_2_torch.root (ChargedHadron + Electron + Muon)
    Output: delphes_data/HZZ4l/HZZ4l_3_0_torch.root (ChargedHadron + Electron + Muon + ChargedHadronSmearing)
    Output: delphes_data/HZZ4l/HZZ4l_3_1_torch.root (ChargedHadron + Electron + Muon + ChargedHadronSmearing + ElectronSmearing)
    Output: delphes_data/HZZ4l/HZZ4l_3_2_torch.root (ChargedHadron + Electron + Muon + ChargedHadronSmearing + ElectronSmearing + MuonSmearing)
"""

import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import uproot
import awkward as ak
import time

# Set PyTorch to use maximum precision (double precision / float64)
torch.set_default_dtype(torch.float64)

# Seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from parnassus.torch_delphes import Efficiency, MomentumSmearing

DEVICE = "cpu"
# DEVICE = "cuda"


def read_branch_data(tree, branch_name, max_events=None):
    """
    Read all attributes from a particle branch.
    
    Args:
        tree: uproot TTree
        branch_name: Name of the branch to read
        max_events: Maximum number of events to read
        
    Returns:
        awkward array with all branch data
    """
    # Determine if this is a GenParticle or Track branch
    # GenParticle branches: ParticleBeforeProp, ParticleAfterProp
    # Track branches: ChargedHadron, Electron, Muon
    
    if branch_name in ['ParticleBeforeProp', 'ParticleAfterProp']:
        # GenParticle attributes
        physics_attrs = ['PID', 'Status', 'Charge', 'E', 'Px', 'Py', 'Pz', 
                         'PT', 'Eta', 'Phi', 'T', 'X', 'Y', 'Z']
    else:
        # Track attributes (ChargedHadron, Electron, Muon)
        # Tracks don't have Status, E, Px, Py, Pz but have P instead
        # Include EtaOuter for position-based eta (used for efficiency calculation in C++ Delphes)
        physics_attrs = ['PID', 'Charge', 'P', 'PT', 'Eta', 'EtaOuter', 'Phi', 'T', 'X', 'Y', 'Z']
    
    # Build list of branch keys for physics attributes
    branch_keys = [f"{branch_name}/{branch_name}.{attr}" for attr in physics_attrs]
    
    # Read all attributes
    arrays = tree.arrays(branch_keys, entry_stop=max_events, library="ak")
    
    return arrays


def apply_tracking_efficiency(track_arrays, branch_prefix, efficiency_formula='charged_hadron_cms', module_name='TrackingEfficiency'):
    """
    Apply tracking efficiency to track data (charged hadrons, electrons, or muons).
    
    Args:
        track_arrays: awkward array with track data (Track objects)
        branch_prefix: prefix for accessing branch attributes
        efficiency_formula: which efficiency formula to use
        module_name: name for logging purposes
        
    Returns:
        List of filtered awkward arrays (one per event)
    """
    # Initialize efficiency module
    eff_module = Efficiency(
        efficiency_formula=efficiency_formula,
        deterministic=False,
        device=DEVICE
    )
    
    n_events = len(track_arrays[f"{branch_prefix}.PT"])
    filtered_events = []
    
    for i in tqdm(range(n_events)):
        # Extract data for this event
        n_particles = len(track_arrays[f"{branch_prefix}.PT"][i])
        
        if n_particles == 0:
            # Empty event - create empty output
            filtered_events.append({})
            continue
        
        # Build tensor (N, 15)
        # Track objects don't have Status, E, Px, Py, Pz - we'll set these to dummy values
        # The efficiency module only uses PT (column 7) and Eta (column 8)
        # NOTE: C++ Delphes uses Position.Eta() by default, which for Track objects
        # corresponds to EtaOuter (position eta at detector surface), not momentum Eta
        particles = np.zeros((n_particles, 15))
        
        # Column 0: PID
        particles[:, 0] = np.array(track_arrays[f"{branch_prefix}.PID"][i])
        # Column 1: Status (not available for Track, set to 1)
        particles[:, 1] = 1
        # Column 2: Charge
        particles[:, 2] = np.array(track_arrays[f"{branch_prefix}.Charge"][i])
        # Column 3: E (not available for Track, will compute from P)
        P = np.array(track_arrays[f"{branch_prefix}.P"][i])
        particles[:, 3] = P  # Approximate E ≈ P for high-energy particles
        # Columns 4-6: Px, Py, Pz (not available, compute from P, PT, Eta, Phi)
        PT = np.array(track_arrays[f"{branch_prefix}.PT"][i])
        Eta = np.array(track_arrays[f"{branch_prefix}.Eta"][i])
        EtaOuter = np.array(track_arrays[f"{branch_prefix}.EtaOuter"][i])  # Position eta
        Phi = np.array(track_arrays[f"{branch_prefix}.Phi"][i])
        particles[:, 4] = PT * np.cos(Phi)  # Px
        particles[:, 5] = PT * np.sin(Phi)  # Py
        particles[:, 6] = PT * np.sinh(Eta)  # Pz
        # Column 7: PT (already have this)
        particles[:, 7] = PT
        # Column 8: Eta - USE ETAOUTER to match C++ Delphes behavior!
        particles[:, 8] = EtaOuter  # Position eta (what C++ Delphes uses by default)
        # Column 9: Phi (already have this)
        particles[:, 9] = Phi
        # Column 10: T
        particles[:, 10] = np.array(track_arrays[f"{branch_prefix}.T"][i])
        # Columns 11-13: X, Y, Z
        particles[:, 11] = np.array(track_arrays[f"{branch_prefix}.X"][i])
        particles[:, 12] = np.array(track_arrays[f"{branch_prefix}.Y"][i])
        particles[:, 13] = np.array(track_arrays[f"{branch_prefix}.Z"][i])
        
        # Convert to torch tensor
        particles_tensor = torch.from_numpy(particles).float()
        
        # Apply efficiency
        filtered_tensor, mask = eff_module(particles_tensor, return_mask=True)
        mask_np = mask.cpu().numpy()
        
        # Extract filtered data for each attribute
        filtered_event = {}
        branch_keys = [key for key in track_arrays.fields if key.startswith(branch_prefix)]
        for key in branch_keys:
            attr_name = key.split('.')[-1]  # Get attribute name (e.g., "PID", "PT", etc.)
            original_data = np.array(track_arrays[key][i])
            filtered_event[attr_name] = original_data[mask_np]
        
        filtered_events.append(filtered_event)
    
    return filtered_events


def apply_momentum_smearing(filtered_particles, smearing_formula='charged_hadron_cms', module_name='MomentumSmearing'):
    """
    Apply momentum smearing to filtered particles.
    
    Args:
        filtered_particles: List of filtered particle events (from apply_tracking_efficiency)
        smearing_formula: which resolution formula to use
        module_name: name for logging purposes
        
    Returns:
        List of smeared awkward arrays (one per event)
    """
    # Initialize smearing module
    smearing_module = MomentumSmearing(
        resolution_formula=smearing_formula,
        deterministic=False,
        device=DEVICE
    )
    
    n_events = len(filtered_particles)
    smeared_events = []
    
    for i in tqdm(range(n_events)):
        event = filtered_particles[i]
        
        if not event or len(event.get('PT', [])) == 0:
            # Empty event - create empty output
            smeared_events.append({})
            continue
        
        # Build tensor (N, 15) from filtered event data
        n_particles = len(event['PT'])
        particles = np.zeros((n_particles, 15))
        
        # Column 0: PID
        particles[:, 0] = event['PID']
        # Column 1: Status (not available for Track, set to 1)
        particles[:, 1] = 1
        # Column 2: Charge
        particles[:, 2] = event['Charge']
        # Column 3: E (compute from P if available, else approximate)
        if 'P' in event:
            particles[:, 3] = event['P']  # Approximate E ≈ P for high energy
        else:
            particles[:, 3] = event['PT']  # Fallback
        # Columns 4-6: Px, Py, Pz (compute from PT, Eta, Phi using momentum-based values)
        PT = event['PT']
        Eta = event.get('Eta', np.zeros_like(PT))  # Momentum-based Eta
        EtaOuter = event.get('EtaOuter', Eta)  # Position-based Eta
        Phi = event['Phi']
        particles[:, 4] = PT * np.cos(Phi)  # Px
        particles[:, 5] = PT * np.sin(Phi)  # Py
        particles[:, 6] = PT * np.sinh(Eta)  # Pz (use momentum Eta)
        # Column 7: PT
        particles[:, 7] = PT
        # Column 8: Eta (use EtaOuter for position-based resolution formula evaluation)
        # Note: MomentumSmearing will recompute momentum Eta from Px,Py,Pz for 4-vector reconstruction
        particles[:, 8] = EtaOuter
        # Column 9: Phi
        particles[:, 9] = Phi
        # Column 10: T
        particles[:, 10] = event.get('T', np.zeros_like(PT))
        # Columns 11-13: X, Y, Z
        particles[:, 11] = event.get('X', np.zeros_like(PT))
        particles[:, 12] = event.get('Y', np.zeros_like(PT))
        particles[:, 13] = event.get('Z', np.zeros_like(PT))
        # Column 14: Mass (assume pion mass for charged hadrons)
        particles[:, 14] = 0.140  # GeV
        
        # Convert to torch tensor
        particles_tensor = torch.from_numpy(particles).float()
        
        # Apply smearing
        smeared_tensor = smearing_module(particles_tensor)
        smeared_np = smeared_tensor.cpu().numpy()
        
        # Extract smeared data back into event dictionary
        smeared_event = {}
        smeared_event['PID'] = smeared_np[:, 0].astype(int)
        smeared_event['Charge'] = smeared_np[:, 2].astype(int)
        smeared_event['P'] = np.sqrt(smeared_np[:, 4]**2 + smeared_np[:, 5]**2 + smeared_np[:, 6]**2)
        smeared_event['PT'] = smeared_np[:, 7]
        smeared_event['Eta'] = event['Eta']  # Preserve original momentum eta
        smeared_event['EtaOuter'] = event['EtaOuter']  # Preserve position eta
        smeared_event['Phi'] = smeared_np[:, 9]
        smeared_event['T'] = event.get('T', np.zeros(len(smeared_event['PT'])))
        smeared_event['X'] = event.get('X', np.zeros(len(smeared_event['PT'])))
        smeared_event['Y'] = event.get('Y', np.zeros(len(smeared_event['PT'])))
        smeared_event['Z'] = event.get('Z', np.zeros(len(smeared_event['PT'])))
        
        smeared_events.append(smeared_event)
    
    return smeared_events


def load_base_branches(input_tree, max_events=None):
    """
    Load base branches from input ROOT file into a dictionary.
    Returns awkward array dictionary with all base particle branches.
    """
    branch_names = ['ParticleBeforeProp', 'ParticleAfterProp', 'ChargedHadron', 'Electron', 'Muon']
    output_data = {}
    
    for branch_name in branch_names:
        branch_data = read_branch_data(input_tree, branch_name, max_events)
        
        for key in branch_data.fields:
            output_key = f"{branch_name}.{key.split('.')[-1]}"
            output_data[output_key] = branch_data[key]
    
    return output_data


def add_particle_branch(tree_data, particles, branch_name):
    """
    Add a particle branch to tree data dictionary.
    
    This is a generic function used to add any particle branch (efficiency outputs,
    smearing outputs, etc.) to the ROOT tree data structure.
    
    Args:
        tree_data: Dictionary of awkward arrays representing tree branches
        particles: List of event particles (from apply_tracking_efficiency or apply_momentum_smearing)
        branch_name: Name for the branch (e.g., "ChargedHadronEfficiency", "ChargedHadronSmeared")
    
    Returns:
        Updated tree_data dictionary with new branch added
    """
    # Get attribute names from first non-empty event
    attr_names = None
    for event in particles:
        if event:
            attr_names = list(event.keys())
            break
    
    # Add each attribute as a branch
    for attr_name in attr_names:
        data_list = []
        for event in particles:
            if event and attr_name in event:
                data_list.append(event[attr_name])
            else:
                data_list.append(np.array([]))
        
        tree_data[f"{branch_name}.{attr_name}"] = ak.Array(data_list)
    
    return tree_data


def write_output_root(output_file, tree_data):
    """
    Write tree data dictionary to ROOT file.
    
    Args:
        output_file: Path to output ROOT file
        tree_data: Dictionary of awkward arrays representing all tree branches
    """
    print(f"  Writing {len(tree_data)} branches to file...")
    with uproot.recreate(output_file) as f:
        f["Delphes"] = tree_data
    print(f"✓ Output written successfully!")


def validate_against_benchmark(torch_output_file, benchmark_file, output_dir):
    """
    Validate PyTorch Delphes implementation against C++ Delphes benchmark.
    
    Args:
        torch_output_file: Path to PyTorch output ROOT file (e.g., HZZ4l_2_2_torch.root)
        benchmark_file: Path to benchmark ROOT file from C++ Delphes
        output_dir: Directory to save validation plots
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PyTorch output from disk
    torch_root = uproot.open(torch_output_file)
    torch_tree = torch_root["Delphes"]
    
    # Load benchmark data
    benchmark_root = uproot.open(benchmark_file)
    benchmark_tree = benchmark_root["Delphes"]
    
    # Kinematic variables to compare
    # Note: Track objects (including smeared particles) don't have E, Px, Py, Pz
    kinematic_vars = ['Charge', 'E', 'P', 'Px', 'Py', 'Pz', 'PT', 'Eta', 'Phi']
    
    # Branches to validate (efficiency modules + momentum smearing)
    branches = ['ChargedHadronEfficiency', 'ElectronEfficiency', 'MuonEfficiency', 
                'ChargedHadronSmeared', 'ElectronSmeared', 'MuonSmeared']
    
    for branch_name in branches:
        print(f"\nValidating {branch_name}...")
        
        # Create branch-specific directory
        branch_dir = output_dir / branch_name
        branch_dir.mkdir(exist_ok=True)
        
        # Check if branch exists in PyTorch output
        torch_branch_keys = [k for k in torch_tree.keys() if k.startswith(f"{branch_name}.")]
        if not torch_branch_keys:
            print(f"  ⚠ {branch_name} not found in PyTorch output, skipping...")
            continue
        
        for var in kinematic_vars:
            # Check if variable exists in both datasets
            torch_key = f"{branch_name}.{var}"
            benchmark_key = f"{branch_name}/{branch_name}.{var}"
            
            if torch_key not in torch_tree.keys():
                print(f"  ⚠ {var} not found in PyTorch {branch_name}, skipping...")
                continue
            
            if benchmark_key not in benchmark_tree.keys():
                print(f"  ⚠ {var} not found in C++ {branch_name}, skipping...")
                continue
            
            try:
                # Load data from both sources (both from disk)
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
                ax.set_ylabel('Normalized Counts', fontsize=12)
                ax.set_title(f'{branch_name}: {var}', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                # Save plot
                plot_file = branch_dir / f"{var}.png"
                plt.tight_layout()
                plt.savefig(plot_file, dpi=150)
                plt.close()
                
                print(f"  ✓ {var}: saved to {plot_file.name}")
                
            except Exception as e:
                print(f"  ✗ {var}: Error - {e}")
                continue
    
    print(f"\n✓ Validation complete! Plots saved to {output_dir}")


def main(input_file, output_file_v2_0, output_file_v2_1, output_file_v2_2, output_file_v3_0, output_file_v3_1, output_file_v3_2, max_events=None):
    """Main processing function."""
    
    ############################## Open input ROOT file #####################################
    print("\n")
    print(f"Input:  {input_file}")
    input_root = uproot.open(input_file)
    tree = input_root["Delphes"]
    tree_data = dict(load_base_branches(tree, max_events))
    n_events = tree.num_entries if max_events is None else min(max_events, tree.num_entries)
    print(f"Total events in file: {tree.num_entries}")
    print(f"Processing: {n_events} events\n")
    
    ############################## ChargedHadronTrackingEfficiency #####################################
    print("\nChargedHadronTrackingEfficiency...")
    charged_hadrons = read_branch_data(tree, "ChargedHadron", max_events)
    
    sample_key = list(charged_hadrons.fields)[0]
    ch_prefix = sample_key.rsplit('.', 1)[0]
    total_input_ch = sum(len(charged_hadrons[f"{ch_prefix}.PT"][i]) 
                         for i in range(n_events))
    filtered_charged_hadrons = apply_tracking_efficiency(
        charged_hadrons, 
        ch_prefix,
        efficiency_formula='charged_hadron_cms',
        module_name='ChargedHadronTrackingEfficiency'
    )

    tree_data = add_particle_branch(tree_data, filtered_charged_hadrons, "ChargedHadronEfficiency")
    write_output_root(output_file_v2_0, tree_data)

    
    ############################## ElectronTrackingEfficiency #####################################
    print("\nElectronTrackingEfficiency...")
    electrons = read_branch_data(tree, "Electron", max_events)
    
    sample_key_el = list(electrons.fields)[0]
    el_prefix = sample_key_el.rsplit('.', 1)[0]
    total_input_el = sum(len(electrons[f"{el_prefix}.PT"][i]) 
                         for i in range(n_events))
    filtered_electrons = apply_tracking_efficiency(
        electrons,
        el_prefix,
        efficiency_formula='electron_cms',
        module_name='ElectronTrackingEfficiency'
    )

    tree_data = add_particle_branch(tree_data, filtered_electrons, "ElectronEfficiency")
    write_output_root(output_file_v2_1, tree_data)

    ############################## MuonTrackingEfficiency #####################################
    print("\nMuonTrackingEfficiency...")
    muons = read_branch_data(tree, "Muon", max_events)
    
    sample_key_mu = list(muons.fields)[0]
    mu_prefix = sample_key_mu.rsplit('.', 1)[0]
    total_input_mu = sum(len(muons[f"{mu_prefix}.PT"][i]) 
                         for i in range(n_events))
    filtered_muons = apply_tracking_efficiency(
        muons,
        mu_prefix,
        efficiency_formula='muon_cms',
        module_name='MuonTrackingEfficiency'
    )

    tree_data = add_particle_branch(tree_data, filtered_muons, "MuonEfficiency")
    write_output_root(output_file_v2_2, tree_data)

    ############################## ChargedHadronMomentumSmearing #####################################
    print("\nChargedHadronMomentumSmearing...")
    smeared_charged_hadrons = apply_momentum_smearing(
        filtered_charged_hadrons,
        smearing_formula='charged_hadron_cms',
        module_name='ChargedHadronMomentumSmearing'
    )
    
    tree_data = add_particle_branch(tree_data, smeared_charged_hadrons, "ChargedHadronSmeared")
    write_output_root(output_file_v3_0, tree_data)

    ############################## ElectronMomentumSmearing #####################################
    print("\nElectronMomentumSmearing...")
    smeared_electrons = apply_momentum_smearing(
        filtered_electrons,
        smearing_formula='electron_cms',
        module_name='ElectronMomentumSmearing'
    )
    
    tree_data = add_particle_branch(tree_data, smeared_electrons, "ElectronSmeared")
    write_output_root(output_file_v3_1, tree_data)

    ############################## MuonMomentumSmearing #####################################
    print("\nMuonMomentumSmearing...")
    smeared_muons = apply_momentum_smearing(
        filtered_muons,
        smearing_formula='muon_cms',
        module_name='MuonMomentumSmearing'
    )
    
    tree_data = add_particle_branch(tree_data, smeared_muons, "MuonSmeared")
    write_output_root(output_file_v3_2, tree_data)

    ############################## Summary #####################################

    total_output_ch = sum(len(event.get('PT', [])) for event in filtered_charged_hadrons)
    total_output_el = sum(len(event.get('PT', [])) for event in filtered_electrons)
    total_output_mu = sum(len(event.get('PT', [])) for event in filtered_muons)
    total_smeared_ch = sum(len(event.get('PT', [])) for event in smeared_charged_hadrons)
    total_smeared_el = sum(len(event.get('PT', [])) for event in smeared_electrons)
    total_smeared_mu = sum(len(event.get('PT', [])) for event in smeared_muons)
    
    print(f"\nStatistics:")
    print(f"  ChargedHadrons: {total_input_ch} → {total_output_ch} ({total_output_ch/total_input_ch*100:.2f}%)")
    print(f"  Electrons:      {total_input_el} → {total_output_el} ({total_output_el/total_input_el*100:.2f}%)")
    print(f"  Muons:          {total_input_mu} → {total_output_mu} ({total_output_mu/total_input_mu*100:.2f}%)")
    print(f"  ChargedHadronsSmeared: {total_output_ch} → {total_smeared_ch} (100.00%)")
    print(f"  ElectronsSmeared:      {total_output_el} → {total_smeared_el} (100.00%)")
    print(f"  MuonsSmeared:          {total_output_mu} → {total_smeared_mu} (100.00%)")
    
    
    print(f"\n{'='*70}")
    print("Processing Complete!")
    print(f"{'='*70}")
    print(f"n_events: {n_events}")
    print(f"\nOutput files created:")
    print("TrackingEfficiency:")
    print(f"  v2.0 (ChargedHadron): {output_file_v2_0}")
    print(f"  v2.1 (ChargedHadron + Electron): {output_file_v2_1}")
    print(f"  v2.2 (ChargedHadron + Electron + Muon): {output_file_v2_2}")
    print("TrackingEfficiency + MomentumSmearing:")
    print(f"  v3.0 (+ ChargedHadronSmearing): {output_file_v3_0}")
    print(f"  v3.1 (+ ChargedHadronSmearing + ElectronSmearing): {output_file_v3_1}")
    print(f"  v3.2 (+ ChargedHadronSmearing + ElectronSmearing + MuonSmearing): {output_file_v3_2}")
    print()
    
    ############################## Validation #####################################

    # Validate against C++ Delphes benchmark
    script_dir = Path(__file__).parent
    benchmark_file = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_3_2.root"
    validation_dir = script_dir / "torch_delphes_validation"
    print(f"\n{'='*70}")
    print("Validation: Comparing PyTorch Delphes vs C++ Delphes")
    print(f"{'='*70}")
    print(f"Benchmark file: {benchmark_file}")
    print(f"Validation directory: {validation_dir}")

    if benchmark_file.exists():
        validate_against_benchmark(output_file_v3_2, benchmark_file, validation_dir)
    else:
        print(f"\n⚠ Benchmark file not found: {benchmark_file}")
        print("  Skipping validation. To enable validation, provide HZZ4l_3_2.root")


if __name__ == "__main__":
    tic = time.time()

    # Set default paths relative to this file
    script_dir = Path(__file__).parent
    default_input = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_1.root"
    default_output_v2_0 = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_2_0_torch.root"
    default_output_v2_1 = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_2_1_torch.root"
    default_output_v2_2 = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_2_2_torch.root"
    default_output_v3_0 = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_3_0_torch.root"
    default_output_v3_1 = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_3_1_torch.root"
    default_output_v3_2 = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_3_2_torch.root"
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments - use defaults
        input_file = str(default_input)
        output_file_v2_0 = str(default_output_v2_0)
        output_file_v2_1 = str(default_output_v2_1)
        output_file_v2_2 = str(default_output_v2_2)
        output_file_v3_0 = str(default_output_v3_0)
        output_file_v3_1 = str(default_output_v3_1)
        output_file_v3_2 = str(default_output_v3_2)
        max_events = None
    elif len(sys.argv) >= 8:
        # Input and outputs specified
        input_file = sys.argv[1]
        output_file_v2_0 = sys.argv[2]
        output_file_v2_1 = sys.argv[3]
        output_file_v2_2 = sys.argv[4]
        output_file_v3_0 = sys.argv[5]
        output_file_v3_1 = sys.argv[6]
        output_file_v3_2 = sys.argv[7]
        max_events = int(sys.argv[8]) if len(sys.argv) > 8 else None
    else:
        print("Usage: python test_torch_delphes.py [input.root] [output_v2_0.root] [output_v2_1.root] [output_v2_2.root] [output_v3_0.root] [output_v3_1.root] [output_v3_2.root] [max_events]")
        print("\nRun with no arguments to use defaults:")
        print(f"  Input:       {default_input}")
        print(f"  Output v2.0: {default_output_v2_0}")
        print(f"  Output v2.1: {default_output_v2_1}")
        print(f"  Output v2.2: {default_output_v2_2}")
        print(f"  Output v3.0: {default_output_v3_0}")
        print(f"  Output v3.1: {default_output_v3_1}")
        print(f"  Output v3.2: {default_output_v3_2}")
        sys.exit(1)
    
    main(input_file, output_file_v2_0, output_file_v2_1, output_file_v2_2, output_file_v3_0, output_file_v3_1, output_file_v3_2, max_events)

    toc = time.time()
    dur = toc - tic
    print(f"Total duration on {DEVICE}: {dur//60} minutes, {dur%60:.2f} seconds")