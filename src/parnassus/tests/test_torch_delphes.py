"""
Apply PyTorch Delphes Efficiency modules to ROOT file and save outputs.

This script emulates three delphes cards:
1. delphes_card_CMS_2_0.tcl: Applies ChargedHadronTrackingEfficiency only
2. delphes_card_CMS_2_1.tcl: Applies ChargedHadronTrackingEfficiency + ElectronTrackingEfficiency
3. delphes_card_CMS_2_2.tcl: Applies ChargedHadronTrackingEfficiency + ElectronTrackingEfficiency + MuonTrackingEfficiency

Process:
1. Reads particles from HZZ4l_1.root (output after ParticlePropagator)
2. Applies tracking efficiency modules using PyTorch
3. Writes three output ROOT files with different structures

Usage:
    python test_torch_delphes.py [input.root] [output_v2_0.root] [output_v2_1.root] [output_v2_2.root]
    
Default:
    Input:  delphes_data/HZZ4l/HZZ4l_1.root
    Output: delphes_data/HZZ4l/HZZ4l_2_0_torch.root (ChargedHadron only)
    Output: delphes_data/HZZ4l/HZZ4l_2_1_torch.root (ChargedHadron + Electron)
    Output: delphes_data/HZZ4l/HZZ4l_2_2_torch.root (ChargedHadron + Electron + Muon)
"""

import sys
import numpy as np
import torch
from pathlib import Path


import uproot
import awkward as ak


from parnassus.torch_delphes.Efficiency import DelphesEfficiency


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
        physics_attrs = ['PID', 'Charge', 'P', 'PT', 'Eta', 'Phi', 'T', 'X', 'Y', 'Z']
    
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
    eff_module = DelphesEfficiency(
        efficiency_formula=efficiency_formula,
        deterministic=False,
        device='cpu'
    )
    
    n_events = len(track_arrays[f"{branch_prefix}.PT"])
    filtered_events = []
    
    print(f"Applying {module_name} to {n_events} events...")
    
    for i in range(n_events):
        # Extract data for this event
        n_particles = len(track_arrays[f"{branch_prefix}.PT"][i])
        
        if n_particles == 0:
            # Empty event - create empty output
            filtered_events.append({})
            continue
        
        # Build tensor (N, 15)
        # Track objects don't have Status, E, Px, Py, Pz - we'll set these to dummy values
        # The efficiency module only uses PT (column 7) and Eta (column 8)
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
        Phi = np.array(track_arrays[f"{branch_prefix}.Phi"][i])
        particles[:, 4] = PT * np.cos(Phi)  # Px
        particles[:, 5] = PT * np.sin(Phi)  # Py
        particles[:, 6] = PT * np.sinh(Eta)  # Pz
        # Column 7: PT (already have this)
        particles[:, 7] = PT
        # Column 8: Eta (already have this)
        particles[:, 8] = Eta
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
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_events} events")
    
    return filtered_events


def write_output_root_v2_0(output_file, input_tree, filtered_charged_hadrons, max_events=None):
    """
    Write output ROOT file matching delphes_card_CMS_2_0.tcl structure.
    Includes ChargedHadronEfficiency only.
    """
    print(f"\nWriting output file (v2.0): {output_file}")
    
    branch_names = ['ParticleBeforeProp', 'ParticleAfterProp', 'ChargedHadron', 'Electron', 'Muon']
    output_data = {}
    
    # Read all existing branches from input
    for branch_name in branch_names:
        print(f"  Reading {branch_name}...")
        branch_data = read_branch_data(input_tree, branch_name, max_events)
        
        for key in branch_data.fields:
            output_key = f"{branch_name}.{key.split('.')[-1]}"
            output_data[output_key] = branch_data[key]
    
    # Add filtered charged hadron efficiency branch
    print(f"  Adding ChargedHadronEfficiency branch...")
    attr_names = None
    for event in filtered_charged_hadrons:
        if event:
            attr_names = list(event.keys())
            break
    
    for attr_name in attr_names:
        data_list = []
        for event in filtered_charged_hadrons:
            if event and attr_name in event:
                data_list.append(event[attr_name])
            else:
                data_list.append(np.array([]))
        
        output_data[f"ChargedHadronEfficiency.{attr_name}"] = ak.Array(data_list)
    
    # Write to ROOT file
    print(f"  Writing {len(output_data)} branches to file...")
    with uproot.recreate(output_file) as f:
        f["Delphes"] = output_data
    
    print(f"✓ Output v2.0 written successfully!")


def write_output_root_v2_1(output_file, input_tree, filtered_charged_hadrons, filtered_electrons, max_events=None):
    """
    Write output ROOT file matching delphes_card_CMS_2_1.tcl structure.
    Includes both ChargedHadronEfficiency and ElectronEfficiency.
    """
    print(f"\nWriting output file (v2.1): {output_file}")
    
    branch_names = ['ParticleBeforeProp', 'ParticleAfterProp', 'ChargedHadron', 'Electron', 'Muon']
    output_data = {}
    
    # Read all existing branches from input
    for branch_name in branch_names:
        print(f"  Reading {branch_name}...")
        branch_data = read_branch_data(input_tree, branch_name, max_events)
        
        for key in branch_data.fields:
            output_key = f"{branch_name}.{key.split('.')[-1]}"
            output_data[output_key] = branch_data[key]
    
    # Add filtered charged hadron efficiency branch
    print(f"  Adding ChargedHadronEfficiency branch...")
    attr_names_ch = None
    for event in filtered_charged_hadrons:
        if event:
            attr_names_ch = list(event.keys())
            break
    
    for attr_name in attr_names_ch:
        data_list = []
        for event in filtered_charged_hadrons:
            if event and attr_name in event:
                data_list.append(event[attr_name])
            else:
                data_list.append(np.array([]))
        
        output_data[f"ChargedHadronEfficiency.{attr_name}"] = ak.Array(data_list)
    
    # Add filtered electron efficiency branch
    print(f"  Adding ElectronEfficiency branch...")
    attr_names_el = None
    for event in filtered_electrons:
        if event:
            attr_names_el = list(event.keys())
            break
    
    for attr_name in attr_names_el:
        data_list = []
        for event in filtered_electrons:
            if event and attr_name in event:
                data_list.append(event[attr_name])
            else:
                data_list.append(np.array([]))
        
        output_data[f"ElectronEfficiency.{attr_name}"] = ak.Array(data_list)
    
    # Write to ROOT file
    print(f"  Writing {len(output_data)} branches to file...")
    with uproot.recreate(output_file) as f:
        f["Delphes"] = output_data
    
    print(f"✓ Output v2.1 written successfully!")


def write_output_root_v2_2(input_tree, filtered_charged_hadrons, filtered_electrons, filtered_muons, output_file, max_events=None):
    """
    Write output ROOT file matching delphes_card_CMS_2_2.tcl structure.
    Includes: All base branches + ChargedHadronEfficiency + ElectronEfficiency + MuonEfficiency
    """
    print(f"\nWriting output file (v2.2): {output_file}")
    
    branch_names = ['ParticleBeforeProp', 'ParticleAfterProp', 'ChargedHadron', 'Electron', 'Muon']
    output_data = {}
    
    # Read all existing branches from input
    for branch_name in branch_names:
        print(f"  Reading {branch_name}...")
        branch_data = read_branch_data(input_tree, branch_name, max_events)
        
        for key in branch_data.fields:
            output_key = f"{branch_name}.{key.split('.')[-1]}"
            output_data[output_key] = branch_data[key]
    
    # Add filtered charged hadron efficiency branch
    print(f"  Adding ChargedHadronEfficiency branch...")
    attr_names_ch = None
    for event in filtered_charged_hadrons:
        if event:
            attr_names_ch = list(event.keys())
            break
    
    for attr_name in attr_names_ch:
        data_list = []
        for event in filtered_charged_hadrons:
            if event and attr_name in event:
                data_list.append(event[attr_name])
            else:
                data_list.append(np.array([]))
        
        output_data[f"ChargedHadronEfficiency.{attr_name}"] = ak.Array(data_list)
    
    # Add filtered electron efficiency branch
    print(f"  Adding ElectronEfficiency branch...")
    attr_names_el = None
    for event in filtered_electrons:
        if event:
            attr_names_el = list(event.keys())
            break
    
    for attr_name in attr_names_el:
        data_list = []
        for event in filtered_electrons:
            if event and attr_name in event:
                data_list.append(event[attr_name])
            else:
                data_list.append(np.array([]))
        
        output_data[f"ElectronEfficiency.{attr_name}"] = ak.Array(data_list)
    
    # Add filtered muon efficiency branch
    print(f"  Adding MuonEfficiency branch...")
    attr_names_mu = None
    for event in filtered_muons:
        if event:
            attr_names_mu = list(event.keys())
            break
    
    for attr_name in attr_names_mu:
        data_list = []
        for event in filtered_muons:
            if event and attr_name in event:
                data_list.append(event[attr_name])
            else:
                data_list.append(np.array([]))
        
        output_data[f"MuonEfficiency.{attr_name}"] = ak.Array(data_list)
    
    # Write to ROOT file
    print(f"  Writing {len(output_data)} branches to file...")
    with uproot.recreate(output_file) as f:
        f["Delphes"] = output_data
    
    print(f"✓ Output v2.2 written successfully!")


def main(input_file, output_file_v2_0, output_file_v2_1, output_file_v2_2, max_events=None):
    """Main processing function."""
    
    print("="*70)
    print("PyTorch Delphes Tracking Efficiency Modules")
    print("="*70)
    print(f"Input:  {input_file}")
    print(f"Output v2.0: {output_file_v2_0} (ChargedHadron only)")
    print(f"Output v2.1: {output_file_v2_1} (ChargedHadron + Electron)")
    print(f"Output v2.2: {output_file_v2_2} (ChargedHadron + Electron + Muon)")
    print(f"Max events: {max_events if max_events else 'All'}")
    print()
    
    # Open input ROOT file
    input_root = uproot.open(input_file)
    tree = input_root["Delphes"]
    
    n_events = tree.num_entries if max_events is None else min(max_events, tree.num_entries)
    print(f"Total events in file: {tree.num_entries}")
    print(f"Processing: {n_events} events\n")
    
    # Read charged hadrons (output from ParticlePropagator)
    print("Reading ChargedHadron branch...")
    charged_hadrons = read_branch_data(tree, "ChargedHadron", max_events)
    
    # Determine actual branch prefix
    sample_key = list(charged_hadrons.fields)[0]
    ch_prefix = sample_key.rsplit('.', 1)[0]
    
    print(f"Found {len(charged_hadrons.fields)} attributes for charged hadrons")
    
    # Count input particles
    total_input_ch = sum(len(charged_hadrons[f"{ch_prefix}.PT"][i]) 
                         for i in range(n_events))
    print(f"Total input charged hadrons: {total_input_ch}\n")
    
    # Read electrons (output from ParticlePropagator)
    print("Reading Electron branch...")
    electrons = read_branch_data(tree, "Electron", max_events)
    
    # Determine actual branch prefix
    sample_key_el = list(electrons.fields)[0]
    el_prefix = sample_key_el.rsplit('.', 1)[0]
    
    print(f"Found {len(electrons.fields)} attributes for electrons")
    
    # Count input particles
    total_input_el = sum(len(electrons[f"{el_prefix}.PT"][i]) 
                         for i in range(n_events))
    print(f"Total input electrons: {total_input_el}\n")
    
    # Read muons (output from ParticlePropagator)
    print("Reading Muon branch...")
    muons = read_branch_data(tree, "Muon", max_events)
    
    # Determine actual branch prefix
    sample_key_mu = list(muons.fields)[0]
    mu_prefix = sample_key_mu.rsplit('.', 1)[0]
    
    print(f"Found {len(muons.fields)} attributes for muons")
    
    # Count input particles
    total_input_mu = sum(len(muons[f"{mu_prefix}.PT"][i]) 
                         for i in range(n_events))
    print(f"Total input muons: {total_input_mu}\n")
    
    # Apply ChargedHadronTrackingEfficiency
    filtered_charged_hadrons = apply_tracking_efficiency(
        charged_hadrons, 
        ch_prefix,
        efficiency_formula='charged_hadron_cms',
        module_name='ChargedHadronTrackingEfficiency'
    )
    
    # Apply ElectronTrackingEfficiency
    filtered_electrons = apply_tracking_efficiency(
        electrons,
        el_prefix,
        efficiency_formula='electron_cms',
        module_name='ElectronTrackingEfficiency'
    )
    
    # Apply MuonTrackingEfficiency
    filtered_muons = apply_tracking_efficiency(
        muons,
        mu_prefix,
        efficiency_formula='muon_cms',
        module_name='MuonTrackingEfficiency'
    )
    
    # Compute statistics
    total_output_ch = sum(len(event.get('PT', [])) for event in filtered_charged_hadrons)
    total_output_el = sum(len(event.get('PT', [])) for event in filtered_electrons)
    total_output_mu = sum(len(event.get('PT', [])) for event in filtered_muons)
    
    print(f"\nStatistics:")
    print(f"  ChargedHadrons: {total_input_ch} → {total_output_ch} ({total_output_ch/total_input_ch*100:.2f}%)")
    print(f"  Electrons:      {total_input_el} → {total_output_el} ({total_output_el/total_input_el*100:.2f}%)")
    print(f"  Muons:          {total_input_mu} → {total_output_mu} ({total_output_mu/total_input_mu*100:.2f}%)")
    
    # Write output ROOT files
    write_output_root_v2_0(output_file_v2_0, tree, filtered_charged_hadrons, max_events)
    write_output_root_v2_1(output_file_v2_1, tree, filtered_charged_hadrons, filtered_electrons, max_events)
    write_output_root_v2_2(tree, filtered_charged_hadrons, filtered_electrons, filtered_muons, output_file_v2_2, max_events)
    
    print(f"\n{'='*70}")
    print("Processing Complete!")
    print(f"{'='*70}")
    print(f"\nOutput files created:")
    print(f"  v2.0: {output_file_v2_0}")
    print(f"    - Emulates delphes_card_CMS_2_0.tcl")
    print(f"    - ChargedHadronEfficiency only")
    print(f"\n  v2.1: {output_file_v2_1}")
    print(f"    - Emulates delphes_card_CMS_2_1.tcl")
    print(f"    - ChargedHadronEfficiency + ElectronEfficiency")
    print(f"\n  v2.2: {output_file_v2_2}")
    print(f"    - Emulates delphes_card_CMS_2_2.tcl")
    print(f"    - ChargedHadronEfficiency + ElectronEfficiency + MuonEfficiency")


if __name__ == "__main__":
    # Set default paths relative to this file
    script_dir = Path(__file__).parent
    default_input = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_1.root"
    default_output_v2_0 = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_2_0_torch.root"
    default_output_v2_1 = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_2_1_torch.root"
    default_output_v2_2 = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_2_2_torch.root"
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments - use defaults
        input_file = str(default_input)
        output_file_v2_0 = str(default_output_v2_0)
        output_file_v2_1 = str(default_output_v2_1)
        output_file_v2_2 = str(default_output_v2_2)
        max_events = None
    elif len(sys.argv) >= 5:
        # Input and outputs specified
        input_file = sys.argv[1]
        output_file_v2_0 = sys.argv[2]
        output_file_v2_1 = sys.argv[3]
        output_file_v2_2 = sys.argv[4]
        max_events = int(sys.argv[5]) if len(sys.argv) > 5 else None
    else:
        print("Usage: python test_torch_delphes.py [input.root] [output_v2_0.root] [output_v2_1.root] [output_v2_2.root] [max_events]")
        print("\nRun with no arguments to use defaults:")
        print(f"  Input:       {default_input}")
        print(f"  Output v2.0: {default_output_v2_0}")
        print(f"  Output v2.1: {default_output_v2_1}")
        print(f"  Output v2.2: {default_output_v2_2}")
        sys.exit(1)
    
    main(input_file, output_file_v2_0, output_file_v2_1, output_file_v2_2, max_events)

