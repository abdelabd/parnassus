"""
Apply PyTorch Delphes Efficiency module to ROOT file and save output.

This script emulates delphes_card_CMS_2_0.tcl:
1. Reads particles from HZZ4l_1.root (output after ParticlePropagator)
2. Applies ChargedHadronTrackingEfficiency module using PyTorch
3. Writes output ROOT file with the same structure as delphes_card_CMS_2_0.tcl

Usage:
    python test_torch_delphes.py [input.root] [output.root]
    
Default input: delphes_data/HZZ4l/HZZ4l_1.root
Default output: delphes_data/HZZ4l/HZZ4l_2_0_torch.root
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


def apply_charged_hadron_efficiency(charged_hadrons_arrays, branch_prefix):
    """
    Apply ChargedHadronTrackingEfficiency to charged hadron data.
    
    Args:
        charged_hadrons_arrays: awkward array with charged hadron data (Track objects)
        branch_prefix: prefix for accessing branch attributes
        
    Returns:
        List of filtered awkward arrays (one per event)
    """
    # Initialize efficiency module
    eff_module = DelphesEfficiency(
        efficiency_formula='charged_hadron_cms',
        deterministic=False,
        device='cpu'
    )
    
    n_events = len(charged_hadrons_arrays[f"{branch_prefix}.PT"])
    filtered_events = []
    
    print(f"Applying ChargedHadronTrackingEfficiency to {n_events} events...")
    
    for i in range(n_events):
        # Extract data for this event
        n_particles = len(charged_hadrons_arrays[f"{branch_prefix}.PT"][i])
        
        if n_particles == 0:
            # Empty event - create empty output
            filtered_events.append({})
            continue
        
        # Build tensor (N, 15)
        # Track objects don't have Status, E, Px, Py, Pz - we'll set these to dummy values
        # The efficiency module only uses PT (column 7) and Eta (column 8)
        particles = np.zeros((n_particles, 15))
        
        # Column 0: PID
        particles[:, 0] = np.array(charged_hadrons_arrays[f"{branch_prefix}.PID"][i])
        # Column 1: Status (not available for Track, set to 1)
        particles[:, 1] = 1
        # Column 2: Charge
        particles[:, 2] = np.array(charged_hadrons_arrays[f"{branch_prefix}.Charge"][i])
        # Column 3: E (not available for Track, will compute from P)
        P = np.array(charged_hadrons_arrays[f"{branch_prefix}.P"][i])
        particles[:, 3] = P  # Approximate E ≈ P for high-energy particles
        # Columns 4-6: Px, Py, Pz (not available, compute from P, PT, Eta, Phi)
        PT = np.array(charged_hadrons_arrays[f"{branch_prefix}.PT"][i])
        Eta = np.array(charged_hadrons_arrays[f"{branch_prefix}.Eta"][i])
        Phi = np.array(charged_hadrons_arrays[f"{branch_prefix}.Phi"][i])
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
        particles[:, 10] = np.array(charged_hadrons_arrays[f"{branch_prefix}.T"][i])
        # Columns 11-13: X, Y, Z
        particles[:, 11] = np.array(charged_hadrons_arrays[f"{branch_prefix}.X"][i])
        particles[:, 12] = np.array(charged_hadrons_arrays[f"{branch_prefix}.Y"][i])
        particles[:, 13] = np.array(charged_hadrons_arrays[f"{branch_prefix}.Z"][i])
        
        # Convert to torch tensor
        particles_tensor = torch.from_numpy(particles).float()
        
        # Apply efficiency
        filtered_tensor, mask = eff_module(particles_tensor, return_mask=True)
        mask_np = mask.cpu().numpy()
        
        # Extract filtered data for each attribute
        filtered_event = {}
        branch_keys = [key for key in charged_hadrons_arrays.fields if key.startswith(branch_prefix)]
        for key in branch_keys:
            attr_name = key.split('.')[-1]  # Get attribute name (e.g., "PID", "PT", etc.)
            original_data = np.array(charged_hadrons_arrays[key][i])
            filtered_event[attr_name] = original_data[mask_np]
        
        filtered_events.append(filtered_event)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_events} events")
    
    return filtered_events


def write_output_root(output_file, input_tree, filtered_charged_hadrons, max_events=None):
    """
    Write output ROOT file with all branches matching delphes_card_CMS_2_0.tcl structure.
    
    Args:
        output_file: Path to output ROOT file
        input_tree: Input uproot TTree
        filtered_charged_hadrons: List of filtered charged hadron events
        max_events: Number of events to write
    """
    print(f"\nWriting output ROOT file: {output_file}")
    
    # Branch names from delphes_card_CMS_1.tcl
    branch_names = ['ParticleBeforeProp', 'ParticleAfterProp', 'ChargedHadron', 'Electron', 'Muon']
    
    output_data = {}
    
    # Read all existing branches from input
    for branch_name in branch_names:
        print(f"  Reading {branch_name}...")
        branch_data = read_branch_data(input_tree, branch_name, max_events)
        
        # Store all attributes for this branch
        for key in branch_data.fields:
            output_key = f"{branch_name}.{key.split('.')[-1]}"
            output_data[output_key] = branch_data[key]
    
    # Add filtered charged hadron efficiency branch
    print(f"  Adding ChargedHadronEfficiency branch...")
    
    # Get attribute names from first non-empty filtered event
    attr_names = None
    for event in filtered_charged_hadrons:
        if event:
            attr_names = list(event.keys())
            break
    
    # Build awkward arrays for filtered data
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
    
    print(f"✓ Output written successfully!")


def main(input_file, output_file, max_events=None):
    """Main processing function."""
    
    print("="*70)
    print("PyTorch Delphes ChargedHadronTrackingEfficiency Module")
    print("="*70)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Max events: {max_events if max_events else 'All'}")
    print()
    
    # Open input ROOT file
    input_root = uproot.open(input_file)
    tree = input_root["Delphes"]
    print(f"tree.keys(): {tree.keys()}")
    
    n_events = tree.num_entries if max_events is None else min(max_events, tree.num_entries)
    print(f"Total events in file: {tree.num_entries}")
    print(f"Processing: {n_events} events\n")
    
    # Read charged hadrons (output from ParticlePropagator)
    print("Reading ChargedHadron branch...")
    charged_hadrons = read_branch_data(tree, "ChargedHadron", max_events)
    
    # Determine actual branch prefix (e.g., "ChargedHadron/ChargedHadron")
    sample_key = list(charged_hadrons.fields)[0]
    actual_prefix = sample_key.rsplit('.', 1)[0]
    
    print(f"Found {len(charged_hadrons.fields)} attributes for charged hadrons")
    
    # Count input particles (use PT since Track objects don't have E)
    total_input = sum(len(charged_hadrons[f"{actual_prefix}.PT"][i]) 
                      for i in range(n_events))
    print(f"Total input charged hadrons: {total_input}\n")
    
    # Apply efficiency module
    filtered_charged_hadrons = apply_charged_hadron_efficiency(
        charged_hadrons, 
        actual_prefix
    )
    
    # Count output particles and compute statistics (use PT)
    total_output = sum(len(event.get('PT', [])) for event in filtered_charged_hadrons)
    
    print(f"\nStatistics:")
    print(f"  Total input charged hadrons: {total_input}")
    print(f"  Total output (after efficiency): {total_output}")
    print(f"  Overall efficiency: {total_output/total_input*100:.2f}%")
    
    # Write output ROOT file
    write_output_root(output_file, tree, filtered_charged_hadrons, max_events)
    
    print(f"\n{'='*70}")
    print("Processing Complete!")
    print(f"{'='*70}")
    print(f"\nOutput file created: {output_file}")
    print("\nThis output file has the same structure as the one created by")
    print("running Delphes with delphes_card_CMS_2_0.tcl, with branches:")
    print("  - ParticleBeforeProp (before ParticlePropagator)")
    print("  - ParticleAfterProp (after ParticlePropagator)")
    print("  - ChargedHadron (separated charged hadrons)")
    print("  - Electron (separated electrons)")
    print("  - Muon (separated muons)")
    print("  - ChargedHadronEfficiency (after PyTorch efficiency filter)")


if __name__ == "__main__":
    # Set default paths relative to this file
    script_dir = Path(__file__).parent
    default_input = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_1.root"
    default_output = script_dir / "delphes_data" / "HZZ4l" / "HZZ4l_2_0_torch.root"
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments - use defaults
        input_file = str(default_input)
        output_file = str(default_output)
        max_events = None
    elif len(sys.argv) >= 3:
        # Input and output specified
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        max_events = int(sys.argv[3]) if len(sys.argv) > 3 else None
    else:
        print("Usage: python test_torch_delphes.py [input.root] [output.root] [max_events]")
        print("\nRun with no arguments to use defaults:")
        print(f"  Input:  {default_input}")
        print(f"  Output: {default_output}")
        print("\nOr specify input and output files:")
        print("  python test_torch_delphes.py input.root output.root")
        print("  python test_torch_delphes.py input.root output.root 100")
        sys.exit(1)
    
    main(input_file, output_file, max_events)
