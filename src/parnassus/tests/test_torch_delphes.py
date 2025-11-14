"""
Test script for delphes_efficiency_pytorch.py using Delphes ROOT output.

This script:
1. Reads particles from a Delphes ROOT file (output.root)
2. Converts them to the PyTorch format
3. Applies the efficiency module
4. Compares results and statistics
5. Creates validation plots

Usage:
    python test_efficiency_on_root.py output.root
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import uproot
except ImportError:
    print("ERROR: uproot not installed. Install with: pip install uproot awkward")
    sys.exit(1)

from  parnassus.torch_delphes.Efficiency import DelphesEfficiency


def read_delphes_root(filename, max_events=None):
    """
    Read particle data from Delphes ROOT file.
    
    Args:
        filename: Path to ROOT file
        max_events: Maximum number of events to read (None = all)
        
    Returns:
        Dictionary with particle arrays per event
    """
    print(f"Opening ROOT file: {filename}")
    
    try:
        file = uproot.open(filename)
        tree = file["Delphes"]
    except Exception as e:
        print(f"ERROR: Could not open ROOT file: {e}")
        sys.exit(1)
    
    # Get available branches
    branches = tree.keys()
    print(f"\nAvailable branches: {', '.join(branches)}")
    
    # Determine which particle branch to use
    particle_branches = ['ParticleBeforeProp', 'ParticleAfterProp', 'Particle']
    particle_branch = None
    
    for branch in particle_branches:
        if branch in branches:
            particle_branch = branch
            print(f"\nUsing branch: {particle_branch}")
            break
    
    if particle_branch is None:
        print("ERROR: No particle branch found in ROOT file!")
        print(f"Expected one of: {particle_branches}")
        sys.exit(1)
    
    # Read particle data
    print(f"Reading events (max={max_events})...")
    
    # Read relevant branches
    arrays = tree.arrays([
        f"{particle_branch}.PID",
        f"{particle_branch}.Status", 
        f"{particle_branch}.Charge",
        f"{particle_branch}.E",
        f"{particle_branch}.Px",
        f"{particle_branch}.Py",
        f"{particle_branch}.Pz",
        f"{particle_branch}.PT",
        f"{particle_branch}.Eta",
        f"{particle_branch}.Phi",
        f"{particle_branch}.X",
        f"{particle_branch}.Y",
        f"{particle_branch}.Z",
        f"{particle_branch}.T",
    ], entry_stop=max_events, library="ak")
    
    n_events = len(arrays)
    print(f"Read {n_events} events")
    
    # Convert to event-wise format
    events = []
    for i in range(n_events):
        event_data = {
            'pid': np.array(arrays[f"{particle_branch}.PID"][i]),
            'status': np.array(arrays[f"{particle_branch}.Status"][i]),
            'charge': np.array(arrays[f"{particle_branch}.Charge"][i]),
            'E': np.array(arrays[f"{particle_branch}.E"][i]),
            'Px': np.array(arrays[f"{particle_branch}.Px"][i]),
            'Py': np.array(arrays[f"{particle_branch}.Py"][i]),
            'Pz': np.array(arrays[f"{particle_branch}.Pz"][i]),
            'PT': np.array(arrays[f"{particle_branch}.PT"][i]),
            'Eta': np.array(arrays[f"{particle_branch}.Eta"][i]),
            'Phi': np.array(arrays[f"{particle_branch}.Phi"][i]),
            'X': np.array(arrays[f"{particle_branch}.X"][i]),
            'Y': np.array(arrays[f"{particle_branch}.Y"][i]),
            'Z': np.array(arrays[f"{particle_branch}.Z"][i]),
            'T': np.array(arrays[f"{particle_branch}.T"][i]),
        }
        events.append(event_data)
    
    return events, particle_branch


def delphes_to_pytorch_format(event_data):
    """
    Convert Delphes event data to PyTorch format (N, 15).
    
    Args:
        event_data: Dictionary with particle arrays
        
    Returns:
        torch.Tensor of shape (N, 15) with columns:
        [PID, Status, Charge, E, Px, Py, Pz, PT, Eta, Phi, T, X, Y, Z]
        
    """
    n_particles = len(event_data['E'])
    
    particles = np.zeros((n_particles, 15))
    
    # Column 0: PID
    particles[:, 0] = event_data['pid']
    
    # Column 1: Status
    particles[:, 1] = event_data['status']
    
    # Column 2: Charge
    particles[:, 2] = event_data['charge']
    
    # Column 3: E (Energy in GeV)
    particles[:, 3] = event_data['E']
    
    # Columns 4-6: Momentum (Px, Py, Pz) in GeV
    particles[:, 4] = event_data['Px']
    particles[:, 5] = event_data['Py']
    particles[:, 6] = event_data['Pz']
    
    # Column 7: PT (transverse momentum, pre-computed by Delphes)
    particles[:, 7] = event_data['PT']
    
    # Column 8: Eta (pseudorapidity, pre-computed by Delphes)
    particles[:, 8] = event_data['Eta']
    
    # Column 9: Phi (azimuthal angle, pre-computed by Delphes)
    particles[:, 9] = event_data['Phi']
    
    # Column 10: T (time in mm/c)
    particles[:, 10] = event_data['T']
    
    # Columns 11-13: Position (X, Y, Z) in mm
    particles[:, 11] = event_data['X']
    particles[:, 12] = event_data['Y']
    particles[:, 13] = event_data['Z']
    
    return torch.from_numpy(particles).float()


def categorize_particles(event_data):
    """
    Categorize particles by type (for testing different efficiencies).
    
    Returns:
        Dictionary with boolean masks for different particle types
    """
    pid = event_data['pid']
    charge = event_data['charge']
    
    return {
        'all': np.ones(len(pid), dtype=bool),
        'charged': np.abs(charge) > 0,
        'neutral': np.abs(charge) < 1e-6,
        'electron': np.abs(pid) == 11,
        'muon': np.abs(pid) == 13,
        'charged_hadron': (np.abs(charge) > 0) & ~np.isin(np.abs(pid), [11, 13]),
        'photon': np.abs(pid) == 22,
    }


def analyze_event(event_data, efficiency_module):
    """
    Analyze a single event with the efficiency module.
    
    Returns:
        Dictionary with analysis results
    """
    particles = delphes_to_pytorch_format(event_data)
    
    # Apply efficiency
    filtered, mask = efficiency_module(particles, return_mask=True)
    
    # Compute statistics
    results = {
        'n_input': len(particles),
        'n_output': mask.sum().item(),
        'efficiency': mask.float().mean().item(),
        'mask': mask.cpu().numpy(),
        'pt_input': event_data['PT'],
        'eta_input': event_data['Eta'],
        'pt_output': event_data['PT'][mask.cpu().numpy()],
        'eta_output': event_data['Eta'][mask.cpu().numpy()],
    }
    
    return results


def plot_efficiency_validation(events, results_list, particle_type, output_dir='efficiency_validation'):
    """
    Create validation plots comparing input and output distributions.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Collect all particles across events
    all_pt_input = np.concatenate([r['pt_input'] for r in results_list])
    all_eta_input = np.concatenate([r['eta_input'] for r in results_list])
    all_pt_output = np.concatenate([r['pt_output'] for r in results_list])
    all_eta_output = np.concatenate([r['eta_output'] for r in results_list])
    
    # Collect efficiencies per event
    efficiencies = [r['efficiency'] for r in results_list]
    n_input = [r['n_input'] for r in results_list]
    n_output = [r['n_output'] for r in results_list]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: pT distribution
    ax1 = plt.subplot(2, 3, 1)
    bins = np.logspace(-1, 2, 50)
    ax1.hist(all_pt_input, bins=bins, alpha=0.5, label='Input', color='blue', density=True)
    ax1.hist(all_pt_output, bins=bins, alpha=0.5, label='Output (after efficiency)', color='red', density=True)
    ax1.set_xlabel('pT (GeV)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_xscale('log')
    ax1.set_title(f'pT Distribution - {particle_type}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: η distribution
    ax2 = plt.subplot(2, 3, 2)
    bins_eta = np.linspace(-3, 3, 50)
    ax2.hist(all_eta_input, bins=bins_eta, alpha=0.5, label='Input', color='blue', density=True)
    ax2.hist(all_eta_output, bins=bins_eta, alpha=0.5, label='Output (after efficiency)', color='red', density=True)
    ax2.set_xlabel('η (pseudorapidity)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'η Distribution - {particle_type}', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=-1.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=-2.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: 2D pT vs η (input)
    ax3 = plt.subplot(2, 3, 3)
    h, xedges, yedges = np.histogram2d(all_eta_input, all_pt_input, 
                                       bins=[50, 50], 
                                       range=[[-3, 3], [0, 50]])
    ax3.pcolormesh(xedges, yedges, h.T, cmap='Blues')
    ax3.set_xlabel('η', fontsize=12)
    ax3.set_ylabel('pT (GeV)', fontsize=12)
    ax3.set_title('Input Particles (2D)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency per event
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(efficiencies, 'o-', alpha=0.6)
    ax4.axhline(y=np.mean(efficiencies), color='r', linestyle='--', 
                label=f'Mean: {np.mean(efficiencies):.3f}')
    ax4.set_xlabel('Event Number', fontsize=12)
    ax4.set_ylabel('Efficiency', fontsize=12)
    ax4.set_title('Per-Event Efficiency', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Plot 5: Particle count per event
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(n_input, label='Input', marker='o', alpha=0.6)
    ax5.plot(n_output, label='Output', marker='s', alpha=0.6)
    ax5.set_xlabel('Event Number', fontsize=12)
    ax5.set_ylabel('Number of Particles', fontsize=12)
    ax5.set_title('Particle Count per Event', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Efficiency histogram
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(efficiencies, bins=30, edgecolor='black', alpha=0.7)
    ax6.axvline(x=np.mean(efficiencies), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(efficiencies):.3f}±{np.std(efficiencies):.3f}')
    ax6.set_xlabel('Efficiency', fontsize=12)
    ax6.set_ylabel('Number of Events', fontsize=12)
    ax6.set_title('Efficiency Distribution', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"{output_dir}/validation_{particle_type.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def plot_kinematic_efficiency(events, results_list, particle_type, output_dir='efficiency_validation'):
    """
    Plot efficiency as a function of kinematics (measured from data).
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Collect all particles with their pass/fail status
    all_pt = []
    all_eta = []
    all_passed = []
    
    for event, result in zip(events, results_list):
        all_pt.extend(result['pt_input'])
        all_eta.extend(result['eta_input'])
        all_passed.extend(result['mask'])
    
    all_pt = np.array(all_pt)
    all_eta = np.array(all_eta)
    all_passed = np.array(all_passed)
    
    # Create efficiency vs pT (in η bins)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    eta_bins_ranges = [
        (0.0, 1.5, 'Central Barrel (|η| < 1.5)'),
        (1.5, 2.5, 'Forward Endcap (1.5 < |η| < 2.5)'),
    ]
    
    for idx, (eta_min, eta_max, label) in enumerate(eta_bins_ranges):
        ax = axes[idx, 0]
        
        # Select particles in this eta range
        mask = (np.abs(all_eta) >= eta_min) & (np.abs(all_eta) < eta_max)
        
        if mask.sum() > 0:
            pt_bins = np.logspace(-1, 2, 20)
            pt_centers = (pt_bins[:-1] + pt_bins[1:]) / 2
            
            # Compute efficiency in each pT bin
            eff_values = []
            eff_errors = []
            
            for i in range(len(pt_bins) - 1):
                bin_mask = mask & (all_pt >= pt_bins[i]) & (all_pt < pt_bins[i+1])
                if bin_mask.sum() > 10:  # Require at least 10 particles
                    eff = all_passed[bin_mask].mean()
                    eff_values.append(eff)
                    # Binomial error
                    eff_errors.append(np.sqrt(eff * (1 - eff) / bin_mask.sum()))
                else:
                    eff_values.append(np.nan)
                    eff_errors.append(np.nan)
            
            ax.errorbar(pt_centers, eff_values, yerr=eff_errors, fmt='o-', 
                       capsize=5, label='Measured', linewidth=2)
            ax.set_xlabel('pT (GeV)', fontsize=12)
            ax.set_ylabel('Efficiency', fontsize=12)
            ax.set_title(label, fontsize=13, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
    
    # Efficiency vs η (in pT bins)
    pt_bins_ranges = [
        (0.5, 1.0, 'Low pT (0.5-1 GeV)'),
        (5.0, 10.0, 'Medium pT (5-10 GeV)'),
    ]
    
    for idx, (pt_min, pt_max, label) in enumerate(pt_bins_ranges):
        ax = axes[idx, 1]
        
        # Select particles in this pT range
        mask = (all_pt >= pt_min) & (all_pt < pt_max)
        
        if mask.sum() > 0:
            eta_bins = np.linspace(-3, 3, 30)
            eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
            
            # Compute efficiency in each eta bin
            eff_values = []
            eff_errors = []
            
            for i in range(len(eta_bins) - 1):
                bin_mask = mask & (all_eta >= eta_bins[i]) & (all_eta < eta_bins[i+1])
                if bin_mask.sum() > 10:
                    eff = all_passed[bin_mask].mean()
                    eff_values.append(eff)
                    eff_errors.append(np.sqrt(eff * (1 - eff) / bin_mask.sum()))
                else:
                    eff_values.append(np.nan)
                    eff_errors.append(np.nan)
            
            ax.errorbar(eta_centers, eff_values, yerr=eff_errors, fmt='o-',
                       capsize=5, label='Measured', linewidth=2)
            ax.set_xlabel('η (pseudorapidity)', fontsize=12)
            ax.set_ylabel('Efficiency', fontsize=12)
            ax.set_title(label, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
            ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=-1.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=-2.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Measured Efficiency vs Kinematics - {particle_type}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f"{output_dir}/kinematic_eff_{particle_type.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()


def main(root_file, max_events=None):
    """Main test function."""
    
    print("="*70)
    print("Testing delphes_efficiency_pytorch.py on Delphes ROOT output")
    print("="*70)
    print()
    
    # Read ROOT file
    events, particle_branch = read_delphes_root(root_file, max_events)
    
    if len(events) == 0:
        print("ERROR: No events found in ROOT file!")
        return
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("Input Data Summary")
    print(f"{'='*70}")
    print(f"Number of events: {len(events)}")
    print(f"Particle branch: {particle_branch}")
    
    total_particles = sum(len(e['E']) for e in events)
    print(f"Total particles: {total_particles}")
    print(f"Avg particles/event: {total_particles/len(events):.1f}")
    
    # Categorize first event to show breakdown
    categories = categorize_particles(events[0])
    print(f"\nFirst event breakdown:")
    for cat_name, mask in categories.items():
        count = mask.sum()
        frac = count / len(mask) * 100 if len(mask) > 0 else 0
        print(f"  {cat_name:15s}: {count:4d} ({frac:5.1f}%)")
    
    # Test with different efficiency formulas
    formulas = [
        ('charged_hadron_cms', 'Charged Hadron'),
        ('electron_cms', 'Electron'),
        ('muon_cms', 'Muon'),
    ]
    
    print(f"\n{'='*70}")
    print("Testing Efficiency Modules")
    print(f"{'='*70}")
    
    for formula_name, display_name in formulas:
        print(f"\n{display_name} Efficiency:")
        print("-" * 50)
        
        # Create efficiency module
        eff_module = DelphesEfficiency(
            efficiency_formula=formula_name,
            deterministic=False,
            device='cpu'
        )
        
        # Process all events
        results_list = []
        for i, event in enumerate(events):
            result = analyze_event(event, eff_module)
            results_list.append(result)
            
            if i < 3:  # Print first 3 events
                print(f"  Event {i}: {result['n_input']:4d} → {result['n_output']:4d} "
                      f"({result['efficiency']*100:5.1f}%)")
        
        # Compute overall statistics
        total_input = sum(r['n_input'] for r in results_list)
        total_output = sum(r['n_output'] for r in results_list)
        overall_eff = total_output / total_input if total_input > 0 else 0
        
        print(f"\n  Overall: {total_input} → {total_output} ({overall_eff*100:.2f}%)")
        print(f"  Mean event efficiency: {np.mean([r['efficiency'] for r in results_list])*100:.2f}% "
              f"± {np.std([r['efficiency'] for r in results_list])*100:.2f}%")
        
        # Create plots
        print(f"\n  Creating validation plots...")
        plot_efficiency_validation(events, results_list, display_name)
        plot_kinematic_efficiency(events, results_list, display_name)
    
    print(f"\n{'='*70}")
    print("Testing Complete!")
    print(f"{'='*70}")
    print("\nValidation plots saved to: efficiency_validation/")
    print("\nFiles generated:")
    print("  - validation_*.png : Distribution comparisons")
    print("  - kinematic_eff_*.png : Measured efficiency vs pT and η")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_efficiency_on_root.py <output.root> [max_events]")
        print("\nExample:")
        print("  python test_efficiency_on_root.py output.root")
        print("  python test_efficiency_on_root.py output.root 100")
        sys.exit(1)
    
    root_file = sys.argv[1]
    max_events = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not Path(root_file).exists():
        print(f"ERROR: File not found: {root_file}")
        sys.exit(1)
    
    main(root_file, max_events)
