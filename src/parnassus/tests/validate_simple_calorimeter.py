
from typing import List, Dict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def get_ylim(ratio):
    ratio = ratio[~np.isnan(ratio)]
    ratio = ratio[~np.isinf(ratio)]
    return [0.9*min(ratio), 1.1*max(ratio)]

def validate_simple_cal_step_1(
    ecal_results: Dict,
    cpp_fractions_file: str,
    output_dir: str,
) -> None:
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
    
def validate_simple_cal_step_2(
    ecal_results: Dict,
    cpp_towerhits_file: str,
    output_dir: str,
) -> None:

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
            ax_eta_ratio.set_ylabel('Torch/C++', fontsize=10)
            ax_eta_ratio.set_ylim(get_ylim(eta_ratio))
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
            ax_phi_ratio.set_ylabel('Torch/C++', fontsize=10)
            ax_phi_ratio.set_ylim(get_ylim(phi_ratio))
            ax_phi_ratio.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / f"{hit_type.lower()}_bins.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {plot_file}")
        
        print(f"  ✓ Step 2 validation complete.")

def validate_simple_cal_step_3(
    ecal_results: Dict,
    cpp_towerenergy_file: str,
    output_dir: str,
) -> None:
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
        ax1_ratio.set_ylim(get_ylim(ratio))
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
            ax2_ratio.set_ylim(get_ylim(track_ratio))
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
        ax4_ratio.set_ylim(get_ylim(towers_ratio))
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

    return tower_results, cpp_towers_df
 
def validate_simple_cal_step_4(
    tower_results: List[Dict],
    cpp_towers_df: pd.DataFrame,
    output_dir: str,
) -> None:
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
    ax_eta_ratio.set_ylim(get_ylim(eta_ratio[valid_eta_ratio]))
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
    ax_phi_ratio.set_ylim(get_ylim(phi_ratio[valid_phi_ratio]))
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

def validate_simple_cal_step_5(
    tower_results: List[Dict],
    cpp_smearing_file: str,
    output_dir: str,
) -> None:
    # Compare against C++ debug output (using law of large numbers - distributions should match)
    print(f"\n  Step 6: Validating Resolution Smearing...")
    
    # Load C++ smearing debug output
    if not Path(cpp_smearing_file).exists():
        print(f"  ⚠ C++ smearing debug file not found: {cpp_smearing_file}")
        print(f"  ⚠ Skipping Step 6 validation. Please recompile and re-run C++ Delphes.")
        print(f"\n  ✓ All SimpleCalorimeter validation complete. Plots saved to {output_dir}")
        return
    
    cpp_smearing_df = pd.read_csv(cpp_smearing_file)
    print(f"  Loaded {len(cpp_smearing_df)} C++ smearing records from {cpp_smearing_file}")
    
    # Collect TorchDelphes smearing results
    torch_energy_before = np.concatenate([r['tower_energy'].numpy() for r in tower_results])
    torch_energy_smeared = np.concatenate([r['tower_energy_smeared'].numpy() for r in tower_results])
    torch_energy_final = np.concatenate([r['tower_energy_final'].numpy() for r in tower_results])
    torch_sigma_before = np.concatenate([r['sigma_before'].numpy() for r in tower_results])
    torch_sigma_after = np.concatenate([r['sigma_after'].numpy() for r in tower_results])
    torch_eta = np.concatenate([r['tower_eta'].numpy() for r in tower_results])
    
    # C++ smearing data
    cpp_energy_before = cpp_smearing_df['energy_before'].values
    cpp_sigma_before = cpp_smearing_df['sigma_before'].values
    cpp_energy_smeared = cpp_smearing_df['energy_smeared'].values
    cpp_sigma_after = cpp_smearing_df['sigma_after'].values
    cpp_energy_final = cpp_smearing_df['energy_final'].values
    cpp_eta = cpp_smearing_df['tower_eta'].values
    
    # Print summary statistics
    print(f"\n  Smearing Statistics:")
    print(f"    TorchDelphes: {len(torch_energy_before)} towers, C++: {len(cpp_energy_before)} towers")
    
    # Create validation plots
    fig = plt.figure(figsize=(18, 12))
    
    # ===== Row 1: Energy before smearing (sigma) comparison =====
    # Plot 1: Sigma before comparison with ratio
    ax1_main = fig.add_axes([0.05, 0.72, 0.28, 0.22])
    ax1_ratio = fig.add_axes([0.05, 0.58, 0.28, 0.10])
    
    sigma_bins = np.linspace(0, 5, 100)
    cpp_sigma_counts, _ = np.histogram(cpp_sigma_before, bins=sigma_bins)
    torch_sigma_counts, _ = np.histogram(torch_sigma_before, bins=sigma_bins)
    
    ax1_main.hist(cpp_sigma_before, bins=sigma_bins, histtype='stepfilled', color='orange', alpha=0.5,
                  label=f'C++ Delphes ({len(cpp_sigma_before)})')
    ax1_main.hist(torch_sigma_before, bins=sigma_bins, histtype='step', color='blue', linewidth=2,
                  label=f'TorchDelphes ({len(torch_sigma_before)})')
    ax1_main.set_ylabel('Counts', fontsize=12)
    ax1_main.set_title('σ Before Smearing', fontsize=14)
    ax1_main.legend(fontsize=9)
    ax1_main.grid(True, alpha=0.3)
    ax1_main.set_xticklabels([])
    
    sigma_bin_centers = 0.5 * (sigma_bins[:-1] + sigma_bins[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_ratio = np.where(cpp_sigma_counts > 0, torch_sigma_counts / cpp_sigma_counts, np.nan)
    valid_sigma_ratio = ~np.isnan(sigma_ratio)
    ax1_ratio.scatter(sigma_bin_centers[valid_sigma_ratio], sigma_ratio[valid_sigma_ratio], s=10, c='purple', alpha=0.7)
    ax1_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
    ax1_ratio.set_xlabel('σ (GeV)', fontsize=12)
    ax1_ratio.set_ylabel('Torch/C++', fontsize=10)
    ax1_ratio.set_ylim(get_ylim(sigma_ratio[valid_sigma_ratio]))
    ax1_ratio.grid(True, alpha=0.3)
    
    # Plot 2: Energy smeared comparison with ratio
    ax2_main = fig.add_axes([0.38, 0.72, 0.28, 0.22])
    ax2_ratio = fig.add_axes([0.38, 0.58, 0.28, 0.10])
    
    # Filter non-zero
    torch_smeared_nz = torch_energy_smeared[torch_energy_smeared > 0]
    cpp_smeared_nz = cpp_energy_smeared[cpp_energy_smeared > 0]
    
    e_min = max(min(torch_smeared_nz.min() if len(torch_smeared_nz) > 0 else 1e-3,
                   cpp_smeared_nz.min() if len(cpp_smeared_nz) > 0 else 1e-3), 1e-3)
    e_max = max(torch_smeared_nz.max() if len(torch_smeared_nz) > 0 else 100,
               cpp_smeared_nz.max() if len(cpp_smeared_nz) > 0 else 100)
    energy_bins = np.logspace(np.log10(e_min), np.log10(e_max * 1.1), 50)
    
    cpp_e_counts, _ = np.histogram(cpp_smeared_nz, bins=energy_bins)
    torch_e_counts, _ = np.histogram(torch_smeared_nz, bins=energy_bins)
    
    ax2_main.hist(cpp_smeared_nz, bins=energy_bins, histtype='stepfilled', color='orange', alpha=0.5,
                  label=f'C++ ({len(cpp_smeared_nz)})')
    ax2_main.hist(torch_smeared_nz, bins=energy_bins, histtype='step', color='blue', linewidth=2,
                  label=f'Torch ({len(torch_smeared_nz)})')
    ax2_main.set_xscale('log')
    ax2_main.set_ylabel('Counts', fontsize=12)
    ax2_main.set_title('Energy After Smearing', fontsize=14)
    ax2_main.legend(fontsize=9)
    ax2_main.grid(True, alpha=0.3)
    ax2_main.set_xticklabels([])
    
    e_bin_centers = np.sqrt(energy_bins[:-1] * energy_bins[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        e_ratio = np.where(cpp_e_counts > 0, torch_e_counts / cpp_e_counts, np.nan)
    valid_e_ratio = ~np.isnan(e_ratio)
    ax2_ratio.scatter(e_bin_centers[valid_e_ratio], e_ratio[valid_e_ratio], s=10, c='purple', alpha=0.7)
    ax2_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
    ax2_ratio.set_xscale('log')
    ax2_ratio.set_xlabel('Energy (GeV)', fontsize=12)
    ax2_ratio.set_ylabel('Torch/C++', fontsize=10)
    ax2_ratio.set_ylim(get_ylim(e_ratio[valid_e_ratio]))
    ax2_ratio.grid(True, alpha=0.3)
    
    # Plot 3: Energy final (after threshold) comparison with ratio
    ax3_main = fig.add_axes([0.71, 0.72, 0.26, 0.22])
    ax3_ratio = fig.add_axes([0.71, 0.58, 0.26, 0.10])
    
    torch_final_nz = torch_energy_final[torch_energy_final > 0]
    cpp_final_nz = cpp_energy_final[cpp_energy_final > 0]
    
    cpp_f_counts, _ = np.histogram(cpp_final_nz, bins=energy_bins)
    torch_f_counts, _ = np.histogram(torch_final_nz, bins=energy_bins)
    
    ax3_main.hist(cpp_final_nz, bins=energy_bins, histtype='stepfilled', color='orange', alpha=0.5,
                  label=f'C++ ({len(cpp_final_nz)})')
    ax3_main.hist(torch_final_nz, bins=energy_bins, histtype='step', color='blue', linewidth=2,
                  label=f'Torch ({len(torch_final_nz)})')
    ax3_main.set_xscale('log')
    ax3_main.set_ylabel('Counts', fontsize=12)
    ax3_main.set_title('Energy After Threshold', fontsize=14)
    ax3_main.legend(fontsize=9)
    ax3_main.grid(True, alpha=0.3)
    ax3_main.set_xticklabels([])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        f_ratio = np.where(cpp_f_counts > 0, torch_f_counts / cpp_f_counts, np.nan)
    valid_f_ratio = ~np.isnan(f_ratio)
    ax3_ratio.scatter(e_bin_centers[valid_f_ratio], f_ratio[valid_f_ratio], s=10, c='purple', alpha=0.7)
    ax3_ratio.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
    ax3_ratio.set_xscale('log')
    ax3_ratio.set_xlabel('Energy (GeV)', fontsize=12)
    ax3_ratio.set_ylabel('Torch/C++', fontsize=10)
    ax3_ratio.set_ylim(get_ylim(f_ratio[valid_f_ratio]))
    ax3_ratio.grid(True, alpha=0.3)
    
    # ===== Row 2: Resolution and threshold comparison =====
    # Plot 4: Resolution (sigma/E) vs eta scatter
    ax4 = fig.add_axes([0.05, 0.08, 0.28, 0.40])
    
    cpp_mask = cpp_energy_before > 0
    torch_mask = torch_energy_before > 0
    
    cpp_resolution = cpp_sigma_before[cpp_mask] / cpp_energy_before[cpp_mask]
    torch_resolution = torch_sigma_before[torch_mask] / torch_energy_before[torch_mask]
    
    ax4.scatter(cpp_eta[cpp_mask], cpp_resolution, s=3, alpha=0.2, c='orange', label='C++')
    ax4.scatter(torch_eta[torch_mask], torch_resolution, s=3, alpha=0.2, c='blue', label='Torch')
    ax4.set_xlabel('Tower Eta', fontsize=12)
    ax4.set_ylabel('σ/E (Resolution)', fontsize=12)
    ax4.set_title('Energy Resolution vs Eta', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 2)
    
    # Plot 5: Threshold effects comparison
    ax5 = fig.add_axes([0.38, 0.08, 0.28, 0.40])
    
    cpp_n_smeared = np.sum(cpp_energy_smeared > 0)
    cpp_n_final = np.sum(cpp_energy_final > 0)
    cpp_n_zeroed = cpp_n_smeared - cpp_n_final
    
    torch_n_smeared = np.sum(torch_energy_smeared > 0)
    torch_n_final = np.sum(torch_energy_final > 0)
    torch_n_zeroed = torch_n_smeared - torch_n_final
    
    x = np.arange(3)
    width = 0.35
    
    ax5.bar(x - width/2, [cpp_n_smeared, cpp_n_final, cpp_n_zeroed], width, 
            label='C++ Delphes', color='orange', alpha=0.7)
    ax5.bar(x + width/2, [torch_n_smeared, torch_n_final, torch_n_zeroed], width,
            label='TorchDelphes', color='blue', alpha=0.7)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(['After\nSmearing', 'After\nThreshold', 'Zeroed by\nThreshold'])
    ax5.set_ylabel('Number of Towers', fontsize=12)
    ax5.set_title('Threshold Effect Comparison', fontsize=14)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Summary statistics
    ax6 = fig.add_axes([0.71, 0.08, 0.26, 0.40])
    ax6.axis('off')
    ax6.set_title('Smearing Summary', fontsize=14)
    
    summary_text = f"""
C++ Delphes:
  Towers total: {len(cpp_energy_before)}
  σ_before: mean={cpp_sigma_before.mean():.3f}
  E_smeared>0: {cpp_n_smeared} ({100*cpp_n_smeared/len(cpp_energy_before):.1f}%)
  E_final>0: {cpp_n_final} ({100*cpp_n_final/len(cpp_energy_before):.1f}%)

TorchDelphes:
  Towers total: {len(torch_energy_before)}
  σ_before: mean={torch_sigma_before.mean():.3f}
  E_smeared>0: {torch_n_smeared} ({100*torch_n_smeared/len(torch_energy_before):.1f}%)
  E_final>0: {torch_n_final} ({100*torch_n_final/len(torch_energy_before):.1f}%)

Ratios (Torch/C++):
  Total towers: {len(torch_energy_before)/len(cpp_energy_before):.3f}
  σ mean: {torch_sigma_before.mean()/cpp_sigma_before.mean():.3f}
  Final count: {torch_n_final/cpp_n_final:.3f}
"""
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plot_file = output_dir / "resolution_smearing.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {plot_file}")
    
    print(f"\n  Threshold Statistics:")
    print(f"    C++: {cpp_n_smeared} → {cpp_n_final} towers ({cpp_n_zeroed} zeroed)")
    print(f"    Torch: {torch_n_smeared} → {torch_n_final} towers ({torch_n_zeroed} zeroed)")
    
    print(f"  ✓ Step 6 validation complete.")

def validate_simple_cal_step_6(
    tower_results: List[Dict],
    cpp_pertrack_file: str,
    cpp_tracksigma_file: str,
    output_dir: str,
    debug: bool = False,
) -> None:
    
    # Compare per-track sigma contributions against C++ debug output
    print(f"\n  Step 7: Validating Track Sigma per Tower...")
    
    # Load C++ per-track sigma debug output
    cpp_pertrack_file = Path(cpp_pertrack_file)
    if not Path(cpp_pertrack_file).exists():
        print(f"  ⚠ C++ per-track debug file not found: {cpp_pertrack_file}")
        print(f"  ⚠ Skipping Step 7 validation. Please recompile and re-run C++ Delphes.")
        print(f"\n  ✓ All SimpleCalorimeter validation complete. Plots saved to {output_dir}")
        return
    
    cpp_pertrack_df = pd.read_csv(cpp_pertrack_file)
    print(f"  Loaded {len(cpp_pertrack_df)} C++ per-track records from {cpp_pertrack_file}")
    
    # Collect TorchDelphes track sigma results
    # We need to gather per-track data from results
    torch_track_energy = []
    torch_track_resolution = []
    torch_track_tower_eta = []
    torch_track_calo_sigma = []
    torch_track_energy_guess = []
    torch_track_sigma_sq = []
    
    for r in tower_results:
        track_sigma_valid = r['track_sigma_valid'].numpy()
        if track_sigma_valid.any():
            torch_track_energy.append(r['track_energy'][track_sigma_valid].numpy())
            torch_track_resolution.append(r['track_momentum_resolution'][track_sigma_valid].numpy())
            torch_track_tower_eta.append(r['track_tower_eta'][track_sigma_valid].numpy())
            torch_track_calo_sigma.append(r['track_calo_sigma'][track_sigma_valid].numpy())
            torch_track_energy_guess.append(r['track_energy_guess'][track_sigma_valid].numpy())
            torch_track_sigma_sq.append(r['track_sigma_sq'][track_sigma_valid].numpy())
    
    if len(torch_track_energy) > 0:
        torch_track_energy = np.concatenate(torch_track_energy)
        torch_track_resolution = np.concatenate(torch_track_resolution)
        torch_track_tower_eta = np.concatenate(torch_track_tower_eta)
        torch_track_calo_sigma = np.concatenate(torch_track_calo_sigma)
        torch_track_energy_guess = np.concatenate(torch_track_energy_guess)
        torch_track_sigma_sq = np.concatenate(torch_track_sigma_sq)
    else:
        torch_track_energy = np.array([])
        torch_track_resolution = np.array([])
        torch_track_tower_eta = np.array([])
        torch_track_calo_sigma = np.array([])
        torch_track_energy_guess = np.array([])
        torch_track_sigma_sq = np.array([])
    
    # C++ per-track data
    cpp_track_energy = cpp_pertrack_df['track_energy'].values
    cpp_track_resolution = cpp_pertrack_df['track_resolution'].values
    cpp_track_tower_eta = cpp_pertrack_df['tower_eta'].values
    cpp_track_calo_sigma = cpp_pertrack_df['calo_sigma'].values
    cpp_track_energy_guess = cpp_pertrack_df['energy_guess'].values
    cpp_track_sigma_sq = cpp_pertrack_df['sigma_sq_contrib'].values
    
    print(f"\n  Per-Track Sigma Statistics:")
    print(f"    TorchDelphes: {len(torch_track_energy)} valid tracks")
    print(f"    C++ Delphes:  {len(cpp_track_energy)} valid tracks")
    
    # Create validation plots
    fig = plt.figure(figsize=(20, 28))
    
    # ===== Row 1: Track energy and resolution comparison =====
    # Plot 1: Track energy distribution with ratio
    ax1 = fig.add_subplot(4, 2, 1)
    ax1_ratio = ax1.inset_axes([0, -0.35, 1, 0.3])
    bins = np.linspace(0, min(cpp_track_energy.max(), 200), 50)
    _cpp_track_energy_counts, _track_energy_bins, _ = ax1.hist(cpp_track_energy, bins=bins, histtype='stepfilled', label=f'C++ Delphes; {len(cpp_track_energy)} tracks', alpha=0.6, color='orange')
    _torch_track_energy_counts, _, _ = ax1.hist(torch_track_energy, bins=_track_energy_bins, histtype='step', label=f'TorchDelphes; {len(torch_track_energy)} tracks', alpha=0.9, color='blue', linewidth=1.5)
    ax1.set_xlabel('Track Energy [GeV]')
    ax1.set_ylabel('Count')
    ax1.set_title('Track Energy Distribution')
    ax1.legend()
    ax1.set_yscale('log')
    # Ratio
    cpp_hist, bin_edges = np.histogram(cpp_track_energy, bins=bins)
    torch_hist, _ = np.histogram(torch_track_energy, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cpp_hist > 0, torch_hist / cpp_hist, 1.0)
    ax1_ratio.step(bin_centers, ratio, where='mid', color='black')
    ax1_ratio.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax1_ratio.set_ylim(get_ylim(ratio))
    ax1_ratio.set_xlabel('Track Energy [GeV]')
    ax1_ratio.set_ylabel('Torch/C++')
    
    # Plot 2: Track momentum resolution distribution with ratio
    ax2 = fig.add_subplot(4, 2, 2)
    ax2_ratio = ax2.inset_axes([0, -0.35, 1, 0.3])
    bins = np.linspace(0, 0.1, 50)
    _cpp_track_res_counts, _track_res_bins, _ = ax2.hist(cpp_track_resolution, bins=bins, histtype='stepfilled', label=f'C++ Delphes; {len(cpp_track_resolution)} tracks', alpha=0.6, color='orange')
    _torch_track_res_counts, _, _ = ax2.hist(torch_track_resolution, bins=_track_res_bins, histtype='step', label=f'TorchDelphes; {len(torch_track_resolution)} tracks', alpha=0.9, color='blue', linewidth=1.5)
    ax2.set_xlabel('Track Resolution (σ/pT)')
    ax2.set_ylabel('Count')
    ax2.set_title('Track Momentum Resolution Distribution')
    ax2.legend()
    # Ratio
    cpp_hist, bin_edges = np.histogram(cpp_track_resolution, bins=bins)
    torch_hist, _ = np.histogram(torch_track_resolution, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cpp_hist > 0, torch_hist / cpp_hist, 1.0)
    ax2_ratio.step(bin_centers, ratio, where='mid', color='black')
    ax2_ratio.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax2_ratio.set_ylim(get_ylim(ratio))
    ax2_ratio.set_xlabel('Track Resolution (σ/pT)')
    ax2_ratio.set_ylabel('Torch/C++')
    
    # ===== Row 2: Calo sigma and tower eta =====
    # Plot 3: Calorimeter sigma with ratio
    ax3 = fig.add_subplot(4, 2, 3)
    ax3_ratio = ax3.inset_axes([0, -0.35, 1, 0.3])
    bins = np.linspace(0, min(cpp_track_calo_sigma.max(), 20), 50)
    _cpp_cal_sigma_counts, _cal_sigma_bins, _ = ax3.hist(cpp_track_calo_sigma, bins=bins, histtype='stepfilled', label=f'C++ Delphes; {len(cpp_track_calo_sigma)} tracks', alpha=0.6, color='orange')
    _torch_cal_sigma_counts, _, _ = ax3.hist(torch_track_calo_sigma, bins=_cal_sigma_bins, histtype='step', label=f'TorchDelphes; {len(torch_track_calo_sigma)} tracks', alpha=0.9, color='blue', linewidth=1.5)
    ax3.set_xlabel('Calorimeter Sigma [GeV]')
    ax3.set_ylabel('Count')
    ax3.set_title('Calorimeter Resolution at Tower η')
    ax3.legend()
    ax3.set_yscale('log')
    # Ratio
    cpp_hist, bin_edges = np.histogram(cpp_track_calo_sigma, bins=bins)
    torch_hist, _ = np.histogram(torch_track_calo_sigma, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cpp_hist > 0, torch_hist / cpp_hist, 1.0)
    ax3_ratio.step(bin_centers, ratio, where='mid', color='black')
    ax3_ratio.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax3_ratio.set_ylim(get_ylim(ratio))
    ax3_ratio.set_xlabel('Calorimeter Sigma [GeV]')
    ax3_ratio.set_ylabel('Torch/C++')
    
    # Plot 4: Tower eta for tracks with ratio
    ax4 = fig.add_subplot(4, 2, 4)
    ax4_ratio = ax4.inset_axes([0, -0.35, 1, 0.3])
    bins = np.linspace(-5, 5, 50)
    _cpp_tower_eta_counts, _tower_eta_bins, _ = ax4.hist(cpp_track_tower_eta, bins=bins, histtype='stepfilled', label=f'C++ Delphes; {len(cpp_track_tower_eta)} tracks', alpha=0.6, color='orange')
    _torch_tower_eta_counts, _, _ = ax4.hist(torch_track_tower_eta, bins=_tower_eta_bins, histtype='step', label=f'TorchDelphes; {len(torch_track_tower_eta)} tracks', alpha=0.9, color='blue', linewidth=1.5)
    ax4.set_xlabel('Tower η')
    ax4.set_ylabel('Count')
    ax4.set_title('Tower η for Valid Tracks')
    ax4.legend()
    # Ratio
    cpp_hist, bin_edges = np.histogram(cpp_track_tower_eta, bins=bins)
    torch_hist, _ = np.histogram(torch_track_tower_eta, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cpp_hist > 0, torch_hist / cpp_hist, 1.0)
    ax4_ratio.step(bin_centers, ratio, where='mid', color='black')
    ax4_ratio.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax4_ratio.set_ylim(get_ylim(ratio))
    ax4_ratio.set_xlabel('Tower η')
    ax4_ratio.set_ylabel('Torch/C++')
    
    # ===== Row 3: Energy guess comparison =====
    # Plot 5: Energy guess distribution with ratio
    ax5 = fig.add_subplot(4, 2, 5)
    ax5_ratio = ax5.inset_axes([0, -0.35, 1, 0.3])
    bins = np.linspace(0, min(cpp_track_energy_guess.max(), 200), 50)
    _cpp_track_e_guess_counts, _e_guess_bins, _ = ax5.hist(cpp_track_energy_guess, bins=bins, histtype='stepfilled', label=f'C++ Delphes; {len(cpp_track_energy_guess)} tracks', alpha=0.6, color='orange')
    _torch_track_e_guess_counts, _, _ = ax5.hist(torch_track_energy_guess, bins=_e_guess_bins, histtype='step', label=f'TorchDelphes; {len(torch_track_energy_guess)} tracks', alpha=0.9, color='blue', linewidth=1.5)
    ax5.set_xlabel('Energy Guess [GeV]')
    ax5.set_ylabel('Count')
    ax5.set_title('Track Energy Guess (based on resolution comparison)')
    ax5.legend()
    ax5.set_yscale('log')
    # Ratio
    cpp_hist, bin_edges = np.histogram(cpp_track_energy_guess, bins=bins)
    torch_hist, _ = np.histogram(torch_track_energy_guess, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cpp_hist > 0, torch_hist / cpp_hist, 1.0)
    ax5_ratio.step(bin_centers, ratio, where='mid', color='black')
    ax5_ratio.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax5_ratio.set_ylim(get_ylim(ratio))
    ax5_ratio.set_xlabel('Energy Guess [GeV]')
    ax5_ratio.set_ylabel('Torch/C++')
    
    # Plot 6: Sigma^2 contribution per track with ratio
    ax6 = fig.add_subplot(4, 2, 6)
    ax6_ratio = ax6.inset_axes([0, -0.35, 1, 0.3])
    bins = np.logspace(-2, 4, 50)
    _cpp_track_sigma_sq_counts, _sigma_sq_bins, _ = ax6.hist(cpp_track_sigma_sq, bins=bins, histtype='stepfilled', label=f'C++ Delphes; {len(cpp_track_sigma_sq)} tracks', alpha=0.6, color='orange')
    _torch_track_sigma_sq_counts, _, _ = ax6.hist(torch_track_sigma_sq, bins=_sigma_sq_bins, histtype='step', label=f'TorchDelphes; {len(torch_track_sigma_sq)} tracks', alpha=0.9, color='blue', linewidth=1.5)
    ax6.set_xlabel('σ² Contribution')
    ax6.set_ylabel('Count')
    ax6.set_title('Per-Track Sigma² Contribution')
    ax6.legend()
    ax6.set_xscale('log')
    ax6.set_yscale('log')
    # Ratio
    cpp_hist, bin_edges = np.histogram(cpp_track_sigma_sq, bins=bins)
    torch_hist, _ = np.histogram(torch_track_sigma_sq, bins=bins)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean for log bins
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(cpp_hist > 0, torch_hist / cpp_hist, 1.0)
    ax6_ratio.step(bin_centers, ratio, where='mid', color='black')
    ax6_ratio.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax6_ratio.set_ylim(get_ylim(ratio))
    ax6_ratio.set_xscale('log')
    ax6_ratio.set_xlabel('σ² Contribution')
    ax6_ratio.set_ylabel('Torch/C++')
    
    # ===== Row 4: Tower-level track sigma =====
    # Aggregate tower track sigma
    torch_tower_track_sigma = np.concatenate([r['tower_track_sigma'].numpy() for r in tower_results])
    
    # Load C++ tower-level track sigma debug output
    has_cpp_tracksigma = Path(cpp_tracksigma_file).exists()
    if has_cpp_tracksigma:
        cpp_tracksigma_df = pd.read_csv(cpp_tracksigma_file)
        cpp_tower_track_sigma = cpp_tracksigma_df['track_sigma'].values
        cpp_tower_track_energy_agg = cpp_tracksigma_df['track_energy'].values
        print(f"  Loaded {len(cpp_tracksigma_df)} C++ tower-level track sigma records")
    else:
        print(f"  ⚠ C++ tower-level track sigma file not found: {cpp_tracksigma_file}")
        cpp_tower_track_sigma = np.array([])
    
    # Plot 7: Tower track sigma distribution with ratio
    ax7 = fig.add_subplot(4, 2, 7)
    ax7_ratio = ax7.inset_axes([0, -0.35, 1, 0.3])
    
    bins = np.logspace(-2, 3, 50)
    torch_data = torch_tower_track_sigma[torch_tower_track_sigma > 0]
    
    if has_cpp_tracksigma and len(cpp_tower_track_sigma) > 0:
        cpp_data = cpp_tower_track_sigma[cpp_tower_track_sigma > 0]
        _cpp_tracksigma_counts, _tracksigma_bins, _ = ax7.hist(cpp_data, bins=bins, histtype='stepfilled', label=f'C++ Delphes; {len(cpp_data)} tracks', alpha=0.6, color='orange')
        _torch_tracksigma_counts, _, _ = ax7.hist(torch_data, bins=_tracksigma_bins, histtype='step', label=f'TorchDelphes; {len(torch_data)} tracks', alpha=0.9, color='blue', linewidth=1.5)
        # Ratio
        cpp_hist, bin_edges = np.histogram(cpp_data, bins=bins)
        torch_hist, _ = np.histogram(torch_data, bins=bins)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(cpp_hist > 0, torch_hist / cpp_hist, 1.0)
        ax7_ratio.step(bin_centers, ratio, where='mid', color='black')
        ax7_ratio.axhline(1.0, color='red', linestyle='--', alpha=0.5)
        ax7_ratio.set_ylim(get_ylim(ratio))
    else:
        ax7.hist(torch_data, bins=bins, histtype='step', label='TorchDelphes', alpha=0.9, color='blue', linewidth=1.5)
        ax7_ratio.text(0.5, 0.5, 'No C++ tower-level data', 
                       ha='center', va='center', transform=ax7_ratio.transAxes)
    
    ax7.set_xlabel('Tower Track Sigma [GeV]')
    ax7.set_ylabel('Count')
    ax7.set_title('Tower Track Sigma (√Σσ²)')
    ax7.legend()
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax7_ratio.set_xlim(bins[0], bins[-1])
    ax7_ratio.set_xscale('log')
    ax7_ratio.set_xlabel('Tower Track Sigma [GeV]')
    ax7_ratio.set_ylabel('Torch/C++')
    
    # Plot 8: Statistics text
    ax8 = fig.add_subplot(4, 2, 8)
    ax8.axis('off')
    
    # Compute statistics
    torch_n_tracks = len(torch_track_energy)
    cpp_n_tracks = len(cpp_track_energy)
    
    # How many use weighted vs full energy?
    cpp_use_weighted = np.sum(cpp_track_calo_sigma / cpp_track_energy < cpp_track_resolution)
    torch_use_weighted = np.sum(torch_track_calo_sigma / (torch_track_energy + 1e-30) < torch_track_resolution)
    
    # Tower-level stats
    torch_n_towers_with_tracks = np.sum(torch_tower_track_sigma > 0)
    torch_mean_tower_sigma = torch_tower_track_sigma[torch_tower_track_sigma > 0].mean() if torch_n_towers_with_tracks > 0 else 0
    
    if has_cpp_tracksigma and len(cpp_tower_track_sigma) > 0:
        cpp_n_towers_with_tracks = np.sum(cpp_tower_track_sigma > 0)
        cpp_mean_tower_sigma = cpp_tower_track_sigma[cpp_tower_track_sigma > 0].mean() if cpp_n_towers_with_tracks > 0 else 0
        tower_sigma_text = f"""
Tower-level Track Sigma:
  Towers with tracks:
    C++: {cpp_n_towers_with_tracks}, Torch: {torch_n_towers_with_tracks}
  Mean tower track sigma:
    C++: {cpp_mean_tower_sigma:.4f}, Torch: {torch_mean_tower_sigma:.4f}"""
    else:
        tower_sigma_text = f"""
Tower-level Track Sigma:
  Towers with tracks: {torch_n_towers_with_tracks}
  Mean tower track sigma: {torch_mean_tower_sigma:.4f}
  (No C++ data for comparison)"""
    
    stats_text = f"""Track Sigma Statistics (Step 7)
    
Tracks with valid sigma:
  C++: {cpp_n_tracks}
  Torch: {torch_n_tracks}
  
Energy Guess Selection:
  Using weighted E (calo σ/E < track res):
    C++: {cpp_use_weighted} ({100*cpp_use_weighted/cpp_n_tracks:.1f}%)
    Torch: {torch_use_weighted} ({100*torch_use_weighted/torch_n_tracks:.1f}%)
  Using full E:
    C++: {cpp_n_tracks - cpp_use_weighted} ({100*(cpp_n_tracks-cpp_use_weighted)/cpp_n_tracks:.1f}%)
    Torch: {torch_n_tracks - torch_use_weighted} ({100*(torch_n_tracks-torch_use_weighted)/torch_n_tracks:.1f}%)

Mean per-track values:
  Track Energy: C++={cpp_track_energy.mean():.2f}, Torch={torch_track_energy.mean():.2f}
  Calo Sigma: C++={cpp_track_calo_sigma.mean():.4f}, Torch={torch_track_calo_sigma.mean():.4f}
  Energy Guess: C++={cpp_track_energy_guess.mean():.2f}, Torch={torch_track_energy_guess.mean():.2f}
  σ² Contrib: C++={cpp_track_sigma_sq.mean():.2f}, Torch={torch_track_sigma_sq.mean():.2f}
{tower_sigma_text}
"""
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    
    plot_file = output_dir / "track_sigma.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved {plot_file}")
    
    print(f"  ✓ Step 7 validation complete.")
    
    print(f"\n  ✓ All SimpleCalorimeter validation complete. Plots saved to {output_dir}")

    if debug:
        debug_dict = {
            "Track Energy":{
                "Bins": _track_energy_bins,
                "C++ Counts": _cpp_track_energy_counts,
                "Torch Counts": _torch_track_energy_counts,
            },
            "Track Momentum Resolution":{
                "Bins": _track_res_bins,
                "C++ Counts": _cpp_track_res_counts,
                "Torch Counts": _torch_track_res_counts,
            },
            "Calorimeter Resolution at Tower η":{
                "Bins": _cal_sigma_bins,
                "C++ Counts": _cpp_cal_sigma_counts,
                "Torch Counts": _torch_cal_sigma_counts,
            },
            "Tower η for Valid Tracks":{
                "Bins": _tower_eta_bins,
                "C++ Counts": _cpp_tower_eta_counts,
                "Torch Counts": _torch_tower_eta_counts,
            },
            "Track Energy Guess":{
                "Bins": _e_guess_bins,
                "C++ Counts": _cpp_track_e_guess_counts,
                "Torch Counts": _torch_track_e_guess_counts,
            },
            "Per-Track Sigma² Contribution":{
                "Bins": _sigma_sq_bins,
                "C++ Counts": _cpp_track_sigma_sq_counts,
                "Torch Counts": _torch_track_sigma_sq_counts,
            },
            "Tower Track Sigma":{
                "Bins": _tracksigma_bins,
                "C++ Counts": _cpp_tracksigma_counts,
                "Torch Counts": _torch_tracksigma_counts,
            },
        }
        print(f"\n\nDEBUG: SimpleCalorimeter Step 7 detailed histogram data:")
        for k,v in debug_dict.items():
            print(f"\n  {k}:")
            for subk, subv in v.items():
                print(f"    {subk}: {subv}")

def validate_simple_cal(
    ecal_results: Dict,
    cpp_fractions_file: str,
    cpp_towerhits_file: str,
    cpp_towerenergy_file: str,
    cpp_smearing_file: str,
    cpp_pertrack_file: str,
    cpp_tracksigma_file: str,
    output_dir: str,
    debug: bool = False
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
    validate_simple_cal_step_1(
        ecal_results,
        cpp_fractions_file,
        output_dir,
    )
    
    # ========== Step 2: Binning (Tower Hits) ==========
    validate_simple_cal_step_2(
        ecal_results,
        cpp_towerhits_file,
        output_dir,
    )

    # ========== Step 3: Tower Energy Aggregation ==========
    tower_results, cpp_towers_df = validate_simple_cal_step_3(
        ecal_results,
        cpp_towerenergy_file,
        output_dir,
    )

    # ============ Step 4: Tower Centers ============
    validate_simple_cal_step_4(
        tower_results,
        cpp_towers_df,
        output_dir,
    )
    
    # ============ Step 5: Resolution Smearing ============
    validate_simple_cal_step_5(
        tower_results,
        cpp_smearing_file,
        output_dir,
    )
    
    # ============ Step 6: Track Sigma per Tower ============
    validate_simple_cal_step_6(
        tower_results,
        cpp_pertrack_file,
        cpp_tracksigma_file,
        output_dir,
        debug = debug,
    )

    # ============ Step 8: Final Tower Outputs ============
    validate_simple_cal_step_8(
        ecal_results,
        output_dir,
    )


def validate_simple_cal_step_8(
    ecal_results: Dict,
    output_dir: str,
) -> None:
    """
    Validate SimpleCalorimeter Step 8: Final tower outputs before ROOT conversion.
    
    This validates the tower_tensor, eflow_photon_tensor, and eflow_track_tensor
    that will be written to the ROOT file.
    
    Args:
        ecal_results: Dict with keys including 'tower_tensors', 'eflow_photon_tensors', 'eflow_track_tensors'
        output_dir: Directory to save validation plots
    """
    from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP
    
    print(f"\n{'='*70}")
    print("Validating SimpleCalorimeter Step 8: Final Tower Outputs")
    print(f"{'='*70}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect tower tensors
    tower_tensors = ecal_results.get('tower_tensors', [])
    eflow_photon_tensors = ecal_results.get('eflow_photon_tensors', [])
    eflow_track_tensors = ecal_results.get('eflow_track_tensors', [])
    
    # Concatenate all tensors
    if tower_tensors and any(t.shape[0] > 0 for t in tower_tensors):
        all_towers = torch.cat([t for t in tower_tensors if t.shape[0] > 0], dim=0)
    else:
        all_towers = torch.zeros(0, 24)
    
    if eflow_photon_tensors and any(t.shape[0] > 0 for t in eflow_photon_tensors):
        all_eflow_photons = torch.cat([t for t in eflow_photon_tensors if t.shape[0] > 0], dim=0)
    else:
        all_eflow_photons = torch.zeros(0, 24)
    
    if eflow_track_tensors and any(t.shape[0] > 0 for t in eflow_track_tensors):
        all_eflow_tracks = torch.cat([t for t in eflow_track_tensors if t.shape[0] > 0], dim=0)
    else:
        all_eflow_tracks = torch.zeros(0, 24)
    
    # Print summary statistics
    print(f"\n  Tower Tensor (ECalTower):")
    print(f"    Count: {all_towers.shape[0]}")
    if all_towers.shape[0] > 0:
        tower_eta = all_towers[:, CMAP["ETA"]].numpy()
        tower_phi = all_towers[:, CMAP["PHI"]].numpy()
        tower_e = all_towers[:, CMAP["E"]].numpy()
        print(f"    Eta range: [{tower_eta.min():.4f}, {tower_eta.max():.4f}]")
        print(f"    Phi range: [{tower_phi.min():.4f}, {tower_phi.max():.4f}]")
        print(f"    E range: [{tower_e.min():.4f}, {tower_e.max():.4f}]")
        
        # Check for edge eta values
        print(f"\n    Towers with |Eta| > 4.5:")
        high_eta_mask = np.abs(tower_eta) > 4.5
        print(f"      Count: {high_eta_mask.sum()}")
        if high_eta_mask.sum() > 0:
            high_eta_values = tower_eta[high_eta_mask]
            unique_high_eta = np.unique(np.round(high_eta_values, 4))
            print(f"      Unique Eta values: {unique_high_eta}")
    
    print(f"\n  EFlowPhoton Tensor:")
    print(f"    Count: {all_eflow_photons.shape[0]}")
    if all_eflow_photons.shape[0] > 0:
        eflow_eta = all_eflow_photons[:, CMAP["ETA"]].numpy()
        eflow_phi = all_eflow_photons[:, CMAP["PHI"]].numpy()
        print(f"    Eta range: [{eflow_eta.min():.4f}, {eflow_eta.max():.4f}]")
        print(f"    Phi range: [{eflow_phi.min():.4f}, {eflow_phi.max():.4f}]")
        
        # Check for edge eta values
        print(f"\n    EFlowPhotons with |Eta| > 4.5:")
        high_eta_mask = np.abs(eflow_eta) > 4.5
        print(f"      Count: {high_eta_mask.sum()}")
        if high_eta_mask.sum() > 0:
            high_eta_values = eflow_eta[high_eta_mask]
            unique_high_eta = np.unique(np.round(high_eta_values, 4))
            print(f"      Unique Eta values: {unique_high_eta}")
    
    print(f"\n  EFlowTrack Tensor:")
    print(f"    Count: {all_eflow_tracks.shape[0]}")
    if all_eflow_tracks.shape[0] > 0:
        track_eta = all_eflow_tracks[:, CMAP["ETA"]].numpy()
        track_phi = all_eflow_tracks[:, CMAP["PHI"]].numpy()
        print(f"    Eta range: [{track_eta.min():.4f}, {track_eta.max():.4f}]")
        print(f"    Phi range: [{track_phi.min():.4f}, {track_phi.max():.4f}]")
    
    # Also check the intermediate tower_results to see if we're losing towers in filtering
    tower_results = ecal_results.get('tower_results', [])
    if tower_results:
        all_tower_eta = []
        all_tower_energy_final = []
        for r in tower_results:
            all_tower_eta.append(r['tower_eta'].numpy())
            all_tower_energy_final.append(r['tower_energy_final'].numpy())
        
        all_tower_eta = np.concatenate(all_tower_eta)
        all_tower_energy_final = np.concatenate(all_tower_energy_final)
        
        print(f"\n  Intermediate tower_results (before energy filtering):")
        print(f"    Total towers: {len(all_tower_eta)}")
        print(f"    Eta range: [{all_tower_eta.min():.4f}, {all_tower_eta.max():.4f}]")
        print(f"    Towers with energy > 0: {(all_tower_energy_final > 0).sum()}")
        
        # Check high-eta towers
        high_eta_mask = np.abs(all_tower_eta) > 4.5
        print(f"\n    Towers with |Eta| > 4.5 (before filtering):")
        print(f"      Count: {high_eta_mask.sum()}")
        if high_eta_mask.sum() > 0:
            high_eta_energies = all_tower_energy_final[high_eta_mask]
            print(f"      With energy > 0: {(high_eta_energies > 0).sum()}")
            unique_high_eta = np.unique(np.round(all_tower_eta[high_eta_mask], 4))
            print(f"      Unique Eta values: {unique_high_eta}")
    
    print(f"\n  ✓ Step 8 validation complete")



