"""
Visualization script for Delphes Efficiency module.

Creates plots showing the efficiency as a function of pt and eta,
matching the formulas from delphes_card_CMS.tcl
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from parnassus.src.parnassus.torch_delphes.Efficiency import DelphesEfficiency


def plot_efficiency_map(formula_name, title):
    """Plot 2D efficiency map."""
    eff_module = DelphesEfficiency(efficiency_formula=formula_name)
    pt_grid, eta_grid, eff_map = eff_module.get_efficiency_map(
        pt_range=(0, 50),
        eta_range=(-3, 3),
        n_pts=200,
        n_etas=200
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # 2D heatmap
    im = ax1.contourf(eta_grid, pt_grid, eff_map, levels=20, cmap='RdYlGn')
    ax1.set_xlabel('η (pseudorapidity)', fontsize=12)
    ax1.set_ylabel('pT (GeV)', fontsize=12)
    ax1.set_title(f'{title}\n2D Efficiency Map', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add region boundaries
    ax1.axvline(x=1.5, color='white', linestyle='--', linewidth=2, alpha=0.7, label='|η|=1.5')
    ax1.axvline(x=-1.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=2.5, color='white', linestyle='--', linewidth=2, alpha=0.7, label='|η|=2.5')
    ax1.axvline(x=-2.5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax1.legend(loc='upper right')
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Efficiency', fontsize=12)
    
    # 1D slices at different eta values
    eta_slices = [0.0, 1.0, 2.0, 2.8]
    colors = ['blue', 'green', 'orange', 'red']
    
    for eta_val, color in zip(eta_slices, colors):
        eta_idx = np.argmin(np.abs(eta_grid[0, :] - eta_val))
        ax2.plot(pt_grid[:, eta_idx], eff_map[:, eta_idx], 
                label=f'η = {eta_val:.1f}', color=color, linewidth=2)
    
    ax2.set_xlabel('pT (GeV)', fontsize=12)
    ax2.set_ylabel('Efficiency', fontsize=12)
    ax2.set_title(f'{title}\nEfficiency vs pT at different η', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    return fig


def plot_efficiency_comparison():
    """Compare efficiency across different particle types."""
    formulas = {
        'Charged Hadron': 'charged_hadron_cms',
        'Electron': 'electron_cms',
        'Muon': 'muon_cms'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot 1: Central barrel (|η| < 1.5)
    ax = axes[0]
    pt_vals = np.linspace(0, 50, 500)
    eta_central = 0.5
    
    for name, formula in formulas.items():
        eff_module = DelphesEfficiency(efficiency_formula=formula)
        pt_tensor = torch.tensor(pt_vals, dtype=torch.float32)
        eta_tensor = torch.full_like(pt_tensor, eta_central)
        eff = eff_module.efficiency_func(pt_tensor, eta_tensor).numpy()
        ax.plot(pt_vals, eff, label=name, linewidth=2)
    
    ax.set_xlabel('pT (GeV)', fontsize=12)
    ax.set_ylabel('Efficiency', fontsize=12)
    ax.set_title(f'Central Barrel: |η| = {eta_central}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: Forward endcap (1.5 < |η| < 2.5)
    ax = axes[1]
    eta_forward = 2.0
    
    for name, formula in formulas.items():
        eff_module = DelphesEfficiency(efficiency_formula=formula)
        pt_tensor = torch.tensor(pt_vals, dtype=torch.float32)
        eta_tensor = torch.full_like(pt_tensor, eta_forward)
        eff = eff_module.efficiency_func(pt_tensor, eta_tensor).numpy()
        ax.plot(pt_vals, eff, label=name, linewidth=2)
    
    ax.set_xlabel('pT (GeV)', fontsize=12)
    ax.set_ylabel('Efficiency', fontsize=12)
    ax.set_title(f'Forward Endcap: |η| = {eta_forward}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 3: Efficiency vs η at fixed pT
    ax = axes[2]
    eta_vals = np.linspace(-3, 3, 500)
    pt_fixed = 10.0
    
    for name, formula in formulas.items():
        eff_module = DelphesEfficiency(efficiency_formula=formula)
        eta_tensor = torch.tensor(eta_vals, dtype=torch.float32)
        pt_tensor = torch.full_like(eta_tensor, pt_fixed)
        eff = eff_module.efficiency_func(pt_tensor, eta_tensor).numpy()
        ax.plot(eta_vals, eff, label=name, linewidth=2)
    
    ax.set_xlabel('η (pseudorapidity)', fontsize=12)
    ax.set_ylabel('Efficiency', fontsize=12)
    ax.set_title(f'Efficiency vs η at pT = {pt_fixed} GeV', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-1.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-2.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: Low pT behavior
    ax = axes[3]
    pt_vals_low = np.linspace(0, 5, 500)
    eta_central = 0.5
    
    for name, formula in formulas.items():
        eff_module = DelphesEfficiency(efficiency_formula=formula)
        pt_tensor = torch.tensor(pt_vals_low, dtype=torch.float32)
        eta_tensor = torch.full_like(pt_tensor, eta_central)
        eff = eff_module.efficiency_func(pt_tensor, eta_tensor).numpy()
        ax.plot(pt_vals_low, eff, label=name, linewidth=2)
    
    ax.set_xlabel('pT (GeV)', fontsize=12)
    ax.set_ylabel('Efficiency', fontsize=12)
    ax.set_title('Low pT Behavior (|η| = 0.5)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='pT threshold')
    ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, label='pT transition')
    
    plt.tight_layout()
    return fig


def demonstrate_stochastic_behavior():
    """Show the stochastic nature of the efficiency filter."""
    # Create test particles with fixed kinematics
    n_tests = 1000
    pt_test = 10.0  # GeV
    eta_test = 0.5
    
    particles = torch.zeros((n_tests, 10))
    particles[:, 4] = pt_test  # E
    particles[:, 5] = pt_test  # px (simplified)
    particles[:, 8] = 211  # charged pion
    particles[:, 9] = 1  # stable
    
    # Test efficiency module
    eff_module = DelphesEfficiency(
        efficiency_formula='charged_hadron_cms',
        deterministic=False
    )
    
    # Expected efficiency for pt=10 GeV, eta=0.5 (central barrel, high pt)
    expected_eff = 0.95
    
    # Run multiple times to get statistics
    n_runs = 100
    pass_rates = []
    
    for _ in range(n_runs):
        filtered, mask = eff_module(particles, return_mask=True)
        pass_rate = mask.float().mean().item()
        pass_rates.append(pass_rate)
    
    pass_rates = np.array(pass_rates)
    
    # Plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(pass_rates, bins=30, edgecolor='black', alpha=0.7, density=True)
    ax1.axvline(expected_eff, color='red', linestyle='--', linewidth=2, 
                label=f'Expected: {expected_eff:.2f}')
    ax1.axvline(np.mean(pass_rates), color='green', linestyle='-', linewidth=2,
                label=f'Observed: {np.mean(pass_rates):.3f}±{np.std(pass_rates):.3f}')
    ax1.set_xlabel('Fraction of particles passing', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'Stochastic Efficiency Distribution\n({n_tests} particles, {n_runs} runs)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Test at different efficiencies
    test_points = [
        (0.5, 0.5, 0.70),   # Low pT, central
        (10.0, 0.5, 0.95),  # High pT, central
        (0.5, 2.0, 0.60),   # Low pT, forward
        (10.0, 2.0, 0.85),  # High pT, forward
    ]
    
    results = []
    for pt, eta, expected in test_points:
        particles_test = torch.zeros((n_tests, 10))
        particles_test[:, 4] = pt
        particles_test[:, 5] = pt
        particles_test[:, 7] = pt * np.sinh(eta)  # pz
        particles_test[:, 8] = 211
        particles_test[:, 9] = 1
        
        pass_rates_test = []
        for _ in range(n_runs):
            _, mask = eff_module(particles_test, return_mask=True)
            pass_rates_test.append(mask.float().mean().item())
        
        results.append({
            'pt': pt,
            'eta': eta,
            'expected': expected,
            'mean': np.mean(pass_rates_test),
            'std': np.std(pass_rates_test)
        })
    
    # Plot comparison
    x_pos = np.arange(len(results))
    expected_vals = [r['expected'] for r in results]
    observed_vals = [r['mean'] for r in results]
    errors = [r['std'] for r in results]
    labels = [f"pt={r['pt']}\nη={r['eta']}" for r in results]
    
    ax2.bar(x_pos - 0.2, expected_vals, 0.4, label='Expected', alpha=0.7)
    ax2.bar(x_pos + 0.2, observed_vals, 0.4, yerr=errors, label='Observed', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Efficiency', fontsize=12)
    ax2.set_title('Expected vs Observed Efficiency', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Generating Delphes Efficiency visualizations...\n")
    
    # Create output directory
    import os
    os.makedirs('efficiency_plots', exist_ok=True)
    
    # Plot 1: Charged Hadron Efficiency
    print("1. Plotting Charged Hadron efficiency map...")
    fig1 = plot_efficiency_map('charged_hadron_cms', 'Charged Hadron Tracking Efficiency (CMS)')
    fig1.savefig('efficiency_plots/charged_hadron_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Electron Efficiency
    print("2. Plotting Electron efficiency map...")
    fig2 = plot_efficiency_map('electron_cms', 'Electron Tracking Efficiency (CMS)')
    fig2.savefig('efficiency_plots/electron_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # Plot 3: Muon Efficiency
    print("3. Plotting Muon efficiency map...")
    fig3 = plot_efficiency_map('muon_cms', 'Muon Tracking Efficiency (CMS)')
    fig3.savefig('efficiency_plots/muon_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # Plot 4: Comparison
    print("4. Plotting efficiency comparison...")
    fig4 = plot_efficiency_comparison()
    fig4.savefig('efficiency_plots/efficiency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # Plot 5: Stochastic behavior
    print("5. Demonstrating stochastic behavior...")
    fig5 = demonstrate_stochastic_behavior()
    fig5.savefig('efficiency_plots/stochastic_behavior.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    print("\n✓ All plots saved to 'efficiency_plots/' directory")
    print("\nGenerated files:")
    print("  - charged_hadron_efficiency.png")
    print("  - electron_efficiency.png")
    print("  - muon_efficiency.png")
    print("  - efficiency_comparison.png")
    print("  - stochastic_behavior.png")
