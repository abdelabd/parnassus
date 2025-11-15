"""
PyTorch implementation of Delphes Efficiency module.

Implements the ChargedHadronTrackingEfficiency from delphes_card_CMS.tcl
as a differentiable PyTorch module.
"""

import torch
import torch.nn as nn
import numpy as np


class Efficiency(nn.Module):
    """
    PyTorch implementation of Delphes Efficiency module.
    
    Applies tracking efficiency based on particle kinematics (pt, eta)
    similar to Delphes ChargedHadronTrackingEfficiency.
    
    Input shape: (N, 15) where:
        - column 0: PID (Particle ID)
        - column 1: Status
        - column 2: Charge
        - column 3: E (Energy)
        - columns 4-6: Px, Py, Pz (3-momentum)
        - column 7: PT (transverse momentum)
        - column 8: Eta (pseudorapidity)
        - column 9: Phi (azimuthal angle)
        - column 10: T (time)
        - columns 11-13: X, Y, Z (position)
    
    The efficiency formula from CMS card:
        (pt <= 0.1)   * (0.00) +
        (abs(eta) <= 1.5) * (pt > 0.1 && pt <= 1.0)   * (0.70) +
        (abs(eta) <= 1.5) * (pt > 1.0)                * (0.95) +
        (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1 && pt <= 1.0)   * (0.60) +
        (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 1.0)                * (0.85) +
        (abs(eta) > 2.5)                                                * (0.00)
    """
    
    def __init__(self, 
                 efficiency_formula='charged_hadron_cms',
                 deterministic=False,
                 device='cpu'):
        """
        Args:
            efficiency_formula: Name of predefined formula or custom callable
            deterministic: If True, apply efficiency deterministically (threshold)
                         If False (default), apply stochastically (random sampling)
            device: torch device ('cpu' or 'cuda')
        """
        super().__init__()
        self.deterministic = deterministic
        self.device = device
        
        # Load efficiency formula
        if efficiency_formula == 'charged_hadron_cms':
            self.efficiency_func = self._charged_hadron_cms_efficiency
        elif efficiency_formula == 'electron_cms':
            self.efficiency_func = self._electron_cms_efficiency
        elif efficiency_formula == 'muon_cms':
            self.efficiency_func = self._muon_cms_efficiency
        elif callable(efficiency_formula):
            self.efficiency_func = efficiency_formula
        else:
            raise ValueError(f"Unknown efficiency formula: {efficiency_formula}")
    
    @staticmethod
    def _charged_hadron_cms_efficiency(pt, eta):
        """
        CMS charged hadron tracking efficiency formula.
        
        Args:
            pt: transverse momentum (GeV)
            eta: pseudorapidity
            
        Returns:
            efficiency: value between 0 and 1
        """
        abs_eta = torch.abs(eta)
        
        # Initialize with zeros
        eff = torch.zeros_like(pt)
        
        # Region 1: Central barrel, low pt (0.1 < pt <= 1.0, |eta| <= 1.5)
        mask1 = (pt > 0.1) & (pt <= 1.0) & (abs_eta <= 1.5)
        eff = torch.where(mask1, torch.tensor(0.70, device=pt.device), eff)
        
        # Region 2: Central barrel, high pt (pt > 1.0, |eta| <= 1.5)
        mask2 = (pt > 1.0) & (abs_eta <= 1.5)
        eff = torch.where(mask2, torch.tensor(0.95, device=pt.device), eff)
        
        # Region 3: Forward endcap, low pt (0.1 < pt <= 1.0, 1.5 < |eta| <= 2.5)
        mask3 = (pt > 0.1) & (pt <= 1.0) & (abs_eta > 1.5) & (abs_eta <= 2.5)
        eff = torch.where(mask3, torch.tensor(0.60, device=pt.device), eff)
        
        # Region 4: Forward endcap, high pt (pt > 1.0, 1.5 < |eta| <= 2.5)
        mask4 = (pt > 1.0) & (abs_eta > 1.5) & (abs_eta <= 2.5)
        eff = torch.where(mask4, torch.tensor(0.85, device=pt.device), eff)
        
        # Region 5: Very forward (|eta| > 2.5) - efficiency is 0 (already initialized)
        
        return eff
    
    @staticmethod
    def _electron_cms_efficiency(pt, eta):
        """CMS electron tracking efficiency formula."""
        abs_eta = torch.abs(eta)
        eff = torch.zeros_like(pt)
        
        # Central barrel
        mask1 = (pt > 0.1) & (pt <= 1.0) & (abs_eta <= 1.5)
        eff = torch.where(mask1, torch.tensor(0.73, device=pt.device), eff)
        
        mask2 = (pt > 1.0) & (pt <= 1.0e2) & (abs_eta <= 1.5)
        eff = torch.where(mask2, torch.tensor(0.95, device=pt.device), eff)
        
        mask3 = (pt > 1.0e2) & (abs_eta <= 1.5)
        eff = torch.where(mask3, torch.tensor(0.99, device=pt.device), eff)
        
        # Forward endcap
        mask4 = (pt > 0.1) & (pt <= 1.0) & (abs_eta > 1.5) & (abs_eta <= 2.5)
        eff = torch.where(mask4, torch.tensor(0.50, device=pt.device), eff)
        
        mask5 = (pt > 1.0) & (pt <= 1.0e2) & (abs_eta > 1.5) & (abs_eta <= 2.5)
        eff = torch.where(mask5, torch.tensor(0.83, device=pt.device), eff)
        
        mask6 = (pt > 1.0e2) & (abs_eta > 1.5) & (abs_eta <= 2.5)
        eff = torch.where(mask6, torch.tensor(0.90, device=pt.device), eff)
        
        return eff
    
    @staticmethod
    def _muon_cms_efficiency(pt, eta):
        """CMS muon tracking efficiency formula."""
        abs_eta = torch.abs(eta)
        eff = torch.zeros_like(pt)
        
        # Central barrel
        mask1 = (pt > 0.1) & (pt <= 1.0) & (abs_eta <= 1.5)
        eff = torch.where(mask1, torch.tensor(0.75, device=pt.device), eff)
        
        mask2 = (pt > 1.0) & (pt <= 1.0e3) & (abs_eta <= 1.5)
        eff = torch.where(mask2, torch.tensor(0.99, device=pt.device), eff)
        
        mask3 = (pt > 1.0e3) & (abs_eta <= 1.5)
        eff3 = 0.99 * torch.exp(0.5 - pt * 5.0e-4)
        eff = torch.where(mask3, eff3, eff)
        
        # Forward endcap
        mask4 = (pt > 0.1) & (pt <= 1.0) & (abs_eta > 1.5) & (abs_eta <= 2.5)
        eff = torch.where(mask4, torch.tensor(0.70, device=pt.device), eff)
        
        mask5 = (pt > 1.0) & (pt <= 1.0e3) & (abs_eta > 1.5) & (abs_eta <= 2.5)
        eff = torch.where(mask5, torch.tensor(0.98, device=pt.device), eff)
        
        mask6 = (pt > 1.0e3) & (abs_eta > 1.5) & (abs_eta <= 2.5)
        eff6 = 0.98 * torch.exp(0.5 - pt * 5.0e-4)
        eff = torch.where(mask6, eff6, eff)
        
        return eff
    
    def forward(self, particles, return_mask=False):
        """
        Apply efficiency filter to particles.
        
        Args:
            particles: tensor of shape (N, 15) or (batch, N, 15)
                column 0: PID (Particle ID)
                column 1: Status
                column 2: Charge
                column 3: E (Energy)
                columns 4-6: Px, Py, Pz (3-momentum)
                column 7: PT (transverse momentum, pre-computed)
                column 8: Eta (pseudorapidity, pre-computed)
                column 9: Phi (azimuthal angle, pre-computed)
                column 10: T (time)
                columns 11-13: X, Y, Z (position)
            return_mask: if True, return (filtered_particles, mask)
                        if False, return only filtered_particles
        
        Returns:
            filtered_particles: particles that pass efficiency cut
            mask (optional): boolean mask of particles that passed
        """
        # Handle both batched and unbatched inputs
        input_shape = particles.shape
        if len(input_shape) == 2:
            particles = particles.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, n_particles, features = particles.shape
        assert features == 15, f"Expected 15 features, got {features}"
        
        # Move to device
        particles = particles.to(self.device)
        
        # Extract pre-computed kinematics from Delphes (columns 7-9)
        pt = particles[..., 7]   # Column 7: PT (transverse momentum)
        eta = particles[..., 8]  # Column 8: Eta (pseudorapidity)
        phi = particles[..., 9]  # Column 9: Phi (azimuthal angle)
        
        # Compute efficiency for each particle
        efficiency = self.efficiency_func(pt, eta)
        
        # Apply efficiency
        if self.deterministic:
            # Deterministic: keep if efficiency > 0.5
            mask = efficiency > 0.5
        else:
            # Stochastic: random sampling
            random_vals = torch.rand_like(efficiency)
            mask = random_vals <= efficiency
        
        # Apply mask to filter particles
        if squeeze_output:
            particles = particles.squeeze(0)
            mask = mask.squeeze(0)
            
            if return_mask:
                return particles[mask], mask
            else:
                return particles[mask]
        else:
            # For batched input, need to handle variable-length outputs
            # Return as list of tensors or use padding
            filtered_particles_list = []
            masks_list = []
            
            for i in range(batch_size):
                batch_mask = mask[i]
                filtered = particles[i][batch_mask]
                filtered_particles_list.append(filtered)
                masks_list.append(batch_mask)
            
            if return_mask:
                return filtered_particles_list, masks_list
            else:
                return filtered_particles_list
    
    def get_efficiency_map(self, pt_range=(0, 100), eta_range=(-3, 3), 
                          n_pts=100, n_etas=100):
        """
        Generate a 2D efficiency map for visualization.
        
        Args:
            pt_range: (min, max) pt values in GeV
            eta_range: (min, max) eta values
            n_pts: number of pt bins
            n_etas: number of eta bins
            
        Returns:
            pt_grid, eta_grid, efficiency_map
        """
        pt_vals = torch.linspace(pt_range[0], pt_range[1], n_pts, device=self.device)
        eta_vals = torch.linspace(eta_range[0], eta_range[1], n_etas, device=self.device)
        
        pt_grid, eta_grid = torch.meshgrid(pt_vals, eta_vals, indexing='ij')
        
        efficiency_map = self.efficiency_func(pt_grid.flatten(), eta_grid.flatten())
        efficiency_map = efficiency_map.reshape(n_pts, n_etas)
        
        return pt_grid.cpu().numpy(), eta_grid.cpu().numpy(), efficiency_map.cpu().numpy()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Delphes Efficiency PyTorch Module\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create some example particles in the (N, 15) format expected by the module
    n_particles = 10
    
    # Generate random particles
    particles = torch.zeros((n_particles, 15))
    
    # Generate random kinematics
    pt_values = torch.rand(n_particles) * 50 + 0.1  # 0.1 to 50 GeV
    eta_values = torch.rand(n_particles) * 6 - 3  # -3 to 3
    phi_values = torch.rand(n_particles) * 2 * np.pi - np.pi
    
    # Column 0: PID (charged pions)
    particles[:, 0] = 211
    # Column 1: Status (stable)
    particles[:, 1] = 1
    # Column 2: Charge
    particles[:, 2] = 1
    # Column 3: E (approximate from pt for massless particles)
    particles[:, 3] = pt_values * torch.cosh(eta_values)
    # Columns 4-6: Px, Py, Pz
    particles[:, 4] = pt_values * torch.cos(phi_values)  # Px
    particles[:, 5] = pt_values * torch.sin(phi_values)  # Py
    particles[:, 6] = pt_values * torch.sinh(eta_values)  # Pz
    # Column 7: PT (pre-computed)
    particles[:, 7] = pt_values
    # Column 8: Eta (pre-computed)
    particles[:, 8] = eta_values
    # Column 9: Phi (pre-computed)
    particles[:, 9] = phi_values
    # Column 10: T (time)
    particles[:, 10] = torch.randn(n_particles)
    # Columns 11-13: X, Y, Z (position)
    particles[:, 11:14] = torch.randn(n_particles, 3)
    
    print("Input particles:")
    print(f"Shape: {particles.shape}")
    print(f"Number of particles: {n_particles}\n")
    
    # Test with different efficiency formulas
    for formula_name in ['charged_hadron_cms', 'electron_cms', 'muon_cms']:
        print(f"\n{'='*60}")
        print(f"Testing {formula_name} efficiency")
        print('='*60)
        
        # Create efficiency module
        eff_module = DelphesEfficiency(
            efficiency_formula=formula_name,
            deterministic=False,  # Stochastic mode
            device='cpu'
        )
        
        # Apply efficiency
        filtered_particles, mask = eff_module(particles, return_mask=True)
        
        print(f"\nResults:")
        print(f"Input particles: {n_particles}")
        print(f"Passed efficiency: {mask.sum().item()}")
        print(f"Rejection rate: {(1 - mask.float().mean().item())*100:.1f}%")
        print(f"\nOutput shape: {filtered_particles.shape}")
        
        # Show which particles passed (use pre-computed PT and Eta from columns 7 and 8)
        pt = particles[:, 7]
        eta = particles[:, 8]
        print(f"\nPer-particle results:")
        print(f"{'Index':<6} {'pt (GeV)':<10} {'eta':<10} {'Passed':<8}")
        print("-" * 40)
        for i in range(n_particles):
            print(f"{i:<6} {pt[i].item():<10.2f} {eta[i].item():<10.2f} "
                  f"{'✓' if mask[i] else '✗':<8}")
    
    # Generate and display efficiency map
    print(f"\n{'='*60}")
    print("Generating efficiency map for charged hadrons")
    print('='*60)
    
    eff_module = DelphesEfficiency(efficiency_formula='charged_hadron_cms')
    pt_grid, eta_grid, eff_map = eff_module.get_efficiency_map()
    
    print(f"\nEfficiency map shape: {eff_map.shape}")
    print(f"Min efficiency: {eff_map.min():.3f}")
    print(f"Max efficiency: {eff_map.max():.3f}")
    print(f"Mean efficiency: {eff_map.mean():.3f}")
    
    # Test batched input
    print(f"\n{'='*60}")
    print("Testing batched input")
    print('='*60)
    
    batch_size = 3
    batched_particles = torch.stack([particles] * batch_size)
    print(f"Batched input shape: {batched_particles.shape}")
    
    filtered_list = eff_module(batched_particles)
    print(f"Number of batches: {len(filtered_list)}")
    for i, filtered in enumerate(filtered_list):
        print(f"Batch {i}: {filtered.shape[0]} particles passed")
    
    print("\n✓ All tests completed successfully!")
