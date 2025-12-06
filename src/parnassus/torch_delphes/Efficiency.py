"""
PyTorch implementation of Delphes Efficiency module.

Implements the ChargedHadronTrackingEfficiency from delphes_card_CMS.tcl
as a differentiable PyTorch module.
"""

import torch
import torch.nn as nn
import numpy as np

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP

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
        - column 14: mass
        - column 15: mask 
    
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
                 device='cpu'):
        """
        Args:
            efficiency_formula: Name of predefined formula or custom callable
            device: torch device ('cpu' or 'cuda')
        """
        super().__init__()
        self.device = device
        self.efficiency_formula = efficiency_formula
        
        # Load efficiency formula
        if self.efficiency_formula == 'charged_hadron_cms':
            self.efficiency_func = self._charged_hadron_cms_efficiency
            self.pdg_filter_func = self._charged_hadron_pdg_filter
        elif self.efficiency_formula == 'electron_cms':
            self.efficiency_func = self._electron_cms_efficiency
            self.pdg_filter_func = self._electron_pdg_filter
        elif self.efficiency_formula == 'muon_cms':
            self.efficiency_func = self._muon_cms_efficiency
            self.pdg_filter_func = self._muon_pdg_filter
        elif callable(self.efficiency_formula):
            self.efficiency_func = self.efficiency_formula
            self.pdg_filter_func = None
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
    
    @staticmethod
    def _charged_hadron_pdg_filter(particles):
        """
        Filter charged hadrons based on PDG IDs.
        
        Args:
            particles: tensor of shape (N, 15) or (B, N, 15)
            
        Returns:
            mask: boolean tensor indicating charged hadrons
        """
        pid = particles[..., CMAP["PID"]]
        q_final = particles[..., CMAP["CHARGE"]]
        abs_pid = torch.abs(pid)
        is_charged = torch.abs(q_final) > 1.0e-9
        
        electron_mask = (abs_pid == 11) & is_charged
        muon_mask = (abs_pid == 13) & is_charged
        pid_mask = is_charged & ~electron_mask & ~muon_mask
        
        return pid_mask
    
    @staticmethod
    def _electron_pdg_filter(particles):
        """
        Filter electrons based on PDG IDs.
        
        Args:
            particles: tensor of shape (N, 15) or (B, N, 15)
            
        Returns:
            mask: boolean tensor indicating electrons
        """
        pid = particles[..., CMAP["PID"]]
        q_final = particles[..., CMAP["CHARGE"]]
        abs_pid = torch.abs(pid)
        is_charged = torch.abs(q_final) > 1.0e-9
        
        pid_mask = (abs_pid == 11) & is_charged
        
        return pid_mask
    
    @staticmethod
    def _muon_pdg_filter(particles):
        """
        Filter muons based on PDG IDs.
        
        Args:
            particles: tensor of shape (N, 15) or (B, N, 15)
            
        Returns:
            mask: boolean tensor indicating muons
        """
        pid = particles[..., CMAP["PID"]]
        q_final = particles[..., CMAP["CHARGE"]]
        abs_pid = torch.abs(pid)
        is_charged = torch.abs(q_final) > 1.0e-9
        
        pid_mask = (abs_pid == 13) & is_charged
        
        return pid_mask
    
    @staticmethod
    def _neutral_pdg_filter(particles):
        """
        Filter neutral particles based on PDG IDs.
        
        Args:
            particles: tensor of shape (N, 15) or (B, N, 15)
            
        Returns:
            mask: boolean tensor indicating neutral particles
        """
        q_final = particles[..., CMAP["CHARGE"]]
        is_charged = torch.abs(q_final) > 1.0e-9
        
        pid_mask = ~is_charged
        
        return pid_mask
    
    def forward(self, particles):
        """
        Apply efficiency filter to particles using mask-based filtering.
        
        Args:
            particles: tensor of shape (N, 15), (N, 16), (B, N, 15), or (B, N, 16)
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
                column 15 (optional): mask (1.0 for real particles, 0.0 for padding)

        Returns:
            filtered_particles: tensor with mask in column 15
                               Single event: (N, 16) with mask
                               Batched: (B, N, 16) with updated mask
        """
    
        # We want to compute effiency vector based on particles that satisfy:
            # 1. real particles (IS_NOT_PAD == 1)
            # 2. particles that passed propagation (PASS_PROP == 1)
            # 3. particles of the desired type

        pid_mask = self.pdg_filter_func(particles)
        mask = particles[:, CMAP["IS_NOT_PAD"]] * particles[:, CMAP["PASS_PROP"]] * pid_mask.float()

        has_pass_eff = False
        if particles.shape[1] > CMAP["PASS_EFF"]:
            has_pass_eff = True
        # if has_pass_eff:
        #     mask = mask * (particles[:, CMAP["PASS_EFF"]]>0.5).float()

        # Extract pre-computed kinematics from Delphes (columns 7-8)
        mask_where = torch.where(mask > 0.5)[0]
        pt = particles[mask_where, CMAP["PT"]]   # Column 7: PT (transverse momentum)
        eta = particles[mask_where, CMAP["ETA"]]  # Column 8: Eta (pseudorapidity)

        # Compute efficiency for each particle
        efficiency = self.efficiency_func(pt, eta)
        
        # Apply efficiency stochastically
        passed = torch.rand_like(efficiency) < efficiency
        
        # Only real particles (mask==1) can pass efficiency
        if has_pass_eff:
            passed_full = particles[:, CMAP["PASS_EFF"]].clone().bool().to(particles.device)
            passed_full[mask_where] = passed
            particles[:, CMAP["PASS_EFF"]] = passed_full.double()
        else:
            passed_full = torch.zeros(particles.shape[0], device=particles.device, dtype=torch.bool)
            passed_full[mask_where] = passed
            particles = torch.cat(
                [particles, passed_full.unsqueeze(-1)], dim=-1
            )
        
        return particles
    
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
    particles[:, CMAP["PID"]] = 211
    # Column 1: Status (stable)
    particles[:, CMAP["STATUS"]] = 1
    # Column 2: Charge
    particles[:, CMAP["CHARGE"]] = 1
    # Column 3: E (approximate from pt for massless particles)
    particles[:, CMAP["E"]] = pt_values * torch.cosh(eta_values)
    # Columns 4-6: Px, Py, Pz
    particles[:, CMAP["PX"]] = pt_values * torch.cos(phi_values)  # Px
    particles[:, CMAP["PY"]] = pt_values * torch.sin(phi_values)  # Py
    particles[:, CMAP["PZ"]] = pt_values * torch.sinh(eta_values)  # Pz
    # Column 7: PT (pre-computed)
    particles[:, CMAP["PT"]] = pt_values
    # Column 8: Eta (pre-computed)
    particles[:, CMAP["ETA"]] = eta_values
    # Column 9: Phi (pre-computed)
    particles[:, CMAP["PHI"]] = phi_values
    # Column 10: T (time)
    particles[:, CMAP["T"]] = torch.randn(n_particles)
    # Columns 11-13: X, Y, Z (position)
    particles[:, CMAP["X"]:CMAP["Z"]+1] = torch.randn(n_particles, 3)

    print("Input particles:")
    print(f"Shape: {particles.shape}")
    print(f"Number of particles: {n_particles}\n")
    
    # Test with different efficiency formulas
    for formula_name in ['charged_hadron_cms', 'electron_cms', 'muon_cms']:
        print(f"\n{'='*60}")
        print(f"Testing {formula_name} efficiency")
        print('='*60)
        
        # Create efficiency module
        eff_module = Efficiency(
            efficiency_formula=formula_name,
            device='cpu'
        )
        
        # Apply efficiency
        filtered_particles, mask = eff_module(particles)
        
        print(f"\nResults:")
        print(f"Input particles: {n_particles}")
        print(f"Passed efficiency: {mask.sum().item()}")
        print(f"Rejection rate: {(1 - mask.float().mean().item())*100:.1f}%")
        print(f"\nOutput shape: {filtered_particles.shape}")
        
        # Show which particles passed (use pre-computed PT and Eta from columns 7 and 8)
        pt = particles[:, CMAP["PT"]]
        eta = particles[:, CMAP["ETA"]]
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
    
    eff_module = Efficiency(efficiency_formula='charged_hadron_cms')
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
