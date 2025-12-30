"""
PyTorch implementation of Delphes MomentumSmearing module.

Performs transverse momentum resolution smearing using a log-normal distribution.
"""

import torch
import torch.nn as nn
import numpy as np

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP

class MomentumSmearing(nn.Module):
    """
    PyTorch implementation of Delphes MomentumSmearing module.
    
    Applies momentum resolution smearing to particles based on their kinematics (pt, eta).
    The smearing is applied using a log-normal distribution to ensure positive PT values.
    
    Input shape: (N, 15) where:
        - column 0: PID (Particle ID)
        - column 1: Status
        - column 2: Charge
        - column 3: E (Energy)
        - columns 4-6: Px, Py, Pz (3-momentum)
        - column 7: PT (transverse momentum)
        - column 8: Eta (pseudorapidity for resolution formula - typically position-based)
        - column 9: Phi (azimuthal angle)
        - column 10: T (time)
        - columns 11-13: X, Y, Z (position)
        - column 14: Mass
        - column 15: mask
    
    NOTE: The Eta in column 8 is used for evaluating the resolution formula.
    However, when reconstructing the 4-vector after smearing, we recompute
    Eta from the momentum components (Px, Py, Pz) to match C++ Delphes behavior.
    
    The resolution formula from CMS card (for charged hadrons):
        (abs(eta) <= 0.5) * (pt > 0.1) * sqrt(0.06^2 + pt^2*1.3e-3^2) +
        (abs(eta) > 0.5 && abs(eta) <= 1.5) * (pt > 0.1) * sqrt(0.10^2 + pt^2*1.7e-3^2) +
        (abs(eta) > 1.5 && abs(eta) <= 2.5) * (pt > 0.1) * sqrt(0.25^2 + pt^2*3.1e-3^2)
    """
    
    def __init__(self, 
                 resolution_formula='charged_hadron_cms',
                 device='cpu'):
        """
        Args:
            resolution_formula: Name of predefined formula or custom callable
            deterministic: If True, no smearing (for testing)
            device: torch device ('cpu' or 'cuda')
        """
        super().__init__()
        self.device = device
        
        # Load resolution formula
        if resolution_formula == 'charged_hadron_cms':
            self.resolution_func = self._charged_hadron_cms_resolution
        elif resolution_formula == 'electron_cms':
            self.resolution_func = self._electron_cms_resolution
        elif resolution_formula == 'muon_cms':
            self.resolution_func = self._muon_cms_resolution
        elif callable(resolution_formula):
            self.resolution_func = resolution_formula
        else:
            raise ValueError(f"Unknown resolution formula: {resolution_formula}")
    
    @staticmethod
    def _charged_hadron_cms_resolution(pt, eta):
        """
        CMS charged hadron momentum resolution formula.
        Based on arXiv:1405.6569
        
        Args:
            pt: transverse momentum (GeV)
            eta: pseudorapidity
            
        Returns:
            resolution: absolute momentum resolution (GeV)
        """
        abs_eta = torch.abs(eta)
        
        # Initialize with zeros
        res = torch.zeros_like(pt)
        
        # Region 1: Central barrel (|eta| <= 0.5, pt > 0.1)
        # Resolution = sqrt(0.06^2 + pt^2 * 1.3e-3^2)
        mask1 = (abs_eta <= 0.5) & (pt > 0.1)
        res1 = torch.sqrt(0.06**2 + pt**2 * (1.3e-3)**2)
        res = torch.where(mask1, res1, res)
        
        # Region 2: Intermediate (0.5 < |eta| <= 1.5, pt > 0.1)
        # Resolution = sqrt(0.10^2 + pt^2 * 1.7e-3^2)
        mask2 = (abs_eta > 0.5) & (abs_eta <= 1.5) & (pt > 0.1)
        res2 = torch.sqrt(0.10**2 + pt**2 * (1.7e-3)**2)
        res = torch.where(mask2, res2, res)
        
        # Region 3: Forward (1.5 < |eta| <= 2.5, pt > 0.1)
        # Resolution = sqrt(0.25^2 + pt^2 * 3.1e-3^2)
        mask3 = (abs_eta > 1.5) & (abs_eta <= 2.5) & (pt > 0.1)
        res3 = torch.sqrt(0.25**2 + pt**2 * (3.1e-3)**2)
        res = torch.where(mask3, res3, res)
        
        return res
    
    @staticmethod
    def _electron_cms_resolution(pt, eta):
        """
        CMS electron momentum resolution formula.
        Based on arXiv:1502.02701
        
        Args:
            pt: transverse momentum (GeV)
            eta: pseudorapidity
            
        Returns:
            resolution: absolute momentum resolution (GeV)
        """
        abs_eta = torch.abs(eta)
        res = torch.zeros_like(pt)
        
        # Central barrel
        mask1 = (abs_eta <= 0.5) & (pt > 0.1)
        res1 = torch.sqrt(0.03**2 + pt**2 * (1.3e-3)**2)
        res = torch.where(mask1, res1, res)
        
        # Intermediate
        mask2 = (abs_eta > 0.5) & (abs_eta <= 1.5) & (pt > 0.1)
        res2 = torch.sqrt(0.05**2 + pt**2 * (1.7e-3)**2)
        res = torch.where(mask2, res2, res)
        
        # Forward
        mask3 = (abs_eta > 1.5) & (abs_eta <= 2.5) & (pt > 0.1)
        res3 = torch.sqrt(0.15**2 + pt**2 * (3.1e-3)**2)
        res = torch.where(mask3, res3, res)
        
        return res
    
    @staticmethod
    def _muon_cms_resolution(pt, eta):
        """
        CMS muon momentum resolution formula.
        Based on arXiv:1306.2016
        
        Args:
            pt: transverse momentum (GeV)
            eta: pseudorapidity
            
        Returns:
            resolution: absolute momentum resolution (GeV)
        """
        abs_eta = torch.abs(eta)
        res = torch.zeros_like(pt)
        
        # Central barrel
        mask1 = (abs_eta <= 0.5) & (pt > 0.1)
        res1 = torch.sqrt(0.01**2 + pt**2 * (1.0e-3)**2)
        res = torch.where(mask1, res1, res)
        
        # Intermediate
        mask2 = (abs_eta > 0.5) & (abs_eta <= 1.5) & (pt > 0.1)
        res2 = torch.sqrt(0.02**2 + pt**2 * (1.3e-3)**2)
        res = torch.where(mask2, res2, res)
        
        # Forward
        mask3 = (abs_eta > 1.5) & (abs_eta <= 2.5) & (pt > 0.1)
        res3 = torch.sqrt(0.10**2 + pt**2 * (2.0e-3)**2)
        res = torch.where(mask3, res3, res)
        
        return res
    
    @staticmethod
    def log_normal_sample(mean, sigma):
        """
        Sample from a log-normal distribution.
        
        This ensures the smeared PT is always positive.
        
        Args:
            mean: mean value (original PT)
            sigma: standard deviation (resolution)
            
        Returns:
            sample: value from log-normal distribution
        """
        # Handle edge cases
        mask_positive = mean > 0.0
        
        # For positive means, compute log-normal parameters
        # Variance = sigma^2, Mean = mean
        # Then: ln(mean) = mu + 0.5*s^2 and sigma^2 = mean^2 * (exp(s^2) - 1)
        # Solving: s^2 = ln(1 + (sigma/mean)^2), mu = ln(mean) - 0.5*s^2
        
        s_squared = torch.log(1.0 + (sigma / (mean + 1e-10))**2)
        s = torch.sqrt(s_squared)
        mu = torch.log(mean + 1e-10) - 0.5 * s_squared
        
        # Sample from standard normal and transform
        z = torch.randn_like(mean)
        sample = torch.exp(mu + s * z)
        
        # Return mean for non-positive cases
        result = torch.where(mask_positive, sample, mean)
        
        return result
    
    def forward(self, particles):
        """
        Apply momentum smearing to particles.
        
        Args:
            particles: tensor of shape (N, 15) or (B, N, 16)
                If (N, 15): single event
                If (B, N, 16): batched events with mask in column 15
                
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
                column 14: Mass
                column 15 (optional): Mask (1.0 = real particle, 0.0 = padding)
        
        Returns:
            smeared_particles: particles with smeared PT
        """
        # Move to device
        particles = particles.to(self.device)

        # Clone particles to avoid modifying input
        smeared = particles.clone()
        
        # Extract mask
        mask = particles[..., CMAP["IS_NOT_PAD"]]  # Works for both (N, 16) and (B, N, 16)
        
        # Extract pre-computed kinematics from Delphes
        pt = particles[..., CMAP["PT"]]   # Column 7: PT (transverse momentum)
        eta = particles[..., CMAP["ETA"]]  # Column 8: Eta (for resolution formula - typically position-based)
        phi = particles[..., CMAP["PHI"]]  # Column 9: Phi (azimuthal angle)
        mass = particles[..., CMAP["MASS"]]  # Column 14: Mass

        # Extract momentum components to compute momentum-based eta for reconstruction
        px = particles[..., CMAP["PX"]]  # Px
        py = particles[..., CMAP["PY"]]  # Py
        pz = particles[..., CMAP["PZ"]]  # Pz

        # Compute momentum-based eta and phi from the momentum vector
        # This matches C++ Delphes which uses candidateMomentum.Eta() and candidateMomentum.Phi()
        # for reconstructing the 4-vector after smearing
        momentum_eta = torch.asinh(pz / (pt + 1e-10))  # atanh(pz/p) = asinh(pz/pt)
        momentum_phi = torch.atan2(py, px)
        
        # Compute resolution for each particle using the eta from column 8
        # (which may be position-based depending on configuration)
        resolution = self.resolution_func(pt, eta)
        
        # Clamp resolution to maximum of 1.0 (100% of PT)
        resolution = torch.clamp(resolution, max=1.0)
        
        # Apply smearing
        smeared_pt = self.log_normal_sample(pt, resolution)
        
        # Apply smearing only to real particles (mask == 1.0)
        # Keep original PT for padded particles (mask == 0.0)
        smeared_pt = torch.where(mask > 0.5, smeared_pt, pt)
        
        # Update PT in the tensor (column 7)
        smeared[..., CMAP["PT"]] = smeared_pt

        # Recompute Px, Py, Pz with smeared PT but using MOMENTUM-based eta and phi
        # This matches C++ Delphes behavior: it uses candidateMomentum.Eta() and Phi()
        # for reconstruction, not the position-based values used for resolution
        smeared[..., CMAP["PX"]] = smeared_pt * torch.cos(momentum_phi)  # Px
        smeared[..., CMAP["PY"]] = smeared_pt * torch.sin(momentum_phi)  # Py
        smeared[..., CMAP["PZ"]] = smeared_pt * torch.sinh(momentum_eta)  # Pz

        # Recompute energy: E = sqrt(P^2 + M^2)
        p_squared = smeared[..., CMAP["PX"]]**2 + smeared[..., CMAP["PY"]]**2 + smeared[..., CMAP["PZ"]]**2
        smeared[..., CMAP["E"]] = torch.sqrt(p_squared + mass**2)  # E

        return smeared
    
    def get_resolution_map(self, pt_range=(0, 100), eta_range=(-3, 3), 
                          n_pts=100, n_etas=100):
        """
        Generate a 2D map of resolution values for visualization.
        
        Args:
            pt_range: (min, max) PT range in GeV
            eta_range: (min, max) eta range
            n_pts: number of PT bins
            n_etas: number of eta bins
            
        Returns:
            pt_grid: PT values (n_pts, n_etas)
            eta_grid: Eta values (n_pts, n_etas)
            resolution_map: Resolution values (n_pts, n_etas)
        """
        pt_vals = torch.linspace(pt_range[0], pt_range[1], n_pts, device=self.device)
        eta_vals = torch.linspace(eta_range[0], eta_range[1], n_etas, device=self.device)
        
        pt_grid, eta_grid = torch.meshgrid(pt_vals, eta_vals, indexing='ij')
        
        resolution_map = self.resolution_func(pt_grid, eta_grid)
        
        return pt_grid.cpu().numpy(), eta_grid.cpu().numpy(), resolution_map.cpu().numpy()


if __name__ == "__main__":
    # Test the module
    print("Testing DelphesMomentumSmearing module...")
    
    # Create module
    smearing = MomentumSmearing(resolution_formula='charged_hadron_cms')
    
    # Create test particles (N=5, features=15)
    # Columns: PID, Status, Charge, E, Px, Py, Pz, PT, Eta, Phi, T, X, Y, Z, Mass
    test_particles = torch.zeros(5, 15)
    test_particles[:, CMAP["PID"]] = torch.tensor([211, -211, 211, -211, 211])  # Pion PIDs
    test_particles[:, CMAP["CHARGE"]] = torch.tensor([1, -1, 1, -1, 1])  # Charges
    test_particles[:, CMAP["PT"]] = torch.tensor([0.5, 1.0, 5.0, 10.0, 50.0])  # PT
    test_particles[:, CMAP["ETA"]] = torch.tensor([0.3, 0.3, 1.0, 2.0, 0.5])  # Eta
    test_particles[:, CMAP["PHI"]] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])  # Phi
    test_particles[:, CMAP["MASS"]] = 0.140  # Pion mass
    
    # Compute Px, Py, Pz, E from PT, Eta, Phi, Mass
    test_particles[:, CMAP["PX"]] = test_particles[:, CMAP["PT"]] * torch.cos(test_particles[:, CMAP["PHI"]])  # Px
    test_particles[:, CMAP["PY"]] = test_particles[:, CMAP["PT"]] * torch.sin(test_particles[:, CMAP["PHI"]])  # Py
    test_particles[:, CMAP["PZ"]] = test_particles[:, CMAP["PT"]] * torch.sinh(test_particles[:, CMAP["ETA"]])  # Pz
    p_squared = test_particles[:, CMAP["PX"]]**2 + test_particles[:, CMAP["PY"]]**2 + test_particles[:, CMAP["PZ"]]**2
    test_particles[:, CMAP["E"]] = torch.sqrt(p_squared + test_particles[:, CMAP["MASS"]]**2)  # E

    print("\nOriginal particles:")
    print(f"PT:  {test_particles[:, CMAP["PT"]]}")
    print(f"Eta: {test_particles[:, CMAP["ETA"]]}")
    
    # Apply smearing
    smeared_particles, resolutions = smearing(test_particles, return_resolution=True)
    
    print("\nResolutions:")
    print(f"{resolutions}")
    
    print("\nSmeared particles:")
    print(f"PT:  {smeared_particles[:, CMAP["PT"]]}")
    print(f"Eta: {smeared_particles[:, CMAP["ETA"]]}")  # Should be unchanged

    print("\nRelative resolution (sigma/PT):")
    print(f"{resolutions / test_particles[:, CMAP["PT"]]}")
    
    print("\n✓ Test complete!")
