"""
PyTorch implementation of Delphes MomentumSmearing module.

Performs transverse momentum resolution smearing using a log-normal distribution.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Union, Tuple

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP
from parnassus.torch_delphes.stochastic_utils import log_normal_sample

class MomentumSmearing(nn.Module):
    """
    PyTorch implementation of Delphes MomentumSmearing module.
    
    Applies momentum resolution smearing to particles based on their kinematics (pt, eta_outer).
    The smearing is applied using a log-normal distribution to ensure positive PT values.
    
    Input shape: (N, N_FEATURES) where:
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
        - column 15: EtaOuter (pseudorapidity at outer position)
        - column 16: PhiOuter (azimuthal angle at closest approach to z-axis)
        - columns 17->23: masks
    
    NOTE: Position-Eta is used for evaluating the resolution formula, but Momentum-Eta is updated

    The resolution formula from CMS card (for charged hadrons):
        (abs(eta_outer) <= 0.5) * (pt > 0.1) * sqrt(0.06^2 + pt^2*1.3e-3^2) +
        (abs(eta_outer) > 0.5 && abs(eta_outer) <= 1.5) * (pt > 0.1) * sqrt(0.10^2 + pt^2*1.7e-3^2) +
        (abs(eta_outer) > 1.5 && abs(eta_outer) <= 2.5) * (pt > 0.1) * sqrt(0.25^2 + pt^2*3.1e-3^2)
    """
    
    def __init__(
        self, 
        resolution_formula: Union[str, Callable] = 'charged_hadron_cms',
    ) -> None:
        """
        Args:
            resolution_formula: Name of predefined formula or custom callable
            deterministic: If True, no smearing (for testing)
        """
        super().__init__()
        
        # Load resolution formula
        if resolution_formula == 'charged_hadron_cms':
            self.resolution_func = self._charged_hadron_cms_momentum_resolution
        elif resolution_formula == 'electron_cms':
            self.resolution_func = self._electron_cms_momentum_resolution
        elif resolution_formula == 'muon_cms':
            self.resolution_func = self._muon_cms_momentum_resolution
        elif callable(resolution_formula):
            self.resolution_func = resolution_formula
        else:
            raise ValueError(f"Unknown resolution formula: {resolution_formula}")
    
    def forward(self, particles: torch.Tensor) -> torch.Tensor:
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
                column 15: etaOuter (pseudorapidity at outer position)
                column 16: phiOuter (azimuthal angle at outer position)
                columns 17->23: masks
        
        Returns:
            smeared_particles: particles with smeared PT
        """

        pt = particles[:, CMAP["PT"]]   # Column 7: PT (transverse momentum)
        eta_outer = particles[:, CMAP["ETA_OUTER"]]  # Column 8: Eta (for resolution formula - typically position-based)
        mass = particles[:, CMAP["MASS"]]  # Column 14: Mass

        eta = particles[:, CMAP["ETA"]]  # atanh(pz/p) = asinh(pz/pt)
        phi = particles[:, CMAP["PHI"]]

        # Compute resolution for each particle using eta_outer
        resolution = self.resolution_func(pt, eta_outer)
        resolution = torch.clamp(resolution, max=1.0)
        
        # Apply smearing using log-normal distribution
        smeared_pt = log_normal_sample(pt, resolution)
        
        # Update PT, PX, PY, PZ, E
        particles[:, CMAP["PT"]] = smeared_pt
        particles[:, CMAP["PX"]] = smeared_pt * torch.cos(phi)  # Px
        particles[:, CMAP["PY"]] = smeared_pt * torch.sin(phi)  # Py
        particles[:, CMAP["PZ"]] = smeared_pt * torch.sinh(eta)  # Pz

        p_squared = particles[:, CMAP["PX"]]**2 + particles[:, CMAP["PY"]]**2 + particles[:, CMAP["PZ"]]**2
        particles[:, CMAP["E"]] = torch.sqrt(p_squared + mass**2)  # E

        return particles

    @staticmethod
    def _charged_hadron_cms_momentum_resolution(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """
        CMS charged hadron momentum resolution formula.
        Based on arXiv:1405.6569
        
        Args:
            pt: transverse momentum (GeV)
            eta_outer: pseudorapidity
            
        Returns:
            resolution: absolute momentum resolution (GeV)
        """
        abs_eta_outer = torch.abs(eta_outer)
        
        # Initialize with zeros
        res = torch.zeros_like(pt)
        
        # Region 1: Central barrel (|eta_outer| <= 0.5, pt > 0.1)
        # Resolution = sqrt(0.06^2 + pt^2 * 1.3e-3^2)
        mask1 = (abs_eta_outer <= 0.5) & (pt > 0.1)
        res1 = torch.sqrt(0.06**2 + pt**2 * (1.3e-3)**2)
        res = torch.where(mask1, res1, res)
        
        # Region 2: Intermediate (0.5 < |eta_outer| <= 1.5, pt > 0.1)
        # Resolution = sqrt(0.10^2 + pt^2 * 1.7e-3^2)
        mask2 = (abs_eta_outer > 0.5) & (abs_eta_outer <= 1.5) & (pt > 0.1)
        res2 = torch.sqrt(0.10**2 + pt**2 * (1.7e-3)**2)
        res = torch.where(mask2, res2, res)
        
        # Region 3: Forward (1.5 < |eta_outer| <= 2.5, pt > 0.1)
        # Resolution = sqrt(0.25^2 + pt^2 * 3.1e-3^2)
        mask3 = (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5) & (pt > 0.1)
        res3 = torch.sqrt(0.25**2 + pt**2 * (3.1e-3)**2)
        res = torch.where(mask3, res3, res)
        
        return res
    
    @staticmethod
    def _electron_cms_momentum_resolution(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """
        CMS electron momentum resolution formula.
        Based on arXiv:1502.02701
        
        Args:
            pt: transverse momentum (GeV)
            eta_outer: pseudorapidity
            
        Returns:
            resolution: absolute momentum resolution (GeV)
        """
        abs_eta_outer = torch.abs(eta_outer)
        res = torch.zeros_like(pt)
        
        # Central barrel
        mask1 = (abs_eta_outer <= 0.5) & (pt > 0.1)
        res1 = torch.sqrt(0.03**2 + pt**2 * (1.3e-3)**2)
        res = torch.where(mask1, res1, res)
        
        # Intermediate
        mask2 = (abs_eta_outer > 0.5) & (abs_eta_outer <= 1.5) & (pt > 0.1)
        res2 = torch.sqrt(0.05**2 + pt**2 * (1.7e-3)**2)
        res = torch.where(mask2, res2, res)
        
        # Forward
        mask3 = (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5) & (pt > 0.1)
        res3 = torch.sqrt(0.15**2 + pt**2 * (3.1e-3)**2)
        res = torch.where(mask3, res3, res)
        
        return res
    
    @staticmethod
    def _muon_cms_momentum_resolution(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """
        CMS muon momentum resolution formula.
        Based on arXiv:1306.2016
        
        Args:
            pt: transverse momentum (GeV)
            eta_outer: pseudorapidity
            
        Returns:
            resolution: absolute momentum resolution (GeV)
        """
        abs_eta_outer = torch.abs(eta_outer)
        res = torch.zeros_like(pt)
        
        # Central barrel
        mask1 = (abs_eta_outer <= 0.5) & (pt > 0.1)
        res1 = torch.sqrt(0.01**2 + pt**2 * (1.0e-3)**2)
        res = torch.where(mask1, res1, res)
        
        # Intermediate
        mask2 = (abs_eta_outer > 0.5) & (abs_eta_outer <= 1.5) & (pt > 0.1)
        res2 = torch.sqrt(0.02**2 + pt**2 * (1.3e-3)**2)
        res = torch.where(mask2, res2, res)
        
        # Forward
        mask3 = (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5) & (pt > 0.1)
        res3 = torch.sqrt(0.10**2 + pt**2 * (2.0e-3)**2)
        res = torch.where(mask3, res3, res)
        
        return res

    def get_resolution_map(
        self, 
        pt_range: Tuple[float, float] = (0, 100), 
        eta_range: Tuple[float, float] = (-3, 3), 
        n_pts: int = 100, 
        n_etas: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        pt_vals = torch.linspace(pt_range[0], pt_range[1], n_pts, device='cpu')
        eta_vals = torch.linspace(eta_range[0], eta_range[1], n_etas, device='cpu')

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
