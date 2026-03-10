"""
PyTorch implementation of Delphes Efficiency module.

Implements the ChargedHadronTrackingEfficiency from delphes_card_CMS.tcl
as a differentiable PyTorch module.
"""


import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Union, Tuple, Optional

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP
from parnassus.torch_delphes import pdg_filters

#TODO: Update docstrings

class Efficiency(nn.Module):
    """
    PyTorch implementation of Delphes Efficiency module.
    
    Applies tracking efficiency based on particle kinematics (pt, eta_outer)
    similar to Delphes ChargedHadronTrackingEfficiency.
    
    Input shape: (N, N_FEATURES) where:
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
        - column 15: etaOuter (pseudorapidity at outer position)
        - column 16: phiOuter (azimuthal angle at outer position)
        - columns 17->23: masks
    
    The efficiency formula from CMS card:
        (pt <= 0.1)   * (0.00) +
        (abs(eta_outer) <= 1.5) * (pt > 0.1 && pt <= 1.0)   * (0.70) +
        (abs(eta_outer) <= 1.5) * (pt > 1.0)                * (0.95) +
        (abs(eta_outer) > 1.5 && abs(eta_outer) <= 2.5) * (pt > 0.1 && pt <= 1.0)   * (0.60) +
        (abs(eta_outer) > 1.5 && abs(eta_outer) <= 2.5) * (pt > 1.0)                * (0.85) +
        (abs(eta_outer) > 2.5)                                                * (0.00)
    """
    
    def __init__(
        self, 
        efficiency_formula: Union[str, Callable] = 'charged_hadron_cms',
    ) -> None:
        """
        Args:
            efficiency_formula: Name of predefined formula or custom callable
        """
        super().__init__()
        self.efficiency_formula = efficiency_formula
        
        # Load efficiency formula
        if self.efficiency_formula == 'charged_hadron_cms':
            self.efficiency_func = self._charged_hadron_cms_efficiency
            self.pdg_filter_func = pdg_filters.charged_hadron_filter
        elif self.efficiency_formula == 'electron_cms':
            self.efficiency_func = self._electron_cms_efficiency
            self.pdg_filter_func = pdg_filters.electron_filter
        elif self.efficiency_formula == 'muon_cms':
            self.efficiency_func = self._muon_cms_efficiency
            self.pdg_filter_func = pdg_filters.muon_filter
        elif callable(self.efficiency_formula):
            self.efficiency_func = self.efficiency_formula
            self.pdg_filter_func = None
        else:
            raise ValueError(f"Unknown efficiency formula: {efficiency_formula}")
    

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
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
                column 15: etaOuter (pseudorapidity at outer position)
                column 16: phiOuter (azimuthal angle at outer position)
                columns 17->23: masks

        Returns:
            filtered_particles: tensor with mask in column 15
                               Single event: (N, 16) with mask
                               Batched: (B, N, 16) with updated mask
        """
    
        # We want to compute effiency vector based on particles that satisfy:
            # 1. particles that passed propagation (PASS_PROP == 1)
            # 2. particles of the desired type

        pt = particles[:, CMAP["PT"]]   # PT (transverse momentum)
        eta_outer = particles[:, CMAP["ETA_OUTER"]]  # EtaOuter (pseudorapidity at outer position)

        # Compute efficiency for each particle
        efficiency = self.efficiency_func(pt, eta_outer)
        
        # Apply efficiency stochastically
        passed = torch.rand_like(efficiency) < efficiency
        
        # Only real particles (mask==1) can pass efficiency
        particles_out = particles[passed]
        
        return particles_out
    
    @staticmethod
    def _charged_hadron_cms_efficiency(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """
        CMS charged hadron tracking efficiency formula.
        
        Args:
            pt: transverse momentum (GeV)
            eta_outer: pseudorapidity
            
        Returns:
            efficiency: value between 0 and 1
        """
        abs_eta_outer = torch.abs(eta_outer)
        
        # Initialize with zeros
        eff = torch.zeros_like(pt)
        
        # Region 1: Central barrel, low pt (0.1 < pt <= 1.0, |eta_outer| <= 1.5)
        mask1 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask1, torch.tensor(0.70, device=pt.device), eff)
        
        # Region 2: Central barrel, high pt (pt > 1.0, |eta_outer| <= 1.5)
        mask2 = (pt > 1.0) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask2, torch.tensor(0.95, device=pt.device), eff)
        
        # Region 3: Forward endcap, low pt (0.1 < pt <= 1.0, 1.5 < |eta_outer| <= 2.5)
        mask3 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask3, torch.tensor(0.60, device=pt.device), eff)
        
        # Region 4: Forward endcap, high pt (pt > 1.0, 1.5 < |eta_outer| <= 2.5)
        mask4 = (pt > 1.0) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask4, torch.tensor(0.85, device=pt.device), eff)
        
        # Region 5: Very forward (|eta_outer| > 2.5) - efficiency is 0 (already initialized)
        
        return eff
    
    @staticmethod
    def _electron_cms_efficiency(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """CMS electron tracking efficiency formula."""
        abs_eta_outer = torch.abs(eta_outer)
        eff = torch.zeros_like(pt)
        
        # Central barrel
        mask1 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask1, torch.tensor(0.73, device=pt.device), eff)
        
        mask2 = (pt > 1.0) & (pt <= 1.0e2) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask2, torch.tensor(0.95, device=pt.device), eff)
        
        mask3 = (pt > 1.0e2) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask3, torch.tensor(0.99, device=pt.device), eff)
        
        # Forward endcap
        mask4 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask4, torch.tensor(0.50, device=pt.device), eff)
        
        mask5 = (pt > 1.0) & (pt <= 1.0e2) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask5, torch.tensor(0.83, device=pt.device), eff)
        
        mask6 = (pt > 1.0e2) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask6, torch.tensor(0.90, device=pt.device), eff)
        
        return eff
    
    @staticmethod
    def _muon_cms_efficiency(pt: torch.Tensor, eta_outer: torch.Tensor) -> torch.Tensor:
        """CMS muon tracking efficiency formula."""
        abs_eta_outer = torch.abs(eta_outer)
        eff = torch.zeros_like(pt)
        
        # Central barrel
        mask1 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask1, torch.tensor(0.75, device=pt.device), eff)
        
        mask2 = (pt > 1.0) & (pt <= 1.0e3) & (abs_eta_outer <= 1.5)
        eff = torch.where(mask2, torch.tensor(0.99, device=pt.device), eff)
        
        mask3 = (pt > 1.0e3) & (abs_eta_outer <= 1.5)
        eff3 = 0.99 * torch.exp(0.5 - pt * 5.0e-4)
        eff = torch.where(mask3, eff3, eff)
        
        # Forward endcap
        mask4 = (pt > 0.1) & (pt <= 1.0) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask4, torch.tensor(0.70, device=pt.device), eff)
        
        mask5 = (pt > 1.0) & (pt <= 1.0e3) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff = torch.where(mask5, torch.tensor(0.98, device=pt.device), eff)
        
        mask6 = (pt > 1.0e3) & (abs_eta_outer > 1.5) & (abs_eta_outer <= 2.5)
        eff6 = 0.98 * torch.exp(0.5 - pt * 5.0e-4)
        eff = torch.where(mask6, eff6, eff)
        
        return eff
