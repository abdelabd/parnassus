import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Callable, Union, Tuple, Optional

from parnassus.torch_delphes.tensor_utils import COLUMN_MAP as CMAP
from parnassus.torch_delphes import pdg_filters
from parnassus.torch_delphes.ParticlePropagator import ParticlePropagator
from parnassus.torch_delphes.Efficiency import Efficiency
from parnassus.torch_delphes.MomentumSmearing import MomentumSmearing
from parnassus.torch_delphes.SimpleCalorimeter import SimpleCalorimeter
from parnassus.torch_delphes.Merger import Merger
from parnassus.torch_delphes.EFlowMerger import EFlowMerger

#TODO: Update docstrings

class ATLASEnergyFlowDefault(nn.Module):
    """
    PyTorch implementation of a default CMS Delphes card.
    
    Combines multiple modules to simulate the full detector response:
        - ParticlePropagator: Propagates particles through the magnetic field
        - Efficiency: Applies tracking efficiency based on kinematics
        - MomentumSmearing: Smears momentum measurements
        - SimpleCalorimeter: Simulates calorimeter response
        - Merger: Merges particles into jets
        - EFlowMerger: Merges charged and neutral particles for eflow reconstruction
    """
    
    def __init__(self, 
                 debug: bool = False) -> None:
        super().__init__()
        self.debug = debug

        self.params = {}

        # ParticlePropagator
        self.ParticlePropagator = ParticlePropagator(
            radius=1.15,
            half_length=3.51,
            bz=2.0,
        )

        # TrackingEfficiency
        self.ChargedHadronTrackingEfficiency = Efficiency(
            efficiency_formula = 'charged_hadron_cms'
        )
        self.ElectronTrackingEfficiency = Efficiency(
            efficiency_formula = 'electron_cms'
        )
        self.MuonTrackingEfficiency = Efficiency(
            efficiency_formula = 'muon_cms'
        )

        # MomentumSmearing
        self.ChargedHadronMomentumSmearing = MomentumSmearing(
            resolution_formula = 'charged_hadron_cms'
        )
        self.ElectronMomentumSmearing = MomentumSmearing(
            resolution_formula = 'electron_cms'
        )
        self.MuonMomentumSmearing = MomentumSmearing(
            resolution_formula = 'muon_cms'
        )

        # TrackMerger
        self.TrackMerger = Merger()

        # ECal (Electromagnetic Calorimeter)
        self._setup_ECal()

        # HCal (Hadronic Calorimeter)
        self._setup_HCal()

        # CalorimeterMerger
        self.CalorimeterMerger = Merger()

        # EFlowMerger
        self.EFlowMerger = EFlowMerger()
        

    def forward(self, stable_particles: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply the CMS Delphes simulation (up to EnergyFlow objects) to the input generator-level event tensors.
        
        Args:
            stable_particles: Tensor of shape (N_events, N_particles, N_FEATURES) containing generator-level particles
        
        Returns:
            output: Dictionary containing reconstructed particle tensors, e.g.:
                {
                    'MergedTower': Tensor of shape (N_events, N_ecal_towers+N_hcal_towers, N_FEATURES) - reconstructed tracks,
                    'EFlowObject': Tensor of shape (N_events, N_eflow_objects, N_FEATURES) - reconstructed EnergyFlow candidates,
                }
        """
        n_part, n_dim = stable_particles.shape
        
        # ParticlePropagator
        particles = stable_particles.reshape(-1, n_dim)
        if self.debug:
            particles_before_prop = particles.clone()
        particles_propagated, neutrals_propagated, charged_hadrons_propagated, electrons_propagated, muons_propagated = self.ParticlePropagator(particles)

        # TrackingEfficiency
        charged_hadrons_eff = self.ChargedHadronTrackingEfficiency(charged_hadrons_propagated)
        electrons_eff = self.ElectronTrackingEfficiency(electrons_propagated)
        muons_eff = self.MuonTrackingEfficiency(muons_propagated)

        # MomentumSmearing
        charged_hadrons_smeared = self.ChargedHadronMomentumSmearing(charged_hadrons_eff)
        electrons_smeared = self.ElectronMomentumSmearing(electrons_eff)
        muons_smeared = self.MuonMomentumSmearing(muons_eff)

        # TrackMerger
        merged_tracks = self.TrackMerger([charged_hadrons_smeared, electrons_smeared, muons_smeared])

        # ECal
        ecal_tracks, ecal_towers, eflow_photons = self.ECal(particles_propagated, merged_tracks)

        # HCal
        hcal_tracks, hcal_towers, eflow_neutral_hadrons = self.HCal(particles_propagated, ecal_tracks)

        # CalorimeterMerger
        merged_towers = self.CalorimeterMerger([ecal_towers, hcal_towers])

        # EFlowMerger
        eflow_objects = self.EFlowMerger([hcal_tracks, eflow_photons, eflow_neutral_hadrons])

        if self.debug:
            return {
                "ParticleBeforeProp": particles_before_prop,

                "ParticleAfterProp": particles_propagated,
                "ChargedHadron": charged_hadrons_propagated,
                "Electron": electrons_propagated,
                "Muon": muons_propagated,

                "ChargedHadronEfficiency": charged_hadrons_eff,
                "ElectronEfficiency": electrons_eff,
                "MuonEfficiency": muons_eff,

                "ChargedHadronSmeared": charged_hadrons_smeared,
                "ElectronSmeared": electrons_smeared,
                "MuonSmeared": muons_smeared,

                "Track": merged_tracks,

                "ECal_EFlowTrack": ecal_tracks,
                "ECalTower": ecal_towers,
                "EFlowPhoton": eflow_photons,

                "EFlowTrack": hcal_tracks,
                "HCalTower": hcal_towers,
                "EFlowNeutralHadron": eflow_neutral_hadrons,

                "Tower": merged_towers,

                "EFlowObject": eflow_objects,
                }
        else:
            return {
                "Track": merged_tracks,
                "Tower": merged_towers,
                "EFlowTrack": hcal_tracks,
                "EFlowPhoton": eflow_photons, 
                "EFlowNeutralHadron": eflow_neutral_hadrons,
            }

    def _setup_ECal(self):
        energy_fractions = {
            0: 0.0,        # default (hadrons) - no ECAL response
            11: 1.0,       # electrons
            22: 1.0,       # photons
            111: 1.0,      # pi0
            12: 0.0,       # neutrino (electron)
            13: 0.0,       # muon
            14: 0.0,       # neutrino (muon)
            16: 0.0,       # neutrino (tau)
            1000022: 0.0,  # neutralino
            1000023: 0.0,  # neutralino
            1000025: 0.0,  # neutralino
            1000035: 0.0,  # neutralino
            1000045: 0.0,  # neutralino
            310: 0.3,      # K0short
            3122: 0.3,     # Lambda
        }

        # Create eta and phi bins from CMS card (delphes_card_CMS_5_0.tcl)
        # The card builds a map: eta_value -> set of phi bins
        # We need to replicate this exactly
        
        # Fine phi bins for barrel and endcap (361 bins, -pi to pi in 1 degree steps)
        phi_bins_fine = [i * np.pi / 180.0 for i in range(-180, 181)]
        
        # Coarse phi bins for HF (37 bins, -pi to pi in 10 degree steps)
        phi_bins_coarse = [i * np.pi / 18.0 for i in range(-18, 19)]

        # Build the eta bins and corresponding phi bins exactly as C++ Delphes does
        # The C++ code uses a map<double, set<double>> which gets sorted by eta
        # Then converts to parallel vectors: fEtaBins and fPhiBins[etaBin]
        
        eta_phi_map = {}  # eta -> set of phi bin edges
        
        # Barrel: 0.02 unit in eta from -85*0.0174 to 86*0.0174
        for i in range(-85, 87):
            eta = i * 0.0174
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_fine)
        
        # Endcap negative: -2.958 + i*0.0174 for i in 1..84
        for i in range(1, 85):
            eta = -2.958 + i * 0.0174
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_fine)

        # Endcap positive: 1.4964 + i*0.0174 for i in 1..84
        for i in range(1, 85):
            eta = 1.4964 + i * 0.0174
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_fine)
        
        # HF: specific eta values with coarse phi binning
        hf_etas = [-5, -4.7, -4.525, -4.35, -4.175, -4, -3.825, -3.65, -3.475, -3.3, -3.125, -2.958,
                3.125, 3.3, 3.475, 3.65, 3.825, 4, 4.175, 4.35, 4.525, 4.7, 5]
        for eta in hf_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_coarse)

        # Convert to sorted lists (matching C++ behavior)
        eta_bins = sorted(eta_phi_map.keys())
        phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

        self.ECal = SimpleCalorimeter(
            eta_bins=eta_bins,
            phi_bins=phi_bins_per_eta,
            energy_min=0.5,
            energy_sig_min=2.0,
            energy_fractions=energy_fractions,
            resolution_formula='ecal_atlas',
            is_ecal=True,
            smear_tower_center=True  # Match C++ Delphes: SmearTowerCenter true
        )

    def _setup_HCal(self):
        energy_fractions = {
            0: 1.0,        # default (hadrons) - full HCAL response
            11: 0.0,       # electrons (no HCAL response - already absorbed by ECAL)
            22: 0.0,       # photons (no HCAL response)
            111: 0.0,      # pi0 (no HCAL response)
            12: 0.0,       # neutrino (electron)
            13: 0.0,       # muon
            14: 0.0,       # neutrino (muon)
            16: 0.0,       # neutrino (tau)
            1000022: 0.0,  # neutralino
            1000023: 0.0,  # neutralino
            1000025: 0.0,  # neutralino
            1000035: 0.0,  # neutralino
            1000045: 0.0,  # neutralino
            310: 0.7,      # K0short (70% HCAL)
            3122: 0.7,     # Lambda (70% HCAL)
        }
        
        
        eta_phi_map = {}  # eta -> set of phi bin edges
        
        # 5 degrees towers (barrel+endcap): phi bins -36 to 36 in steps of pi/36
        phi_bins_10deg = [i * np.pi / 18.0 for i in range(-18, 19)]
        barrel_etas = [-3.2, -2.5, -2.4, -2.3, -2.2, -2.1, -2, -1.9, -1.8, -1.7,
                        -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1, -0.9, -0.8, -0.7,
                          -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3,
                            0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3,
                              1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3,
                                2.4, 2.5, 2.6, 3.3]
        for eta in barrel_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_10deg)
        
        # 20 degrees towers (forward): phi bins -18 to 18 in steps of pi/18
        phi_bins_20deg = [i * np.pi / 9.0 for i in range(-9, 10)]
        endcap_etas = [-4.9, -4.7, -4.5, -4.3, -4.1, -3.9, -3.7, -3.5, -3.3, -3,
                        -2.8, -2.6, 2.8, 3, 3.2, 3.5, 3.7, 3.9, 4.1, 4.3,
                          4.5, 4.7, 4.9] 
        for eta in endcap_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_20deg)
        
        # Convert to sorted lists (matching C++ behavior)
        eta_bins = sorted(eta_phi_map.keys())
        phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

        self.HCal = SimpleCalorimeter(
            eta_bins=eta_bins,
            phi_bins=phi_bins_per_eta,
            energy_min=1.0,          # HCal has higher threshold
            energy_sig_min=2.0,      # HCal has lower significance threshold
            energy_fractions=energy_fractions,
            resolution_formula='hcal_atlas',
            is_ecal=False,
            smear_tower_center=True
        )

