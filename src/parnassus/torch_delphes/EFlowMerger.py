"""
EFlowMerger: Energy Flow Merger Module

This module merges Track and Tower objects into ParticleFlowCandidate objects.
It handles the necessary transformations to ensure consistency between Track and Tower
representations when they are merged together.

Based on the Delphes C++ Merger module, but with special handling for ParticleFlowCandidate
output format.

TODO: Rely on Eem/Ehad fields from SimpleCalorimeter
Currently, we infer Eem/Ehad during ROOT writing based on PID values.
If SimpleCalorimeter is updated to include Eem/Ehad in the tensor (see TODO in
SimpleCalorimeter.py), this module can preserve those values directly without
needing to recompute them. See EFlowMerger.md for validation details.
"""

import torch
import torch.nn as nn
from typing import List

from .tensor_utils import (
    PID, CHARGE, E, PX, PY, PZ, PT, ETA, PHI, T, X, Y, Z,
    ETA_OUTER, PHI_OUTER, EVENT_NUMBER, IS_NOT_PAD
)


class EFlowMerger(nn.Module):
    """
    EFlowMerger: Merges Track and Tower objects into ParticleFlowCandidate objects.

    This merger takes three input streams:
    1. Tracks (from HCal/eflowTracks): charged particles with track information
    2. Photons (from ECal/eflowPhotons): electromagnetic calorimeter towers
    3. Neutral Hadrons (from HCal/eflowNeutralHadrons): hadronic calorimeter towers

    Key transformations applied:
    - For Tracks: Eta field is set to EtaOuter (position eta) for consistency with ParticleFlow
    - For Photons: PID set to 22, X/Y/Z set to 0, Eem set to E, Ehad set to 0
    - For Neutral Hadrons: PID set to 0, X/Y/Z set to 0, Eem set to 0, Ehad set to E

    The output is a single tensor with all ParticleFlowCandidate objects.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_arrays: List[torch.Tensor]) -> torch.Tensor:
        """
        Merge Track and Tower objects into ParticleFlowCandidate objects.

        Args:
            input_arrays: List of 3 tensors:
                [0] tracks: (N_tracks, N_FEATURES) - Track objects from HCal/eflowTracks
                [1] photons: (N_photons, N_FEATURES) - Tower objects from ECal/eflowPhotons
                [2] neutral_hadrons: (N_neutrals, N_FEATURES) - Tower objects from HCal/eflowNeutralHadrons

        Returns:
            merged: (N_total, N_FEATURES) - ParticleFlowCandidate objects
        """
        if len(input_arrays) != 3:
            raise ValueError(f"EFlowMerger expects exactly 3 input arrays, got {len(input_arrays)}")

        tracks, photons, neutral_hadrons = input_arrays

        # Transform Track objects for ParticleFlow representation
        if tracks.shape[0] > 0:
            tracks = self._transform_tracks(tracks)

        # Transform Tower objects (photons) for ParticleFlow representation
        if photons.shape[0] > 0:
            photons = self._transform_photons(photons)

        # Transform Tower objects (neutral hadrons) for ParticleFlow representation
        if neutral_hadrons.shape[0] > 0:
            neutral_hadrons = self._transform_neutral_hadrons(neutral_hadrons)

        # Concatenate all objects
        all_objects = []
        if tracks.shape[0] > 0:
            all_objects.append(tracks)
        if photons.shape[0] > 0:
            all_objects.append(photons)
        if neutral_hadrons.shape[0] > 0:
            all_objects.append(neutral_hadrons)

        if len(all_objects) == 0:
            # No objects - return empty tensor
            return torch.zeros((0, tracks.shape[1] if tracks.shape[0] > 0 else photons.shape[1]),
                             dtype=tracks.dtype if tracks.shape[0] > 0 else photons.dtype,
                             device=tracks.device if tracks.shape[0] > 0 else photons.device)

        merged = torch.cat(all_objects, dim=0)

        return merged

    def _transform_tracks(self, tracks: torch.Tensor) -> torch.Tensor:
        """
        Transform Track objects for ParticleFlow representation.

        For ParticleFlow, the Eta field should be the position eta (EtaOuter),
        not the momentum eta. This ensures consistency with calorimeter towers.

        Args:
            tracks: (N, N_FEATURES) Track objects

        Returns:
            transformed: (N, N_FEATURES) Track objects with Eta set to EtaOuter
        """
        tracks = tracks.clone()

        # Set Eta to EtaOuter (position eta at calorimeter edge)
        # This is the key transformation for ParticleFlow consistency
        tracks[:, ETA] = tracks[:, ETA_OUTER]

        # Note: X, Y, Z should already be non-zero for tracks (from vertex position)
        # PID, Charge, E, PT, Phi, etc. remain unchanged

        return tracks

    def _transform_photons(self, photons: torch.Tensor) -> torch.Tensor:
        """
        Transform photon Tower objects for ParticleFlow representation.

        Photons are electromagnetic calorimeter deposits. For ParticleFlow:
        - PID should be 22 (photon PDG code)
        - X, Y, Z should be 0 (no vertex position)
        - Eem = E (all energy is electromagnetic)
        - Ehad = 0 (no hadronic energy)

        Args:
            photons: (N, N_FEATURES) Tower objects from ECal

        Returns:
            transformed: (N, N_FEATURES) Tower objects with proper PID and position
        """
        photons = photons.clone()

        # Set PID to 22 (photon)
        photons[:, PID] = 22

        # Set vertex position to zero (towers don't have vertex position)
        photons[:, X] = 0.0
        photons[:, Y] = 0.0
        photons[:, Z] = 0.0
        photons[:, T] = 0.0

        # Note: Eem and Ehad are not stored in the tensor, they will be computed
        # during ROOT writing based on E and PID

        return photons

    def _transform_neutral_hadrons(self, neutral_hadrons: torch.Tensor) -> torch.Tensor:
        """
        Transform neutral hadron Tower objects for ParticleFlow representation.

        Neutral hadrons are hadronic calorimeter deposits. For ParticleFlow:
        - PID should be 0 (neutral hadron - C++ Delphes convention)
        - X, Y, Z should be 0 (no vertex position)
        - Eem = 0 (no electromagnetic energy)
        - Ehad = E (all energy is hadronic)

        Args:
            neutral_hadrons: (N, N_FEATURES) Tower objects from HCal

        Returns:
            transformed: (N, N_FEATURES) Tower objects with proper PID and position
        """
        neutral_hadrons = neutral_hadrons.clone()

        # Set PID to 0 (neutral hadron - matches C++ Delphes convention)
        neutral_hadrons[:, PID] = 0

        # Set vertex position to zero (towers don't have vertex position)
        neutral_hadrons[:, X] = 0.0
        neutral_hadrons[:, Y] = 0.0
        neutral_hadrons[:, Z] = 0.0
        neutral_hadrons[:, T] = 0.0


        # Note: Eem and Ehad are not stored in the tensor, they will be computed
        # during ROOT writing based on E and PID

        return neutral_hadrons
