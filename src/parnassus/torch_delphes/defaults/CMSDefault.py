"""PyTorch implementation of default CMS detector simulation.

Implements the CMS detector response chain from the Delphes TCL card
(delphes_card_CMS_6_1.tcl), producing energy flow objects suitable for
jet clustering and physics analysis.

Detector parameters:
- Tracker radius: 1.29 m
- Tracker half-length: 3.0 m
- Magnetic field: 3.8 T

Processing chain:
1. ParticlePropagator → propagate particles to tracker surface
2. Efficiency → apply tracking efficiency (charged hadrons, electrons, muons)
3. MomentumSmearing → smear track momenta
4. TrackMerger → combine all tracks
5. ECal/HCal → calorimeter simulation with energy flow
6. EFlowMerger → combine tracks and calorimeter objects

Reference:
    C++ Delphes card: cards/delphes_card_CMS_6_1.tcl
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from parnassus.data.particle_io import ColumnMap
from parnassus.torch_delphes.Efficiency import Efficiency
from parnassus.torch_delphes.EFlowMerger import EFlowMerger
from parnassus.torch_delphes.learnable import (
    CMSChargedHadronLearnableEfficiency,
    CMSElectronLearnableEfficiency,
    CMSMuonLearnableEfficiency,
    LearnableEcalCMSResolution,
    LearnableHadronFractions,
    LearnableHcalCMSResolution,
    make_cms_ecal_scale,
    make_cms_hcal_scale,
    make_cms_track_resolution,
)
from parnassus.torch_delphes.Merger import Merger
from parnassus.torch_delphes.MomentumSmearing import MomentumSmearing
from parnassus.torch_delphes.ParticlePropagator import ParticlePropagator
from parnassus.torch_delphes.SimpleCalorimeter import SimpleCalorimeter

from .base import DelphesBaseCard


class CMSEnergyFlowDefault(DelphesBaseCard):
    """PyTorch implementation of the default CMS Delphes detector simulation.

    Simulates the full CMS detector response chain including:

    - **Tracking**: Particle propagation through 3.8T magnetic field,
      tracking efficiency, and momentum smearing
    - **Calorimetry**: ECal and HCal simulation with energy deposits,
      tower clustering, and energy resolution smearing
    - **Particle Flow**: Energy flow reconstruction combining tracks
      and calorimeter deposits

    The module can operate in two modes controlled by the `debug` flag:

    - **Normal mode** (debug=False): Returns only final reconstructed objects
      (Track, Tower, EFlowTrack, EFlowPhoton, EFlowNeutralHadron)
    - **Debug mode** (debug=True): Returns all intermediate objects for
      validation against C++ Delphes

    Attributes
    ----------
    debug: bool
        If True, return all intermediate processing stages
    ParticlePropagator: ParticlePropagator
        Propagates particles to tracker surface
    ChargedHadronTrackingEfficiency: Efficiency
        Tracking efficiency for hadrons
    ElectronTrackingEfficiency: Efficiency
        Tracking efficiency for electrons
    MuonTrackingEfficiency: Efficiency
        Tracking efficiency for muons
    ChargedHadronMomentumSmearing: MomentumSmearing
        Momentum resolution for hadrons
    ElectronMomentumSmearing: MomentumSmearing
        Momentum resolution for electrons
    MuonMomentumSmearing: MomentumSmearing
        Momentum resolution for muons
    TrackMerger: Merger
        Combines all track types
    ECal: SimpleCalorimeter
        Electromagnetic calorimeter
    HCal: SimpleCalorimeter
        Hadronic calorimeter
    CalorimeterMerger: Merger
        Combines ECal and HCal towers
    EFlowMerger: EFlowMerger
        Creates particle flow candidates

    Examples
    --------
    >>> cms = CMSEnergyFlowDefault(debug=False)
    >>> results = cms(stable_particles)
    >>> tracks = results['Track']
    >>> eflow_tracks = results['EFlowTrack']
    """

    def __init__(
        self,
        debug: bool = False,
        learnable: bool = False,
        gumbel_temperature: float = 0.5,
    ) -> None:
        """Initialize the CMS detector simulation.

        Parameters
        ----------
        debug: bool
            If True, return all intermediate processing stages
            for validation. If False, return only final objects.
        learnable: bool
            If True, replace the static resolution / efficiency / fraction
            constants with ``nn.Parameter``-backed learnable modules so that
            the entire detector card can be optimized end-to-end with
            standard PyTorch optimizers (Adam, etc.). All parameters are
            initialized to the same numerical defaults as the static
            formulas, so a freshly-constructed ``learnable=True`` card
            produces statistically-identical output (modulo Gumbel-ST
            sampling noise on the efficiency mask).
            See :mod:`parnassus.torch_delphes.learnable` for the parameter
            inventory.
        gumbel_temperature: float
            Only used when ``learnable=True``. Temperature for the
            Gumbel-sigmoid straight-through tracking efficiency mask. Lower
            values (e.g. 0.1) give sharper Bernoulli-like behavior; higher
            values (e.g. 1.0) give smoother gradients but more sampling
            noise. The training loop may anneal this value over time.
        """
        super().__init__()
        self.debug = debug
        self.learnable = learnable

        # Attribute-type declarations so mypy accepts the learnable / legacy
        # union. At runtime ``nn.Module.__setattr__`` registers whichever
        # concrete subclass is assigned below.
        self.ChargedHadronTrackingEfficiency: nn.Module
        self.ElectronTrackingEfficiency: nn.Module
        self.MuonTrackingEfficiency: nn.Module
        self.HadronFractions: LearnableHadronFractions | None

        # ParticlePropagator (geometry — not learnable per design)
        self.ParticlePropagator = ParticlePropagator(
            radius=1.29,
            half_length=3.0,
            bz=3.8,
        )

        # TrackingEfficiency
        if learnable:
            self.ChargedHadronTrackingEfficiency = CMSChargedHadronLearnableEfficiency(
                temperature=gumbel_temperature
            )
            self.ElectronTrackingEfficiency = CMSElectronLearnableEfficiency(
                temperature=gumbel_temperature
            )
            self.MuonTrackingEfficiency = CMSMuonLearnableEfficiency(temperature=gumbel_temperature)
        else:
            self.ChargedHadronTrackingEfficiency = Efficiency(
                efficiency_formula="charged_hadron_cms"
            )
            self.ElectronTrackingEfficiency = Efficiency(efficiency_formula="electron_cms")
            self.MuonTrackingEfficiency = Efficiency(efficiency_formula="muon_cms")

        # MomentumSmearing — wire learnable resolution + scale per species.
        if learnable:
            chad_res = make_cms_track_resolution("charged_hadron")
            ele_res = make_cms_track_resolution("electron")
            mu_res = make_cms_track_resolution("muon")
            self.ChargedHadronMomentumSmearing = MomentumSmearing(
                resolution_formula=chad_res, scale_fn=chad_res.scale
            )
            self.ElectronMomentumSmearing = MomentumSmearing(
                resolution_formula=ele_res, scale_fn=ele_res.scale
            )
            self.MuonMomentumSmearing = MomentumSmearing(
                resolution_formula=mu_res, scale_fn=mu_res.scale
            )
        else:
            self.ChargedHadronMomentumSmearing = MomentumSmearing(
                resolution_formula="charged_hadron_cms"
            )
            self.ElectronMomentumSmearing = MomentumSmearing(resolution_formula="electron_cms")
            self.MuonMomentumSmearing = MomentumSmearing(resolution_formula="muon_cms")

        # TrackMerger
        self.TrackMerger = Merger()

        # Shared learnable hadron-fraction module (only created when
        # learnable=True). Lives on the parent card so that ECal and HCal
        # see the *same* parameters and stay consistent.
        if learnable:
            self.HadronFractions = LearnableHadronFractions()
        else:
            self.HadronFractions = None

        # ECal (Electromagnetic Calorimeter)
        self._setup_ECal()

        # HCal (Hadronic Calorimeter)
        self._setup_HCal()

        # CalorimeterMerger
        self.CalorimeterMerger = Merger()

        # EFlowMerger
        self.EFlowMerger = EFlowMerger()

    def forward(self, stable_particles: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply the full CMS detector simulation to input particles.

        Processes generator-level stable particles through the complete
        detector simulation chain: propagation, tracking, calorimetry,
        and particle flow reconstruction.

        Parameters
        ----------
        stable_particles: torch.Tensor
            Tensor of shape (N, N_FEATURES) containing
            generator-level stable particles. Should be flattened
            (not batched by event). Required columns include:

            - PID, CHARGE, E, PX, PY, PZ, PT, ETA, PHI
            - X, Y, Z, T (production vertex)
            - MASS, EVENT_NUMBER

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping branch names to tensors. Contents depend on
            debug mode:

            **Normal mode** (debug=False):

            - 'Track': Merged tracks after smearing
            - 'Tower': Merged calorimeter towers
            - 'EFlowTrack': Tracks for particle flow
            - 'EFlowPhoton': Photon candidates from ECal
            - 'EFlowNeutralHadron': Neutral hadron candidates from HCal

            **Debug mode** (debug=True): All of the above plus:

            - 'ParticleBeforeProp', 'ParticleAfterProp'
            - 'ChargedHadron', 'Electron', 'Muon', 'NeutralParticle'
            - 'ChargedHadronEfficiency', 'ElectronEfficiency', 'MuonEfficiency'
            - 'ChargedHadronSmeared', 'ElectronSmeared', 'MuonSmeared'
            - 'ECal_EFlowTrack', 'ECalTower', 'HCalTower'
            - 'EFlowObject'
        """
        _, n_dim = stable_particles.shape

        # ParticlePropagator
        particles = stable_particles.reshape(-1, n_dim)
        particles_before_prop = particles.clone() if self.debug else torch.empty(0)

        (
            particles_propagated,
            neutrals_propagated,
            charged_hadrons_propagated,
            electrons_propagated,
            muons_propagated,
        ) = self.ParticlePropagator(particles)

        # TrackingEfficiency + MomentumSmearing.
        #
        # Legacy (non-learnable) order: filter-then-smear, matching C++ Delphes.
        # The Efficiency module drops rows; MomentumSmearing then operates on
        # the surviving subset and the bit-exact RNG sequence matches C++.
        #
        # Learnable order: smear-then-mask. The LearnableEfficiency multiplies
        # (PT, PX, PY, PZ, E) by a Gumbel-ST 0/1 mask but leaves the row in the
        # tensor so gradients flow. This must run AFTER MomentumSmearing,
        # because MomentumSmearing recomputes E from sqrt(p^2 + m^2) and would
        # otherwise resurrect masked particles' energy as their rest mass.
        if self.learnable:
            charged_hadrons_smeared_pre = self.ChargedHadronMomentumSmearing(
                charged_hadrons_propagated
            )
            electrons_smeared_pre = self.ElectronMomentumSmearing(electrons_propagated)
            muons_smeared_pre = self.MuonMomentumSmearing(muons_propagated)

            charged_hadrons_eff = self.ChargedHadronTrackingEfficiency(charged_hadrons_smeared_pre)
            electrons_eff = self.ElectronTrackingEfficiency(electrons_smeared_pre)
            muons_eff = self.MuonTrackingEfficiency(muons_smeared_pre)

            # Tag each track with its pre-reco tracking-efficiency region in the
            # EFF_REGION column (per-species global 1-based label; 0 = outside all
            # regions) BEFORE it enters the reco chain. EFlowTrack creation clones the
            # tensor and only rescales the momentum columns, so this label is carried
            # unchanged to the EFlowObject. The loss uses it to build the
            # per-(species, reco-bin) <- pre-reco-region migration for the
            # differentiable count terms -- the only gradient source for eff_logits
            # (the Gumbel mask above is detached). The label is taken from the
            # pre-mask smeared kinematics (same pt/eta the efficiency was evaluated
            # at); killed rows keep their label but are dropped later by valid=pt>0.
            charged_hadrons_eff = self._tag_eff_region(
                charged_hadrons_eff, charged_hadrons_smeared_pre, self.ChargedHadronTrackingEfficiency
            )
            electrons_eff = self._tag_eff_region(
                electrons_eff, electrons_smeared_pre, self.ElectronTrackingEfficiency
            )
            muons_eff = self._tag_eff_region(
                muons_eff, muons_smeared_pre, self.MuonTrackingEfficiency
            )

            charged_hadrons_smeared = charged_hadrons_eff
            electrons_smeared = electrons_eff
            muons_smeared = muons_eff
        else:
            charged_hadrons_eff = self.ChargedHadronTrackingEfficiency(charged_hadrons_propagated)
            electrons_eff = self.ElectronTrackingEfficiency(electrons_propagated)
            muons_eff = self.MuonTrackingEfficiency(muons_propagated)

            charged_hadrons_smeared = self.ChargedHadronMomentumSmearing(charged_hadrons_eff)
            electrons_smeared = self.ElectronMomentumSmearing(electrons_eff)
            muons_smeared = self.MuonMomentumSmearing(muons_eff)

        # TrackMerger
        merged_tracks = self.TrackMerger([
            charged_hadrons_smeared,
            electrons_smeared,
            muons_smeared,
        ])

        # ECal (4th return: differentiable per-region expected photon count; None
        # unless learnable)
        ecal_tracks, ecal_towers, eflow_photons, ecal_calo_counts = self.ECal(
            particles_propagated, merged_tracks
        )

        # HCal (4th return: per-region expected neutral-hadron count; None unless learnable)
        hcal_tracks, hcal_towers, eflow_neutral_hadrons, hcal_calo_counts = self.HCal(
            particles_propagated, ecal_tracks
        )

        # CalorimeterMerger
        merged_towers = self.CalorimeterMerger([ecal_towers, hcal_towers])

        # EFlowMerger
        eflow_objects = self.EFlowMerger([hcal_tracks, eflow_photons, eflow_neutral_hadrons])

        # Differentiable expected reconstructed track count per RECO bin, per track
        # species, built from the reco-bin <- pre-reco-region migration of the
        # trainee's own hard reco (see _expected_reco_counts). This is the gradient
        # source for the eff_logits. None in the legacy (non-learnable) path.
        if self.learnable:
            chad_expected_counts = self._expected_reco_counts(
                eflow_objects, self.ChargedHadronTrackingEfficiency
            )
            electron_expected_counts = self._expected_reco_counts(
                eflow_objects, self.ElectronTrackingEfficiency
            )
            muon_expected_counts = self._expected_reco_counts(
                eflow_objects, self.MuonTrackingEfficiency
            )
        else:
            chad_expected_counts = None
            electron_expected_counts = None
            muon_expected_counts = None

        if self.debug:
            return {
                "ParticleBeforeProp": particles_before_prop,
                "ParticleAfterProp": particles_propagated,
                "ChargedHadron": charged_hadrons_propagated,
                "Electron": electrons_propagated,
                "Muon": muons_propagated,
                "NeutralParticle": neutrals_propagated,
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
                "ChargedHadronExpectedCounts": chad_expected_counts,
                "ElectronExpectedCounts": electron_expected_counts,
                "MuonExpectedCounts": muon_expected_counts,
                "EcalPhotonExpectedCounts": ecal_calo_counts,
                "HcalNeutralHadronExpectedCounts": hcal_calo_counts,
            }
        return {
            "Track": merged_tracks,
            "Tower": merged_towers,
            "EFlowTrack": hcal_tracks,
            "EFlowPhoton": eflow_photons,
            "EFlowNeutralHadron": eflow_neutral_hadrons,
            "EFlowObject": eflow_objects,
            "ChargedHadronExpectedCounts": chad_expected_counts,
            "ElectronExpectedCounts": electron_expected_counts,
            "MuonExpectedCounts": muon_expected_counts,
            "EcalPhotonExpectedCounts": ecal_calo_counts,
            "HcalNeutralHadronExpectedCounts": hcal_calo_counts,
        }

    @staticmethod
    def _tag_eff_region(
        masked: torch.Tensor, pre_mask: torch.Tensor, eff_module: nn.Module
    ) -> torch.Tensor:
        """Write the per-species pre-reco region label into ``ColumnMap.EFF_REGION``.

        The label is computed from the PRE-mask (post-smear) kinematics -- the same
        ``(pt, eta_outer)`` the efficiency was evaluated at -- and written into the
        post-mask tensor ``masked`` (a functional masked write via ``torch.where``,
        so it does not assume ``EFF_REGION`` is the last column).
        """
        idx = eff_module.region_index_1based(
            pre_mask[:, ColumnMap.PT], pre_mask[:, ColumnMap.ETA_OUTER]
        )
        is_col = torch.zeros(masked.shape[1], dtype=torch.bool, device=masked.device)
        is_col[ColumnMap.EFF_REGION] = True
        return torch.where(
            is_col.unsqueeze(0),  # (1, F)
            idx.unsqueeze(1).to(masked.dtype),  # (N, 1)
            masked,
        )

    def _expected_reco_counts(
        self, eflow_objects: torch.Tensor, eff_module: nn.Module
    ) -> torch.Tensor:
        """Differentiable expected reconstructed track count per RECO bin, for one
        track species.

        Each reconstructed track carries its pre-reco efficiency region (global
        label) in the ``EFF_REGION`` column. We histogram the trainee's survivors of
        this species into the migration ``M[b, r]`` = number landing in reco-bin ``b``
        whose pre-reco region was ``r`` (a gradient-free count of the current hard
        reco), then form

            expected[b] = sum_r (eff_r / eff_r.detach()) * M[b, r]

        which equals the hard reco count ``sum_r M[b, r]`` at the current efficiencies
        but is differentiable wrt ``eff_logits``. The tuning loss matches ``expected``
        to the reconstructed-data per-reco-bin counts; its minimum is a fixed point at
        the true efficiency (at ``eff = truth`` the trainee's migration reproduces the
        data's, so ``expected = target``).

        The reco bins use the SAME ``(pt, |eta|)`` binning (from the module's
        :class:`~parnassus.torch_delphes.learnable.EfficiencyRegionSpec`) as the
        target in ``tune_cms_fullsim.data.load_pflow_targets``. Selection on the reco
        side is by the species' label range (``EFF_REGION in [offset+1, offset+n]``);
        no PID test is needed because the label already encodes the species.

        Returns
        -------
        torch.Tensor
            ``(n_regions,)`` differentiable expected reco-bin counts.
        """
        spec = eff_module.region_spec
        effs = eff_module.get_efficiencies()  # (n_regions,)
        # A module with per-particle structure INSIDE a region (currently only the
        # muon efficiency, whose top-pt bins roll off as exp(0.5 - rate * pt)) needs
        # the per-particle reweight below; the per-region scalar cannot reach those
        # ``rate_raw`` constants. See the branch comment further down.
        has_rates = hasattr(eff_module, "get_rates")
        if eflow_objects.shape[0] == 0:
            # Graph-connected zeros so backward() always has a path through effs
            # (and through rate_raw for the muon module, whose gradient would
            # otherwise be missing on a batch with no tracks of this species).
            zeros = effs * 0.0
            if has_rates:
                zeros = zeros + eff_module.get_rates().sum() * 0.0
            return zeros
        eff_det = effs.detach().clamp_min(1e-12)  # guard the 1/eff division
        pt = eflow_objects[:, ColumnMap.PT]
        abs_eta = eflow_objects[:, ColumnMap.ETA].abs()
        region = eflow_objects[:, ColumnMap.EFF_REGION]  # global label; 0 = untagged
        valid = pt > 0  # drop efficiency-killed ghosts (zeroed momentum)
        offset = spec.label_offset

        if has_rates:
            # PER-PARTICLE reweight ``eff_i / eff_i.detach()``, summed into the
            # track's reco bin. Identical in VALUE to the per-region form below
            # (every ratio is exactly 1, so the forward count is unchanged and the
            # fixed point at the true efficiency is preserved), but it carries
            # gradient to any parameter ``compute_efficiency`` uses -- including the
            # muon ``rate_raw`` roll-off constants, which enter non-multiplicatively
            # inside exp() and are therefore invisible to a per-region scalar.
            #
            # Only valid where reco pt == pre-reco pt, i.e. the efficiency can be
            # re-evaluated at the track's CURRENT kinematics. Muons satisfy this
            # (their ECal/HCal energy fractions are 0, so they traverse both
            # calorimeters unrescaled); charged hadrons and electrons do NOT -- the
            # PF rescale moves their pt -- so those keep the per-region path.
            # ETA_OUTER (not ETA) is used because that is what the efficiency module
            # itself binned on; both columns are carried through unchanged.
            eta_outer = eflow_objects[:, ColumnMap.ETA_OUTER]
            per_particle = eff_module.compute_efficiency(pt, eta_outer)
            weight = per_particle / per_particle.detach().clamp_min(1e-12)
            in_species = (region > offset) & (region <= offset + spec.n_regions)
            expected = []
            for b_mask in spec.region_masks(pt, abs_eta):
                sel = (valid & b_mask & in_species).to(weight.dtype)
                expected.append((weight * sel).sum())
            return torch.stack(expected)

        # For each reco bin, find which pre-reco region the survivors came from, then
        # add their counts reweighted by eff/eff.detach(); the ratio is 1 in value
        # (forward count stays exact) but passes gradient M[b,r]/eff_r to eff_logits.
        expected = []
        for b_mask in spec.region_masks(pt, abs_eta):
            term = torch.zeros((), dtype=effs.dtype, device=effs.device)
            for r in range(spec.n_regions):
                label = float(offset + r + 1)
                m_br = (valid & b_mask & (region == label)).sum().to(effs.dtype)
                term = term + (effs[r] / eff_det[r]) * m_br
            expected.append(term)
        return torch.stack(expected)

    def _setup_ECal(self):
        energy_fractions = {
            0: 0.0,  # default (hadrons) - no ECAL response
            11: 1.0,  # electrons
            22: 1.0,  # photons
            111: 1.0,  # pi0
            12: 0.0,  # neutrino (electron)
            13: 0.0,  # muon
            14: 0.0,  # neutrino (muon)
            16: 0.0,  # neutrino (tau)
            1000022: 0.0,  # neutralino
            1000023: 0.0,  # neutralino
            1000025: 0.0,  # neutralino
            1000035: 0.0,  # neutralino
            1000045: 0.0,  # neutralino
            310: 0.3,  # K0short
            3122: 0.3,  # Lambda
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
        hf_etas = [
            -5,
            -4.7,
            -4.525,
            -4.35,
            -4.175,
            -4,
            -3.825,
            -3.65,
            -3.475,
            -3.3,
            -3.125,
            -2.958,
            3.125,
            3.3,
            3.475,
            3.65,
            3.825,
            4,
            4.175,
            4.35,
            4.525,
            4.7,
            5,
        ]
        for eta in hf_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_coarse)

        # Convert to sorted lists (matching C++ behavior)
        eta_bins = sorted(eta_phi_map.keys())
        phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

        ecal_res: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ecal_scale_fn: Callable[[torch.Tensor], torch.Tensor] | None
        learnable_fractions: nn.Module | None
        if self.learnable:
            ecal_res = LearnableEcalCMSResolution()
            ecal_scale_fn = make_cms_ecal_scale()
            learnable_fractions = self.HadronFractions
        else:
            ecal_res = "ecal_cms"
            ecal_scale_fn = None
            learnable_fractions = None

        self.ECal = SimpleCalorimeter(
            eta_bins=eta_bins,
            phi_bins=phi_bins_per_eta,
            energy_min=0.5,
            energy_sig_min=2.0,
            energy_fractions=energy_fractions,
            resolution_formula=ecal_res,
            is_ecal=True,
            smear_tower_center=True,  # Match C++ Delphes: SmearTowerCenter true
            scale_fn=ecal_scale_fn,
            learnable_fractions=learnable_fractions,
            # Differentiable per-region count term: on exactly in learnable mode,
            # mirroring the efficiency count term (off => generation byte-identical).
            compute_soft_count=self.learnable,
        )

    def _setup_HCal(self):
        energy_fractions = {
            0: 1.0,  # default (hadrons) - full HCAL response
            11: 0.0,  # electrons (no HCAL response - already absorbed by ECAL)
            22: 0.0,  # photons (no HCAL response)
            111: 0.0,  # pi0 (no HCAL response)
            12: 0.0,  # neutrino (electron)
            13: 0.0,  # muon
            14: 0.0,  # neutrino (muon)
            16: 0.0,  # neutrino (tau)
            1000022: 0.0,  # neutralino
            1000023: 0.0,  # neutralino
            1000025: 0.0,  # neutralino
            1000035: 0.0,  # neutralino
            1000045: 0.0,  # neutralino
            310: 0.7,  # K0short (70% HCAL)
            3122: 0.7,  # Lambda (70% HCAL)
        }

        eta_phi_map = {}  # eta -> set of phi bin edges

        # 5 degrees towers (barrel): phi bins -36 to 36 in steps of pi/36
        phi_bins_5deg = [i * np.pi / 36.0 for i in range(-36, 37)]
        barrel_etas = [
            -1.566,
            -1.479,
            -1.392,
            -1.305,
            -1.218,
            -1.131,
            -1.044,
            -0.957,
            -0.87,
            -0.783,
            -0.696,
            -0.609,
            -0.522,
            -0.435,
            -0.348,
            -0.261,
            -0.174,
            -0.087,
            0,
            0.087,
            0.174,
            0.261,
            0.348,
            0.435,
            0.522,
            0.609,
            0.696,
            0.783,
            0.87,
            0.957,
            1.044,
            1.131,
            1.218,
            1.305,
            1.392,
            1.479,
            1.566,
            1.653,
        ]
        for eta in barrel_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_5deg)

        # 10 degrees towers (endcap): phi bins -18 to 18 in steps of pi/18
        phi_bins_10deg = [i * np.pi / 18.0 for i in range(-18, 19)]
        endcap_etas = [
            -4.35,
            -4.175,
            -4,
            -3.825,
            -3.65,
            -3.475,
            -3.3,
            -3.125,
            -2.95,
            -2.868,
            -2.65,
            -2.5,
            -2.322,
            -2.172,
            -2.043,
            -1.93,
            -1.83,
            -1.74,
            -1.653,
            1.74,
            1.83,
            1.93,
            2.043,
            2.172,
            2.322,
            2.5,
            2.65,
            2.868,
            2.95,
            3.125,
            3.3,
            3.475,
            3.65,
            3.825,
            4,
            4.175,
            4.35,
            4.525,
        ]
        for eta in endcap_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_10deg)

        # 20 degrees towers (forward): phi bins -9 to 9 in steps of pi/9
        phi_bins_20deg = [i * np.pi / 9.0 for i in range(-9, 10)]
        forward_etas = [-5, -4.7, -4.525, 4.7, 5]
        for eta in forward_etas:
            if eta not in eta_phi_map:
                eta_phi_map[eta] = set()
            eta_phi_map[eta].update(phi_bins_20deg)

        # Convert to sorted lists (matching C++ behavior)
        eta_bins = sorted(eta_phi_map.keys())
        phi_bins_per_eta = [sorted(eta_phi_map[eta]) for eta in eta_bins]

        hcal_res: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        hcal_scale_fn: Callable[[torch.Tensor], torch.Tensor] | None
        learnable_fractions: nn.Module | None
        if self.learnable:
            hcal_res = LearnableHcalCMSResolution()
            hcal_scale_fn = make_cms_hcal_scale()
            learnable_fractions = self.HadronFractions
        else:
            hcal_res = "hcal_cms"
            hcal_scale_fn = None
            learnable_fractions = None

        self.HCal = SimpleCalorimeter(
            eta_bins=eta_bins,
            phi_bins=phi_bins_per_eta,
            energy_min=1.0,  # HCal has higher threshold
            energy_sig_min=1.0,  # HCal has lower significance threshold
            energy_fractions=energy_fractions,
            resolution_formula=hcal_res,
            is_ecal=False,
            smear_tower_center=True,
            scale_fn=hcal_scale_fn,
            learnable_fractions=learnable_fractions,
            compute_soft_count=self.learnable,
        )
