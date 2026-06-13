"""Synthetic ROOT fixture writer for ``tune_cms_fullsim``.

When no real CMS full-simulation ROOT file is available, this module writes
a tiny stand-in file with the same cms-flow schema so the whole pipeline is
runnable for sandbox validation and CI. The ``pflow_*`` branches are produced
by running a *deliberately mis-scaled* learnable card on synthetic truth, so
fitting a fresh card against the fixture should recover the known perturbation.
"""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.utils import class_to_pid_vectorized

# =============================================================================
# Synthetic fixture (for sandbox validation / CI)
# =============================================================================


def write_synthetic_fixture(
    out_path: Path,
    n_events: int = 50,
    particles_per_event: int = 60,
    seed: int = 0,
) -> None:
    """Write a tiny ROOT file with the cms-flow schema for validation.

    The ``truth_*`` arrays are a plausible mixed-species QCD-like batch.
    The ``pflow_*`` arrays are generated from the same underlying truth
    by applying a *deliberately mis-scaled* version of the CMS default
    Delphes card: charged-hadron pT is multiplied by 1.2 and the ECal
    energy scale by 1.1. This mimics a "Delphes-with-wrong-scales" ->
    "CMS full sim" gap, so that fitting the fresh trainee card against
    the pflow branches should move those scale parameters back toward
    1.2 / 1.1.

    The fixture is NOT a substitute for the real Zenodo sample -- it's
    just a local stand-in so we can run and test the full pipeline.
    """
    rng = np.random.default_rng(seed)

    truth_pt_list: list[np.ndarray] = []
    truth_eta_list: list[np.ndarray] = []
    truth_phi_list: list[np.ndarray] = []
    truth_class_list: list[np.ndarray] = []
    truth_pdgid_list: list[np.ndarray] = []

    # Underlying particle tensor we can feed into a card directly.
    truth_arrays_flat: list[np.ndarray] = []
    for i in range(n_events):
        n_p = int(max(10, rng.normal(particles_per_event, particles_per_event * 0.2)))
        pt = rng.uniform(0.5, 80.0, size=n_p).astype(np.float32)
        eta = rng.uniform(-3.0, 3.0, size=n_p).astype(np.float32)
        phi = rng.uniform(-np.pi, np.pi, size=n_p).astype(np.float32)
        cls = rng.integers(0, 5, size=n_p).astype(np.int32)
        truth_pt_list.append(pt)
        truth_eta_list.append(eta)
        truth_phi_list.append(phi)
        truth_class_list.append(cls)

        # Corresponding flat particle row for running through a card.
        pids = class_to_pid_vectorized(cls.astype(np.int64))
        truth_pdgid_list.append(pids.astype(np.int32))
        abs_pid = np.abs(pids)
        mass = np.where(
            abs_pid == 11,
            0.000511,
            np.where(abs_pid == 13, 0.10566, 0.13957),
        ).astype(np.float64)
        pt_d = pt.astype(np.float64)
        eta_d = eta.astype(np.float64)
        phi_d = phi.astype(np.float64)
        px = pt_d * np.cos(phi_d)
        py = pt_d * np.sin(phi_d)
        pz = pt_d * np.sinh(eta_d)
        e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
        charge = np.where(
            abs_pid == 211,
            1.0,
            np.where(
                abs_pid == 11,
                -1.0,
                np.where(abs_pid == 13, -1.0, 0.0),
            ),
        ).astype(np.float64)
        row = np.zeros((n_p, N_FEATURES), dtype=np.float64)
        row[:, ColumnMap.PID] = pids
        row[:, ColumnMap.STATUS] = 1
        row[:, ColumnMap.CHARGE] = charge
        row[:, ColumnMap.E] = e
        row[:, ColumnMap.PX] = px
        row[:, ColumnMap.PY] = py
        row[:, ColumnMap.PZ] = pz
        row[:, ColumnMap.PT] = pt_d
        row[:, ColumnMap.ETA] = eta_d
        row[:, ColumnMap.PHI] = phi_d
        row[:, ColumnMap.MASS] = mass
        row[:, ColumnMap.EVENT_NUMBER] = i
        truth_arrays_flat.append(row)

    truth_tensor = torch.from_numpy(np.concatenate(truth_arrays_flat, axis=0))

    # "Full-sim-like" reco is obtained by running a PERTURBED learnable card
    # on the truth particles and using its EFlowObject output as the
    # pflow_* truth. The perturbation is exactly what we'd like Adam to
    # recover.
    #
    # We make the perturbation fairly large (chad pT scale = 1.25, ECal
    # energy scale = 1.20 in every region) so the target distributions
    # are clearly distinguishable from the defaults, and we average the
    # target card's output over ``n_target_passes`` runs to beat down the
    # per-realization sampling noise in the saved pflow branches.
    torch.manual_seed(seed)
    target_card = CMSEnergyFlowDefault(debug=False, learnable=True)
    with torch.no_grad():
        # Range-guarded scale -> raw via the shared param_config transform.
        chad_res = target_card.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr]
        chad_name = "ChargedHadronMomentumSmearing.resolution_module.scale_raw"
        chad_res.scale_raw.copy_(
            pc.to_raw(chad_name, [1.25] * chad_res.scale_raw.numel()).reshape(
                chad_res.scale_raw.shape
            )
        )
        ecal_scale = target_card.ECal.scale_module  # type: ignore[union-attr]
        ecal_scale.scale_raw.copy_(
            pc.to_raw("ECal.scale_module.scale_raw", [1.20] * ecal_scale.scale_raw.numel()).reshape(
                ecal_scale.scale_raw.shape
            )
        )
    for p in target_card.parameters():
        p.requires_grad_(False)
    target_card.eval()

    n_target_passes = 5
    with torch.no_grad():
        eflow_collected: list[torch.Tensor] = [
            target_card(truth_tensor)["EFlowObject"] for _ in range(n_target_passes)
        ]
    # Concatenate all passes into a single "pooled" EFlow tensor. The
    # corresponding event-number column is already correctly set because
    # each pass uses the same truth input, so we can sort by event and
    # then stack all passes' particles under each event.
    eflow = torch.cat(eflow_collected, dim=0)
    event_idx = eflow[:, ColumnMap.EVENT_NUMBER].long().cpu().numpy()
    pflow_pt_arr = eflow[:, ColumnMap.PT].cpu().numpy().astype(np.float32)
    pflow_eta_arr = eflow[:, ColumnMap.ETA].cpu().numpy().astype(np.float32)
    pflow_phi_arr = eflow[:, ColumnMap.PHI].cpu().numpy().astype(np.float32)
    pflow_pid = eflow[:, ColumnMap.PID].cpu().numpy().astype(np.int64)
    # Convert PID -> class for the fixture (matching the real schema).
    abs_pid = np.abs(pflow_pid)
    pflow_cls_arr = np.where(
        abs_pid == 11,
        1,
        np.where(
            abs_pid == 13,
            2,
            np.where(abs_pid == 22, 4, np.where(abs_pid == 211, 0, 3)),
        ),
    ).astype(np.int32)
    # Drop zero-pt ghost tracks from the fixture — no physical observable
    # depends on them and they'd pollute the target multiplicity.
    keep = pflow_pt_arr > 1e-6
    event_idx = event_idx[keep]
    pflow_pt_arr = pflow_pt_arr[keep]
    pflow_eta_arr = pflow_eta_arr[keep]
    pflow_phi_arr = pflow_phi_arr[keep]
    pflow_cls_arr = pflow_cls_arr[keep]

    pflow_pt_list = [pflow_pt_arr[event_idx == i] for i in range(n_events)]
    pflow_eta_list = [pflow_eta_arr[event_idx == i] for i in range(n_events)]
    pflow_phi_list = [pflow_phi_arr[event_idx == i] for i in range(n_events)]
    pflow_cls_list = [pflow_cls_arr[event_idx == i] for i in range(n_events)]

    with uproot.recreate(str(out_path)) as f:
        f["event_tree"] = {
            "truth_pt": ak.Array(truth_pt_list),
            "truth_eta": ak.Array(truth_eta_list),
            "truth_phi": ak.Array(truth_phi_list),
            "truth_class": ak.Array(truth_class_list),
            "truth_pdgid": ak.Array(truth_pdgid_list),
            "pflow_pt": ak.Array(pflow_pt_list),
            "pflow_eta": ak.Array(pflow_eta_list),
            "pflow_phi": ak.Array(pflow_phi_list),
            "pflow_class": ak.Array(pflow_cls_list),
        }
