"""M0 gate checks for PhotonClusterMerger (photon_merger_fraction_design.md sec 6).

Runs the default-init learnable card (round-0 trainee) on the dijet and mee
samples, applies the merger to the eflow photon stream, and reports:
  - dijet: photons/event unmerged vs merged vs CMS PF (gate: merged within
    ~10% of PF), and the high-pt tail population (700-1100 GeV bump);
  - mee: merged ~= unmerged (gate: < ~5% change on the sparse sample);
  - gradient smoke: backward through merged kinematics is finite.

Usage (login node is fine):
  <env>/bin/python notebooks/m0_photon_merger_check.py [--dijet-events 600] [--mee-events 1000]
"""

import argparse

import numpy as np
import torch

from parnassus.data.particle_io import ColumnMap
from parnassus.torch_delphes.defaults.CMSDefault import CMSEnergyFlowDefault
from parnassus.torch_delphes.PhotonClusterMerger import PhotonClusterMerger
from parnassus.torch_delphes.tune_cms_fullsim.data import load_cms_flow_root, load_truth_events

DIJET = "/global/cfs/cdirs/m3246/diff_delphes/cms_nopu_postpro/cms_nopileup_sim_pt2500to3000_processed_selected.root"
MEE = "/global/cfs/cdirs/m3246/diff_delphes/cms_nopu_mee/cms_nopileup_mee120to200_processed.root"
RADIUS = 0.045
TRUTH_PT_CUT, RECO_PT_CUT, ETA_CUT = 0.25, 1.0, 2.7
PT_TAIL_EDGES = (300.0, 500.0, 700.0, 1100.0)


def pf_photon_stats(arrays) -> tuple[float, dict[float, float]]:
    """CMS PF photons/event in acceptance + tail counts/event above edges."""
    n_ev = len(arrays["pflow_pt"])
    total, tails = 0, dict.fromkeys(PT_TAIL_EDGES, 0)
    for i in range(n_ev):
        cls = np.asarray(arrays["pflow_class"][i])  # canonical class 4 = photon
        pt = arrays["pflow_pt"][i]
        eta = arrays["pflow_eta"][i]
        sel = (cls == 4) & (pt >= RECO_PT_CUT) & (np.abs(eta) <= ETA_CUT)
        total += int(sel.sum())
        for edge in PT_TAIL_EDGES:
            tails[edge] += int((sel & (pt >= edge)).sum())
    return total / n_ev, {e: c / n_ev for e, c in tails.items()}


def trainee_photons(card, truth, chunk_events: int = 100) -> torch.Tensor:
    """Run the card in event chunks; return the concatenated flat photon stream."""
    outs = []
    with torch.no_grad():
        for i0 in range(0, truth.shape[0], chunk_events):
            chunk = truth[i0 : i0 + chunk_events]
            flat = chunk[torch.any(chunk != 0, dim=-1)]  # advanced indexing copies
            outs.append(card(flat)["EFlowPhoton"])
    return torch.cat(outs, dim=0)


def photon_stats(photons: torch.Tensor, n_events: int) -> tuple[float, dict[float, float]]:
    pt = photons[:, ColumnMap.PT]
    eta = photons[:, ColumnMap.ETA]
    sel = (pt >= RECO_PT_CUT) & (eta.abs() <= ETA_CUT)
    tails = {e: float((sel & (pt >= e)).sum()) / n_events for e in PT_TAIL_EDGES}
    return float(sel.sum()) / n_events, tails


def report(name: str, path: str, n_events: int, merger: PhotonClusterMerger) -> None:
    print(f"\n=== {name}: {n_events} events ===")
    arrays = load_cms_flow_root(path, n_events=n_events)
    n_ev = len(arrays["pflow_pt"])
    truth = load_truth_events(arrays, truth_pt_cut=TRUTH_PT_CUT, abs_eta_cut=ETA_CUT)

    torch.manual_seed(20260811)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    unmerged = trainee_photons(card, truth)
    merged = merger(unmerged)

    pf_rate, pf_tails = pf_photon_stats(arrays)
    un_rate, un_tails = photon_stats(unmerged, n_ev)
    me_rate, me_tails = photon_stats(merged, n_ev)

    print(f"photons/event (pt>={RECO_PT_CUT}, |eta|<={ETA_CUT}):")
    print(f"  PF target      {pf_rate:7.2f}")
    print(f"  trainee raw    {un_rate:7.2f}  ({un_rate / pf_rate:.2f}x PF)")
    print(f"  trainee merged {me_rate:7.2f}  ({me_rate / pf_rate:.2f}x PF)")
    print(f"  merged/raw = {me_rate / max(un_rate, 1e-9):.3f}")
    print("high-pt tail (photons/event above pt edge):")
    print(f"  {'edge':>6} {'PF':>9} {'raw':>9} {'merged':>9}")
    for e in PT_TAIL_EDGES:
        print(f"  {e:6.0f} {pf_tails[e]:9.4f} {un_tails[e]:9.4f} {me_tails[e]:9.4f}")


def grad_smoke(merger: PhotonClusterMerger) -> None:
    arrays = load_cms_flow_root(DIJET, n_events=20)
    truth = load_truth_events(arrays, truth_pt_cut=TRUTH_PT_CUT, abs_eta_cut=ETA_CUT)
    torch.manual_seed(1)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    flat = truth[torch.any(truth != 0, dim=-1)]
    merged = merger(card(flat)["EFlowPhoton"])
    merged[:, ColumnMap.E].sum().backward()
    n_grads = bad = 0
    for name, p in card.named_parameters():
        if p.grad is not None:
            n_grads += 1
            if not torch.isfinite(p.grad).all():
                bad += 1
                print(f"  NON-FINITE grad: {name}")
    print(f"\n[grad smoke] params with grads through merged photons: {n_grads}, non-finite: {bad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dijet-events", type=int, default=600)
    ap.add_argument("--mee-events", type=int, default=1000)
    ap.add_argument("--radius", type=float, default=RADIUS)
    args = ap.parse_args()

    merger = PhotonClusterMerger(args.radius)
    print(f"PhotonClusterMerger R = {args.radius}")
    report("dijet pt2500-3000", DIJET, args.dijet_events, merger)
    report("mee 120-200", MEE, args.mee_events, merger)
    grad_smoke(merger)
