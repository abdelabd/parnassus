"""M1 gate: c_E/c_S count-gradient sign test through the merged-count composition.

For the legacy per-tower count (merger off) and the cluster-composition count
(merger on, R = 0.045), compares the autograd gradient of the per-region
expected ECal-photon count against a central finite difference of the same
(straight-through, hard-integer) count under +-2% parameter shifts. The M1
gate: signs agree wherever the FD response is nonzero, and the merger-on
gradient survives (attenuation consistent with the measured singleton
fractions: barrel ~2x, forward ~1.3x). Also checks that the hard merged count
is monotone decreasing in R (the future M3 gradient channel has the right
sign end-to-end).

Usage: <env>/bin/python notebooks/m1_sign_test.py [--events 200]
"""

import argparse

import torch

from parnassus.torch_delphes.defaults.CMSDefault import CMSEnergyFlowDefault
from parnassus.torch_delphes.PhotonClusterMerger import PhotonClusterMerger
from parnassus.torch_delphes.tune_cms_fullsim.data import load_cms_flow_root, load_truth_events

DIJET = "/global/cfs/cdirs/m3246/diff_delphes/cms_nopu_postpro/cms_nopileup_sim_pt2500to3000_processed_selected.root"
PARAMS = (
    "ECal.resolution_func.common_c_E",
    "ECal.resolution_func.common_c_S",
    "ECal.resolution_func.forward_c_E",
    "ECal.resolution_func.forward_c_S",
)
REGIONS = ("|eta|<1.5", "1.5-2.5", "2.5-2.7")
SEED = 77
CHUNK = 100
FD_REL = 0.02


def make_card(radius: float | None) -> CMSEnergyFlowDefault:
    return CMSEnergyFlowDefault(
        debug=False, learnable=True, count_pt_min=1.0, count_abs_eta_max=2.7,
        photon_merger=None if radius is None else PhotonClusterMerger(radius),
    )


def counts_vec(card, truth, grad: bool = False):
    """Per-region expected ECal-photon counts over all events (fixed RNG)."""
    torch.manual_seed(SEED)
    total = torch.zeros(3, dtype=torch.float64)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for i0 in range(0, truth.shape[0], CHUNK):
            chunk = truth[i0 : i0 + CHUNK]
            flat = chunk[torch.any(chunk != 0, dim=-1)]
            c = card(flat)["EcalPhotonExpectedCounts"]
            if grad:
                total = total + c
            else:
                total += c.detach()
    return total


def autograd_grads(card, truth) -> dict[str, torch.Tensor]:
    """d(counts_r)/d(theta) per region via per-chunk backward accumulation."""
    params = {n: p for n, p in card.named_parameters() if n in PARAMS}
    grads = {n: torch.zeros(3, dtype=torch.float64) for n in params}
    torch.manual_seed(SEED)
    for i0 in range(0, truth.shape[0], CHUNK):
        chunk = truth[i0 : i0 + CHUNK]
        flat = chunk[torch.any(chunk != 0, dim=-1)]
        c = card(flat)["EcalPhotonExpectedCounts"]
        for r in range(3):
            gs = torch.autograd.grad(
                c[r], list(params.values()), retain_graph=(r < 2), allow_unused=True
            )
            for (n, _), g in zip(params.items(), gs, strict=True):
                if g is not None:
                    grads[n][r] += g.reshape(-1).sum()
    return grads


def fd_grads(card, truth) -> dict[str, torch.Tensor]:
    """Central FD of the (hard, ST-pinned) counts under +-FD_REL shifts."""
    params = dict(card.named_parameters())
    out = {}
    for name in PARAMS:
        p = params[name]
        base = p.detach().clone()
        delta = float(base.abs().reshape(-1)[0]) * FD_REL or 1e-4
        with torch.no_grad():
            p.copy_(base + delta)
        up = counts_vec(card, truth)
        with torch.no_grad():
            p.copy_(base - delta)
        dn = counts_vec(card, truth)
        with torch.no_grad():
            p.copy_(base)
        out[name] = (up - dn) / (2.0 * delta)
    return out


def report(tag, auto, fd):
    print(f"\n=== {tag} ===")
    print(f"{'param':>34} {'region':>9} {'autograd':>12} {'FD(hard)':>12}  sign")
    ok = mismatches = 0
    for name in PARAMS:
        for r, rname in enumerate(REGIONS):
            a, f = float(auto[name][r]), float(fd[name][r])
            if f == 0.0 and abs(a) < 1e-12:
                verdict = "quiet"
            elif f == 0.0:
                verdict = "fd-flat"
            elif a * f > 0:
                verdict = "MATCH"
                ok += 1
            else:
                verdict = "MISMATCH"
                mismatches += 1
            print(f"{name:>34} {rname:>9} {a:12.4g} {f:12.4g}  {verdict}")
    print(f"sign matches: {ok}, mismatches: {mismatches}")
    return mismatches


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=200)
    args = ap.parse_args()

    arrays = load_cms_flow_root(DIJET, n_events=args.events)
    truth = load_truth_events(arrays, truth_pt_cut=0.25, abs_eta_cut=2.7)
    print(f"{truth.shape[0]} dijet events")

    torch.manual_seed(0)
    total_mismatch = 0
    results = {}
    for tag, radius in (("merger OFF (legacy per-tower count)", None),
                        ("merger ON, R=0.045 (cluster composition)", 0.045)):
        card = make_card(radius)
        auto = autograd_grads(card, truth)
        fd = fd_grads(card, truth)
        total_mismatch += report(tag, auto, fd)
        results[tag] = auto

    print("\n=== attenuation (autograd on/off ratio, summed |grad| over regions) ===")
    off, on = results.values()
    for name in PARAMS:
        o, n = float(off[name].abs().sum()), float(on[name].abs().sum())
        print(f"{name:>34}  off {o:10.4g}  on {n:10.4g}  ratio {n / o if o else float('nan'):.3f}")

    print("\n=== hard merged count vs R (M3 channel sign check) ===")
    for r in (0.02, 0.045, 0.08):
        card = make_card(r)
        print(f"  R={r}: total={float(counts_vec(card, truth).sum()):.0f}")

    print(f"\nGATE: {'PASS' if total_mismatch == 0 else 'FAIL'} (sign mismatches: {total_mismatch})")
