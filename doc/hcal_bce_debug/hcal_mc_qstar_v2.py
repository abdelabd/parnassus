"""MC q* calibration v2 — A/B of the tower-BCE threshold modes.

Fixes the v1 corruption: card(flat) MUTATES its input in place (10 columns),
so v1's replicas each saw the previous replica's mutated input. Here every
forward gets flat.clone(); with that, the forward is deterministic per seed.

A = tower_bce_threshold "sampled_sigma"  (legacy: c2/c4 use sigma at this
    draw's sampled smeared energy)
B = "self_consistent"  (c2/c4 are the fixed points E* = S*sigma(E*), exact
    trackless tail under the lognormal)
Both under conditioning="expected". Same seed per replica => identical draws,
identical tower sets; q*_MC from the hard emission flags (mode-independent).
"""
import sys
import torch
import numpy as np

sys.path.insert(0, "/pscratch/sd/a/aelabd/parnassus/src")
from pathlib import Path
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.tune_cms_fullsim.data import load_cms_flow_root, load_truth_events_ragged

TRUTH_CFG = "/pscratch/sd/a/aelabd/parnassus/src/parnassus/torch_delphes/param_configs/param_config_all.yaml"
SAMPLE = "/global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_200k_param_config_all_dijet_hungarian_matched_survival.root"
N_EVENTS, N_REP = 256, 150
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make(mode):
    card = CMSEnergyFlowDefault(debug=False, learnable=True,
                                tower_bce_conditioning="expected",
                                tower_bce_threshold=mode)
    cfg = pc.load_param_config_over_defaults(TRUTH_CFG, card)
    pc.apply_param_config(card, cfg)
    for p in card.parameters():
        p.requires_grad_(False)
    return card.to(DEV)


cardA, cardB = make("sampled_sigma"), make("self_consistent")
arrays = load_cms_flow_root(Path(SAMPLE), n_events=N_EVENTS)
flat = torch.cat(load_truth_events_ragged(arrays), dim=0).to(DEV)

keys_l, emit_l, qa_l, qb_l, trk_l, dep_l = [], [], [], [], [], []
with torch.no_grad():
    for rep in range(N_REP):
        torch.manual_seed(10_000 + rep)
        exA = cardA(flat.clone())["HcalCountExport"]
        torch.manual_seed(10_000 + rep)
        exB = cardB(flat.clone())["HcalCountExport"]
        kA = (exA["bce_event"].long() * 10_000_000
              + torch.round(exA["bce_eta_lo"] * 1000).long() * 1000
              + torch.round((exA["bce_phi_lo"] + 4) * 100).long())
        kB = (exB["bce_event"].long() * 10_000_000
              + torch.round(exB["bce_eta_lo"] * 1000).long() * 1000
              + torch.round((exB["bce_phi_lo"] + 4) * 100).long())
        assert torch.equal(kA, kB), f"rep {rep}: tower sets differ between modes"
        keys_l.append(kA.cpu().numpy())
        emit_l.append(exA["bce_emitted"].double().cpu().numpy())
        qa_l.append(exA["bce_logq"].exp().cpu().numpy())
        qb_l.append(exB["bce_logq"].exp().cpu().numpy())
        trk_l.append(exA["bce_track_e"].cpu().numpy())
        dep_l.append(exA["bce_deposit"].cpu().numpy())
        if rep % 25 == 0:
            print(f"rep {rep}: {kA.numel()} towers", flush=True)

keys = np.concatenate(keys_l)
uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
agg = lambda x: np.bincount(inv, weights=np.concatenate(x)) / cnt
q_star, q_a, q_b = agg(emit_l), agg(qa_l), agg(qb_l)
trk, dep = agg(trk_l), agg(dep_l)
stable = cnt >= 0.98 * N_REP
q_star, q_a, q_b, trk, dep = (v[stable] for v in (q_star, q_a, q_b, trk, dep))
f_trk = trk / np.clip(trk + dep, 1e-12, None)
print(f"\n{stable.sum()} stable towers (of {uk.size} keys), {N_REP} replicas, device={DEV}")

edges = [(-0.01, 0.001), (0.001, 0.3), (0.3, 0.6), (0.6, 0.85), (0.85, 1.01)]


def table(mask, title):
    print(f"\n== {title} ==")
    print(f"{'trackfrac bin':>16} {'n':>7} {'<q*>':>8} {'biasA(sampled)':>15} {'biasB(selfcon)':>15}")
    for lo, hi in edges:
        m = mask & (f_trk > lo) & (f_trk <= hi)
        if m.sum() < 20:
            continue
        print(f"[{lo:5.2f},{hi:5.2f}] {int(m.sum()):>7} {q_star[m].mean():>8.4f} "
              f"{(q_a[m]-q_star[m]).mean():>+15.4f} {(q_b[m]-q_star[m]).mean():>+15.4f}")
    print(f"{'ALL':>16} {int(mask.sum()):>7} {q_star[mask].mean():>8.4f} "
          f"{(q_a[mask]-q_star[mask]).mean():>+15.4f} {(q_b[mask]-q_star[mask]).mean():>+15.4f}")


table(np.ones_like(q_star, dtype=bool), "all stable towers")
table((q_star > 0.05) & (q_star < 0.95), "near-threshold (0.05 < q* < 0.95)")
# RMS error too (bias can hide compensating errors)
for name, q in (("A sampled_sigma", q_a), ("B self_consistent", q_b)):
    m = (q_star > 0.05) & (q_star < 0.95)
    print(f"near-threshold RMS |q-q*| {name}: {np.sqrt(((q[m]-q_star[m])**2).mean()):.4f}")
