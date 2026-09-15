"""Frozen-truth HCal scale scan v2 — A/B of the tower-BCE threshold modes.

Everything at truth except one HCal physical scale; sweep it and record the
TowerBceHcal value under threshold mode A (sampled_sigma, legacy) and
B (self_consistent), plus the count chi^2 for reference. v1 (mode A only,
conditioning=expected) showed region 0 monotone-decreasing (no minimum at
truth 0.849) and region 1 with a shallow biased-low minimum ~0.83-0.85.
If B restores minima at ~0.849 / ~0.893, the sampled-sigma threshold is the
HCal pseudo-truth mechanism. Inputs are safe: tp[mask] copies (the in-place
card(input) mutation bug does not bite here).
"""
import sys
import math
import torch
import numpy as np

sys.path.insert(0, "/pscratch/sd/a/aelabd/parnassus/src")
from pathlib import Path
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.tune_cms_fullsim.data import (
    load_cms_flow_root, load_truth_events_ragged, load_pflow_targets_ragged,
    load_pflow_targets_from_tensor, restore_event_format, batch_event_ids,
)
from parnassus.torch_delphes.tune_cms_fullsim.dataloader import DelphesDataSet, DelphesDataLoader
from parnassus.torch_delphes.tune_cms_fullsim.training import _inject_tower_bce
from parnassus.torch_delphes.tune_cms_fullsim.loss import _tower_bce_terms, _count_terms
from parnassus.torch_delphes.tune_cms_fullsim.config import COUNT_TERM_KEYS, CALO_COUNT_TERM_KEYS

TRUTH_CFG = "/pscratch/sd/a/aelabd/parnassus/src/parnassus/torch_delphes/param_configs/param_config_all.yaml"
SAMPLE = "/global/cfs/cdirs/m3246/diff_delphes/allsamples/pseudo_data_200k_param_config_all_dijet_hungarian_matched_survival.root"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_card(threshold):
    card = CMSEnergyFlowDefault(debug=False, learnable=True,
                                tower_bce_conditioning="expected",
                                tower_bce_threshold=threshold)
    cfg = pc.load_param_config_over_defaults(TRUTH_CFG, card)
    pc.apply_param_config(card, cfg)
    for p in card.parameters():
        p.requires_grad_(False)
    return card.to(DEV)


arrays = load_cms_flow_root(Path(SAMPLE), n_events=3072)
ds = DelphesDataSet(load_truth_events_ragged(arrays), load_pflow_targets_ragged(arrays), device=DEV)
bs = list(DelphesDataLoader(ds, batch_size=1024, shuffle=False))


def tower_terms_for(card, batch, seed=0):
    torch.manual_seed(seed)
    tp = batch["truth_particles"]
    mask = torch.any(tp != 0, dim=-1)
    out = card(tp[mask])
    eflow = restore_event_format(out["EFlowObject"], mask, event_ids=batch_event_ids(tp, mask))
    pred = load_pflow_targets_from_tensor(eflow)
    for ok, pk, _tk in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        pred[pk] = out[ok]
    tgt = {k: batch[k] for k in batch if k != "truth_particles"}
    _inject_tower_bce(pred, tgt, out, batch)
    comps = []
    terms = _tower_bce_terms(pred, tgt, calo_bce_weight=1.0, out_components=comps)
    tower = dict(zip([c[0] for c in comps], terms))
    ccomps = []
    _count_terms(pred, tgt, count_weight=0.0, calo_count_weight=1.0,
                 include_tracking=False, out_components=ccomps)
    counts = {lbl: raw for (lbl, raw, _w) in ccomps}
    return tower, counts


cards = {"A sampled": make_card("sampled_sigma"), "B selfcon": make_card("self_consistent")}
truth_raw = cards["A sampled"].HCal.scale_module.scale_raw.detach().clone()
truth_phys = (1 + 0.3 * torch.tanh(truth_raw)).tolist()

with torch.no_grad():
    for region in (0, 1):
        print(f"\n-- region {region} (truth phys = {truth_phys[region]:.4f})")
        print(f"{'scale':>7} {'BceHcal A':>12} {'BceHcal B':>12} {'count chi2':>12}")
        for s in np.linspace(0.70, 1.05, 8):
            raw = math.atanh(max(min((s - 1.0) / 0.3, 0.999), -0.999))
            vals = {k: [] for k in cards}
            vc = []
            for name, card in cards.items():
                card.HCal.scale_module.scale_raw.copy_(truth_raw)
                card.HCal.scale_module.scale_raw[region] = raw
                for b in bs:
                    for seed in (0, 1):
                        t, c = tower_terms_for(card, b, seed=seed)
                        vals[name].append(float(t["TowerBceHcal"]))
                        if name.startswith("A"):
                            vc.append(float(c.get("HcalNeutralHadron", torch.tensor(float("nan")))))
            print(f"{s:>7.3f} {np.mean(vals['A sampled']):>12.5f} "
                  f"{np.mean(vals['B selfcon']):>12.5f} {np.mean(vc):>12.5f}", flush=True)
