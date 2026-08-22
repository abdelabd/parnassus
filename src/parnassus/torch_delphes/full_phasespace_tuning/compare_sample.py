r"""Per-species distributions on an independent sample: target vs CMS-default vs fitted card.

Closure plot of the sequential fit, in the style of
:mod:`plotting_scripts.plot_distribution` (same pages and drawing): the trainee is run on
the truth particles of ``--sample`` at the CMS card constructor defaults ("initial") and at
the fitted values of ``--fitted-config`` (e.g. ``<OUT_BASE>/fitted_config.yaml``, "tuned"),
both overlaid on the sample's pflow target -- log pT / log E / eta per species, the
leading-2 pair-mass response of electrons / muons / charged hadrons, and in addition the
dilepton masses m_ee / m_mumu (leading 2 same-flavour leptons) and the four-lepton mass
m_4l (leading 4 e/mu). One PDF page per (species, observable); paper layout as in
plot_distribution (no page title, x label only with its ``X_LABEL`` knob -- both are added
in LaTeX; the page order is printed).

    python -m parnassus.torch_delphes.full_phasespace_tuning.compare_sample \\
        --sample <SAMPLE_DIR>/pseudo_data_100k_param_config_all_HZZ4l.root \\
        --fitted-config doc/figure_sequential/fitted_config.yaml

Delphes mode only (no acceptance cuts, no photon merger -- what the sequential fit ran
with); all events by default; CPU; the same RNG seed before both passes.
"""

import argparse
from pathlib import Path

import torch
import uproot
from matplotlib.backends.backend_pdf import PdfPages

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.plotting_scripts.plot_distribution import (
    BATCH_SIZE,
    OBSERVABLES,
    PAIR_SPECIES,
    PAIR_XLABEL,
    SEED,
    SPECIES,
    draw_page,
    values,
)
from parnassus.torch_delphes.tune_cms_fullsim.data import (
    load_cms_flow_root,
    load_pflow_targets_from_tensor,
    restore_event_format,
)
from parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results import (
    _build_val_dataloader,
    _set_trainee_from_snapshot,
    _trainee_observables,
)

LEPTON_MASS = {11: 0.000511, 13: 0.1056584}
# Mass pages: (key, title, xlabel); key is also the entry of the per-event mass dict. The
# title is only printed in the page index; xlabel is drawn only with plot_distribution.X_LABEL.
MASS_PAGES = (
    ("mass_ee", "Electron pair mass (leading 2)", r"$m_{ee}$ [GeV]"),
    ("mass_mumu", "Muon pair mass (leading 2)", r"$m_{\mu\mu}$ [GeV]"),
    ("mass_4l", r"Four-lepton mass (leading 4 e/$\mu$)", r"$m_{4\ell}$ [GeV]"),
)


def leading_n_mass(pt, eta, phi, sel, obj_mass, n):
    """Invariant mass of the ``n`` highest-pt selected objects per event.

    Padded ``(n_events, max_n)`` inputs, ``obj_mass`` per object; events with fewer than
    ``n`` selected objects are skipped.

    Returns
    -------
    torch.Tensor
        ``(n_kept_events,)`` masses in GeV.
    """
    ev = (sel.sum(dim=1) >= n).nonzero(as_tuple=True)[0]
    if ev.numel() == 0:
        return pt.new_zeros(0)
    ranked = torch.where(sel, pt, torch.full_like(pt, float("-inf")))
    idx = ranked.topk(n, dim=1).indices[ev]
    rows = ev.unsqueeze(1)
    pt_n, eta_n, phi_n, m_n = pt[rows, idx], eta[rows, idx], phi[rows, idx], obj_mass[rows, idx]
    px, py, pz = pt_n * torch.cos(phi_n), pt_n * torch.sin(phi_n), pt_n * torch.sinh(eta_n)
    e = torch.sqrt(px * px + py * py + pz * pz + m_n * m_n)
    m2 = e.sum(dim=1) ** 2 - px.sum(dim=1) ** 2 - py.sum(dim=1) ** 2 - pz.sum(dim=1) ** 2
    return torch.sqrt(m2.clamp(min=0.0))


def event_masses(obs):
    """Per-event lepton masses of one batch's padded object dict.

    ``obs`` holds ``pt``/``eta``/``phi``/``pid`` as ``(n_events, max_n)`` tensors; padding
    and efficiency-killed slots have ``pt == 0``.

    Returns
    -------
    dict[str, torch.Tensor]
        ``mass_ee`` / ``mass_mumu`` (leading 2 same-flavour leptons) and ``mass_4l``
        (leading 4 e/mu), each over the events with enough leptons.
    """
    pt, eta, phi, pid = obs["pt"], obs["eta"], obs["phi"], obs["pid"].abs()
    real = pt > 0
    out = {}
    for key, lep in (("mass_ee", 11), ("mass_mumu", 13)):
        sel = real & (pid == lep)
        out[key] = leading_n_mass(pt, eta, phi, sel, torch.full_like(pt, LEPTON_MASS[lep]), 2)
    is_lep = real & ((pid == 11) | (pid == 13))
    obj_mass = torch.where(pid == 11, LEPTON_MASS[11], LEPTON_MASS[13]).to(pt.dtype)
    out["mass_4l"] = leading_n_mass(pt, eta, phi, is_lep, obj_mass, 4)
    return out


def run_masses(card, loader):
    """One pass of ``card`` over ``loader`` -> per-event masses of both sides.

    Seed the RNG before calling: with the same seed and loader as
    ``_trainee_observables`` the reco is identical.

    Returns
    -------
    tuple[dict, dict]
        ``(pred, target)`` mass dicts (:func:`event_masses` keys, flat tensors).
    """
    acc = {"pred": {}, "target": {}}
    with torch.no_grad():
        for batch in loader:
            truth = batch["truth_particles"]
            mask = torch.any(truth != 0, dim=-1)
            pred = load_pflow_targets_from_tensor(
                restore_event_format(card(truth[mask])["EFlowObject"], mask)
            )
            for side, obs in (("pred", pred), ("target", batch)):
                for key, val in event_masses(obs).items():
                    acc[side].setdefault(key, []).append(val.cpu())
    return tuple({k: torch.cat(v) for k, v in acc[s].items()} for s in ("pred", "target"))


def run_card(card, params, loader):
    """Load ``params`` (physical values) into ``card`` and run it over ``loader``.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        ``(obs, target_obs, masses, target_masses)``; observables as in plot_distribution
        (pair responses included), masses from :func:`run_masses`.
    """
    _set_trainee_from_snapshot(card, params)
    torch.manual_seed(SEED)
    obs, target = _trainee_observables(card, loader)
    torch.manual_seed(SEED)
    masses, target_masses = run_masses(card, loader)
    return obs, target, masses, target_masses


def main():
    """Plot the comparison to ``--output`` (default ``<fitted dir>/distributions_<sample>.pdf``)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample", required=True, type=Path, help="ROOT file (truth + pflow target)")
    ap.add_argument("--fitted-config", required=True, type=Path, help="fitted param config")
    ap.add_argument(
        "--init-config", type=Path, default=None, help="'initial' card; default: CMS defaults"
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--n-events", type=int, default=-1, help="first N events; <= 0 = all")
    args = ap.parse_args()

    with uproot.open(str(args.sample)) as f:
        n_total = int(f["event_tree"].num_entries)
    n_events = n_total if args.n_events <= 0 else min(args.n_events, n_total)
    arrays = load_cms_flow_root(args.sample, n_events=n_events)
    loader = _build_val_dataloader(arrays, BATCH_SIZE, torch.device("cpu"))  # delphes: no cuts
    card = CMSEnergyFlowDefault(debug=False, learnable=True)

    def physical(path):  # {scalar key: physical value} of a (partial) config over the defaults
        flat = (
            pc.load_param_config_over_defaults(path, card) if path else pc.card_default_config(card)
        )
        return {k: s["value"] for k, s in flat.items()}

    print(
        f"{n_events} events of {args.sample}; initial = {args.init_config or 'CMS card defaults'}"
    )
    initial, target, m_initial, m_target = run_card(card, physical(args.init_config), loader)
    tuned, _, m_tuned, _ = run_card(card, physical(args.fitted_config), loader)

    output = args.output or args.fitted_config.parent / f"distributions_{args.sample.stem}.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    page = 0
    with PdfPages(output) as pdf:
        samples = {"target": target, "initial": initial, "tuned": tuned}
        for title, pid in SPECIES.items():
            for key, xlabel in OBSERVABLES.items():
                arrays = {n: values(o, pid, key) for n, o in samples.items()}
                if not sum(len(v) for v in arrays.values()):
                    continue  # species absent from this sample
                draw_page(pdf, arrays, xlabel=xlabel)
                page += 1
                print(f"page {page}: {title} {key}")
        for title, pid in PAIR_SPECIES.items():
            key = f"pair_r:{pid}"
            if not all(key in o for o in samples.values()):
                continue  # class has no pairs in this sample
            draw_page(pdf, {n: o[key].numpy() for n, o in samples.items()}, xlabel=PAIR_XLABEL)
            page += 1
            print(f"page {page}: {title} pair-mass response (leading 2)")
        masses = {"target": m_target, "initial": m_initial, "tuned": m_tuned}
        for key, title, xlabel in MASS_PAGES:
            arrays = {n: m[key].numpy() for n, m in masses.items() if key in m}
            if len(arrays) < 3 or not all(len(v) for v in arrays.values()):
                continue  # not enough leptons on some side
            draw_page(pdf, arrays, ylabel="Events", xlabel=xlabel)
            page += 1
            print(f"page {page}: {title}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
