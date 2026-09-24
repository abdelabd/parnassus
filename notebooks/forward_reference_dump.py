"""Step 0 of the critic-loss plan: byte-identity reference of the learnable card forward.

Runs ``CMSEnergyFlowDefault(learnable=True)`` on the first ``--n-events`` events of a
pseudodata ROOT file, on CPU in float64, with a fixed seed and a fixed thread count, and
dumps everything downstream code consumes:

- the concatenated ``EFlowObject`` tensor (all columns, stored BY COLUMN NAME so a later
  ColumnMap extension can still be compared on the shared columns),
- the five ``*ExpectedCounts`` outputs per batch,
- the per-event ``multiplicity`` / ``ht`` / ``log_ht`` from ``load_pflow_targets_from_tensor``,
- the torch RNG state after the pass,

for two passes: ``grad`` (the training path: card.train(), autograd on) and ``nograd``
(the generation path: torch.no_grad()). ``--compare A B`` checks two dumps for exact
equality on every shared column / key and exits non-zero on any difference.

Usage (login node is fine at the default size; keep --threads fixed across compares):

    .venv/bin/python notebooks/forward_reference_dump.py --out $SCRATCH/critic_loss_refs/step0_a.npz
    .venv/bin/python notebooks/forward_reference_dump.py --compare A.npz B.npz

Not a committed test: the plan's permanent T1 test replaces this once a weight toggle exists.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.torch_delphes.defaults.CMSDefault import CMSEnergyFlowDefault
from parnassus.torch_delphes.tune_cms_fullsim.config import CALO_COUNT_TERM_KEYS, COUNT_TERM_KEYS
from parnassus.torch_delphes.tune_cms_fullsim.data import (
    batch_event_ids,
    load_cms_flow_root,
    load_pflow_targets_from_tensor,
    load_pflow_targets_ragged,
    load_truth_events_ragged,
    restore_event_format,
)
from parnassus.torch_delphes.tune_cms_fullsim.dataloader import DelphesDataLoader, DelphesDataSet
from parnassus.torch_delphes import param_config as pc

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "src" / "parnassus" / "tests" / "benchmark_data" / "cms_pseudodata.root"
COUNT_KEYS = [row[0] for row in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS)]
EVENT_KEYS = ("multiplicity", "ht", "log_ht")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_pass(card, loader, *, grad: bool, seed: int) -> dict[str, np.ndarray]:
    """One full pass over ``loader``; returns flat numpy arrays keyed for the npz."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    card.train(grad)
    objects, counts, events = [], {k: [] for k in COUNT_KEYS}, {k: [] for k in EVENT_KEYS}
    n_obj_per_batch = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for batch in loader:
            truth = batch["truth_particles"]
            mask = torch.any(truth != 0, dim=-1)
            out = card(truth[mask])
            eflow = out["EFlowObject"]
            restored = restore_event_format(eflow, mask, event_ids=batch_event_ids(truth, mask))
            obs = load_pflow_targets_from_tensor(restored)
            objects.append(eflow.detach().to(torch.float64).cpu().numpy())
            n_obj_per_batch.append(eflow.shape[0])
            for k in COUNT_KEYS:
                counts[k].append(out[k].detach().to(torch.float64).cpu().numpy().reshape(1, -1))
            for k in EVENT_KEYS:
                events[k].append(obs[k].detach().to(torch.float64).cpu().numpy().reshape(-1))
    tag = "grad" if grad else "nograd"
    result: dict[str, np.ndarray] = {}
    eflow_all = np.concatenate(objects, axis=0) if objects else np.zeros((0, N_FEATURES))
    for col in ColumnMap:  # stored by NAME so a later extra column is still comparable
        result[f"{tag}/eflow/{col.name}"] = eflow_all[:, int(col)].copy()
    result[f"{tag}/n_obj_per_batch"] = np.asarray(n_obj_per_batch, dtype=np.int64)
    for k in COUNT_KEYS:
        result[f"{tag}/counts/{k}"] = np.concatenate(counts[k], axis=0)
    for k in EVENT_KEYS:
        result[f"{tag}/event/{k}"] = np.concatenate(events[k], axis=0)
    result[f"{tag}/rng_state_after"] = torch.get_rng_state().numpy().copy()
    return result


def dump(args: argparse.Namespace) -> Path:
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")

    t0 = time.time()
    arrays = load_cms_flow_root(Path(args.root_file), n_events=args.n_events)
    truth = load_truth_events_ragged(arrays)
    target = load_pflow_targets_ragged(arrays)
    del arrays
    dataset = DelphesDataSet(truth, target, device=device)
    loader = DelphesDataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    n_truth = int(sum(t.shape[0] for t in truth))
    print(f"[step0] loaded {len(dataset)} events, {n_truth} truth particles in {time.time() - t0:.1f}s")

    # Same construction as cli.py / optuna_search.py in --mode delphes: no count
    # acceptance cuts, no photon merger.
    torch.manual_seed(args.seed)
    card = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    if args.param_config:
        pc.apply_param_config(card, pc.load_param_config(args.param_config))

    result: dict[str, np.ndarray] = {}
    for grad in (True, False):
        t1 = time.time()
        result.update(_run_pass(card, loader, grad=grad, seed=args.seed))
        tag = "grad" if grad else "nograd"
        n_obj = int(result[f"{tag}/n_obj_per_batch"].sum())
        print(f"[step0] pass {tag}: {n_obj} eflow objects in {time.time() - t1:.1f}s")

    # Card parameters (flat, sorted by name) so a drift in defaults is visible too.
    flat = pc.dump_flat_config(card) if hasattr(pc, "dump_flat_config") else None
    if flat is None:
        sd = card.state_dict()
        names = sorted(sd.keys())
        result["card/param_names"] = np.asarray(names)
        result["card/param_values"] = np.concatenate(
            [sd[n].detach().to(torch.float64).reshape(-1).cpu().numpy() for n in names]
        )
    result["meta/column_names"] = np.asarray([c.name for c in ColumnMap])
    result["meta/n_features"] = np.asarray(N_FEATURES)
    result["meta/n_events"] = np.asarray(len(dataset))
    result["meta/seed"] = np.asarray(args.seed)
    result["meta/threads"] = np.asarray(args.threads)
    result["meta/batch_size"] = np.asarray(args.batch_size)
    result["meta/torch"] = np.asarray(torch.__version__)
    result["meta/git"] = np.asarray(_git_commit())
    result["meta/root_file"] = np.asarray(str(Path(args.root_file).resolve()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    digest = hashlib.md5(out.read_bytes()).hexdigest()
    print(f"[step0] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) md5 {digest}")
    return out


def compare(path_a: str, path_b: str) -> int:
    a, b = np.load(path_a, allow_pickle=False), np.load(path_b, allow_pickle=False)
    keys_a, keys_b = set(a.files), set(b.files)
    shared = sorted(k for k in keys_a & keys_b if not k.startswith("meta/"))
    only_a, only_b = sorted(keys_a - keys_b), sorted(keys_b - keys_a)
    n_bad = 0
    for k in shared:
        va, vb = a[k], b[k]
        if va.shape != vb.shape:
            print(f"  SHAPE   {k}: {va.shape} vs {vb.shape}")
            n_bad += 1
            continue
        if va.dtype.kind in "fc":
            same = np.array_equal(va, vb, equal_nan=True)
        else:
            same = np.array_equal(va, vb)
        if not same:
            n_bad += 1
            if va.dtype.kind in "fc":
                diff = np.abs(va.astype(np.float64) - vb.astype(np.float64))
                print(f"  DIFF    {k}: {int((diff != 0).sum())}/{va.size} entries differ, max |d| = {np.nanmax(diff):.3e}")
            else:
                print(f"  DIFF    {k}: {int((va != vb).sum())}/{va.size} entries differ")
    for k in only_a:
        print(f"  ONLY-A  {k}")
    for k in only_b:
        print(f"  ONLY-B  {k}  (new key; ignored)")
    meta = {k: (str(a[k]) if k in keys_a else "-", str(b[k]) if k in keys_b else "-") for k in
            sorted(k for k in keys_a | keys_b if k.startswith("meta/"))}
    for k, (ma, mb) in meta.items():
        flag = "" if ma == mb else "   <- differs"
        print(f"  META    {k}: {ma} | {mb}{flag}")
    status = "IDENTICAL" if n_bad == 0 else f"{n_bad} MISMATCHING KEYS"
    print(f"[step0] compare: {len(shared)} shared keys, {status}")
    return 0 if n_bad == 0 else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-file", default=str(DEFAULT_ROOT))
    p.add_argument("--n-events", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--param-config", default=None, help="optional param yaml applied to the card")
    p.add_argument(
        "--out",
        default=str(Path(os.environ.get("SCRATCH", "/tmp")) / "critic_loss_refs" / "step0_reference.npz"),
    )
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two dumps and exit")
    args = p.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    dump(args)


if __name__ == "__main__":
    main()
