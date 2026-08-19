"""DDP autograd-graph parity in the ``tune_cms_fullsim`` loss gathers.

The gathers in :mod:`parnassus.torch_delphes.tune_cms_fullsim.loss` run **only**
under DDP (``per_pid_wasserstein_1d_loss_distributed`` returns to the
single-process loss when ``not _is_dist()``), so nothing single-process exercises
them. These tests therefore launch real multi-rank gloo jobs on CPU.

What they protect: ``torch.distributed.nn.functional.all_gather`` registers its
backward (``REDUCE_SCATTER``) node **only when its input requires grad**. If ranks
disagree, the FORWARD still matches -- every rank gathers the same padded shapes --
and the job deadlocks in BACKWARD, surfacing minutes later as an opaque NCCL
watchdog timeout. A forward-only test passes against that bug, which is exactly how
it shipped, so these tests assert on the autograd graph and on backward completing.

The concrete regression: ``compute_pair_masses`` omits a pid it found no pair for,
and whether a rank has a pair depends on its own data shard. The pair-mass block
must therefore build its empty pred-side fallback by slicing a graph tensor to zero
length -- empty, but still carrying ``grad_fn`` -- not with ``torch.zeros(0)``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO_SRC = str(Path(__file__).resolve().parents[2])
_TIMEOUT_S = 300  # a regression HANGS; the timeout is what turns that into a failure


def _run_ranks(mode: str, nproc: int = 2) -> subprocess.CompletedProcess:
    """Run this file as a ``torch.distributed`` job of ``nproc`` gloo ranks."""
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",  # force gloo/CPU: the bug is backend-agnostic
        "PYTHONPATH": _REPO_SRC + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    return subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", f"--nproc-per-node={nproc}",
            str(Path(__file__).resolve()), mode,
        ],
        capture_output=True, text=True, timeout=_TIMEOUT_S, env=env, check=False,
    )


pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed with the gloo backend is required",
)


def test_pair_mass_backward_completes_when_a_rank_has_no_pair():
    """The real regression: rank 0 has a muon pair, rank 1 has none.

    Drives ``per_pid_wasserstein_1d_loss_distributed(..., pair_mass=True)`` and
    calls ``.backward()``. Before the fix this deadlocked -- rank 1's empty
    ``torch.zeros(0)`` fallback carried no ``grad_fn``, so rank 1 skipped the
    backward REDUCE_SCATTER that rank 0 still issued.
    """
    try:
        proc = _run_ranks("loss")
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"DDP pair-mass backward deadlocked (no exit in {_TIMEOUT_S}s). A rank "
            "with no pair for a hypothesis is skipping a backward collective."
        )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "PAIR_MASS_BACKWARD_OK" in proc.stdout, proc.stdout


def test_gather_rejects_autograd_graph_mismatch_by_name():
    """The guard in ``_all_gather_varlen``: a graph mismatch must fail loudly.

    Without it the same mistake at any future call site is a silent multi-minute
    hang instead of an error naming the offending ranks.
    """
    try:
        proc = _run_ranks("guard")
    except subprocess.TimeoutExpired:
        pytest.fail(f"grad-parity guard did not fire; job hung for {_TIMEOUT_S}s")
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "GUARD_FIRED_OK" in proc.stdout, proc.stdout


def test_pair_mass_loss_matches_single_process():
    """DDP must not just RUN, it must agree with the single-GPU answer.

    ``per_pid_wasserstein_1d_loss_distributed`` promises that gathering makes "DDP
    and single-process results identical". The pair-mass term has never executed a
    collective in practice -- the whole gather path is DDP-only and the feature was
    developed on one GPU -- so that promise is untested for it. This computes the
    loss on 8 events in one process and on the same 8 events split across 2 ranks,
    and requires them to agree.
    """
    try:
        proc = _run_ranks("equiv")
    except subprocess.TimeoutExpired:
        pytest.fail(f"DDP/single-process equivalence check hung for {_TIMEOUT_S}s")
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "EQUIV_OK" in proc.stdout, proc.stdout


# =============================================================================
# Worker side: executed inside each spawned rank, not by pytest.
# =============================================================================

def _build_event_batch(n_events: int, n_muons_in_first_event: int):
    """Minimal pred/target/truth trio for the pair-mass path.

    ``n_muons_in_first_event`` of 2 gives this rank a muon pair; 0 gives it none,
    so ``compute_pair_masses`` omits pid 13 and the empty fallback is taken.
    """
    from parnassus.data.particle_io import N_FEATURES, ColumnMap

    n_slots = 4
    pid = torch.zeros(n_events, n_slots, dtype=torch.float64)
    pt = torch.zeros(n_events, n_slots, dtype=torch.float64)
    eta = torch.zeros(n_events, n_slots, dtype=torch.float64)
    phi = torch.zeros(n_events, n_slots, dtype=torch.float64)

    # One charged hadron per event everywhere, so the per-particle observables are
    # non-empty on every rank and only the PAIR hypothesis differs.
    pid[:, 0], pt[:, 0], eta[:, 0], phi[:, 0] = 211.0, 12.0, 0.30, 0.10

    for k in range(n_muons_in_first_event):
        # Reco pids are UNSIGNED in this schema ({11.0, 13.0, 22.0} in real files) and
        # compute_pair_masses selects with `pid == p`, no abs -- so a signed -13 here
        # would match nothing and quietly make this test vacuous.
        pid[0, 1 + k] = 13.0
        pt[0, 1 + k] = 40.0 + 5.0 * k
        eta[0, 1 + k] = 0.20 + 0.30 * k
        phi[0, 1 + k] = 0.50 + 1.40 * k

    truth = torch.zeros(n_events, n_slots, N_FEATURES, dtype=torch.float64)
    truth[..., ColumnMap.PID] = pid
    truth[..., ColumnMap.CHARGE] = torch.where(pid != 0, torch.ones_like(pid), torch.zeros_like(pid))
    truth[..., ColumnMap.PT] = pt
    truth[..., ColumnMap.ETA] = eta
    truth[..., ColumnMap.PHI] = phi

    def _obs(scale: torch.Tensor) -> dict[str, torch.Tensor]:
        # `scale` is the graph leaf; every differentiable observable flows from it,
        # mirroring how the real pred hangs off the learnable card params.
        s_pt = pt * scale
        return {
            "pid": pid,
            "pt": s_pt,
            "eta": eta,
            "phi": phi,
            "log_pt": torch.log(s_pt.clamp_min(1e-6)),
            "log_E": torch.log((s_pt * torch.cosh(eta)).clamp_min(1e-6)),
            "log_ht": s_pt.sum(dim=1),
        }

    return truth, _obs


def _worker_loss() -> int:
    import torch.distributed as dist
    from parnassus.torch_delphes.tune_cms_fullsim.loss import (
        attach_truth_pair_lnm,
        per_pid_wasserstein_1d_loss_distributed,
    )

    dist.init_process_group("gloo")
    rank = dist.get_rank()

    # THE ASYMMETRY: rank 0 finds a muon pair, rank 1 finds none.
    truth, obs = _build_event_batch(n_events=6, n_muons_in_first_event=2 if rank == 0 else 0)

    scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    pred = obs(scale)
    target = {k: v.detach() for k, v in obs(torch.tensor(1.07, dtype=torch.float64)).items()}
    attach_truth_pair_lnm(truth, pred, target)

    loss = per_pid_wasserstein_1d_loss_distributed(pred, target, pair_mass=True)
    loss.backward()  # deadlocks here if the ranks' graphs disagree

    assert scale.grad is not None, "no gradient reached the leaf"
    assert torch.isfinite(scale.grad).all(), f"non-finite grad {scale.grad}"
    dist.barrier()
    if rank == 0:
        print("PAIR_MASS_BACKWARD_OK", flush=True)
    dist.destroy_process_group()
    return 0


def _worker_guard() -> int:
    import torch.distributed as dist
    from parnassus.torch_delphes.tune_cms_fullsim.loss import _all_gather_varlen

    dist.init_process_group("gloo")
    rank = dist.get_rank()

    leaf = torch.randn(4, dtype=torch.float64, requires_grad=True)
    # Rank 0 on the graph, rank 1 a bare constant -- the exact mistake the guard exists for.
    local = (leaf * 2.0) if rank == 0 else torch.zeros(0, dtype=torch.float64)

    try:
        _all_gather_varlen(local, differentiable=True)
    except RuntimeError as exc:
        msg = str(exc)
        assert "autograd graph" in msg, msg
        assert "requires_grad=True on ranks [0]" in msg, msg
        if rank == 0:
            print("GUARD_FIRED_OK", flush=True)
        dist.destroy_process_group()
        return 0

    print(f"[rank {rank}] FAIL: grad-parity mismatch was not rejected", flush=True)
    dist.destroy_process_group()
    return 1


def _build_mixed_batch(n_events: int):
    """Events carrying muon pairs in two events and an ELECTRON pair in only one.

    Split across 2 ranks the electron hypothesis lands entirely on rank 0, so the
    asymmetric fallback (the fixed path) is exercised inside the equivalence check
    rather than only in isolation.
    """
    from parnassus.data.particle_io import N_FEATURES, ColumnMap

    n_slots = 4
    pid = torch.zeros(n_events, n_slots, dtype=torch.float64)
    pt = torch.zeros(n_events, n_slots, dtype=torch.float64)
    eta = torch.zeros(n_events, n_slots, dtype=torch.float64)
    phi = torch.zeros(n_events, n_slots, dtype=torch.float64)

    ev = torch.arange(n_events, dtype=torch.float64)
    pid[:, 0], pt[:, 0] = 211.0, 10.0 + ev
    eta[:, 0], phi[:, 0] = 0.10 + 0.05 * ev, 0.20 + 0.10 * ev

    def _pair(i, p, pt0, pt1):
        pid[i, 1], pid[i, 2] = p, p
        pt[i, 1], pt[i, 2] = pt0, pt1
        eta[i, 1], eta[i, 2] = 0.20, 0.90
        phi[i, 1], phi[i, 2] = 0.40, 2.10

    _pair(0, 13.0, 45.0, 38.0)                       # rank 0 shard
    _pair(n_events - 2, 13.0, 51.0, 33.0)            # rank 1 shard
    _pair(1, 11.0, 27.0, 22.0)                       # rank 0 ONLY -> asymmetric

    truth = torch.zeros(n_events, n_slots, N_FEATURES, dtype=torch.float64)
    truth[..., ColumnMap.PID] = pid
    truth[..., ColumnMap.CHARGE] = torch.where(pid != 0, torch.ones_like(pid), torch.zeros_like(pid))
    truth[..., ColumnMap.PT], truth[..., ColumnMap.ETA], truth[..., ColumnMap.PHI] = pt, eta, phi
    return truth, pid, pt, eta, phi


def _obs_from(pid, pt, eta, phi, scale):
    s_pt = pt * scale
    return {
        "pid": pid, "pt": s_pt, "eta": eta, "phi": phi,
        "log_pt": torch.log(s_pt.clamp_min(1e-6)),
        "log_E": torch.log((s_pt * torch.cosh(eta)).clamp_min(1e-6)),
        "log_ht": s_pt.sum(dim=1),
    }


def _worker_equiv() -> int:
    import torch.distributed as dist
    from parnassus.torch_delphes.tune_cms_fullsim.loss import (
        attach_truth_pair_lnm,
        per_pid_wasserstein_1d_loss,
        per_pid_wasserstein_1d_loss_distributed,
    )

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    N = 8
    truth, pid, pt, eta, phi = _build_mixed_batch(N)
    one = torch.tensor(1.0, dtype=torch.float64)
    tgt_scale = torch.tensor(1.09, dtype=torch.float64)

    # Reference: the plain (non-dist) loss on ALL events. No collectives.
    full_pred, full_tgt = _obs_from(pid, pt, eta, phi, one), _obs_from(pid, pt, eta, phi, tgt_scale)
    full_tgt = {k: v.detach() for k, v in full_tgt.items()}
    attach_truth_pair_lnm(truth, full_pred, full_tgt)
    reference = float(per_pid_wasserstein_1d_loss(full_pred, full_tgt, pair_mass=True))

    # This rank's contiguous shard, through the DDP path.
    per = N // world
    sl = slice(rank * per, (rank + 1) * per)
    s_truth = truth[sl]
    s_pred = _obs_from(pid[sl], pt[sl], eta[sl], phi[sl], one)
    s_tgt = {k: v.detach() for k, v in _obs_from(pid[sl], pt[sl], eta[sl], phi[sl], tgt_scale).items()}
    attach_truth_pair_lnm(s_truth, s_pred, s_tgt)
    distributed = float(per_pid_wasserstein_1d_loss_distributed(s_pred, s_tgt, pair_mass=True))

    if rank == 0:
        rel = abs(distributed - reference) / max(abs(reference), 1e-30)
        print(f"single-process={reference!r}  distributed={distributed!r}  rel_diff={rel:.3e}", flush=True)
        assert rel < 1e-9, (
            f"DDP disagrees with single-process: {distributed!r} vs {reference!r} "
            f"(rel {rel:.3e}). per_pid_wasserstein_1d_loss_distributed promises they match."
        )
        print("EQUIV_OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit({
        "loss": _worker_loss, "guard": _worker_guard, "equiv": _worker_equiv,
    }[sys.argv[1]]())
