"""Distributed (DDP) bootstrap and rank helpers for ``tune_cms_fullsim``.

These helpers turn a SLURM ``srun`` launch into a ``torch.distributed``
process group and provide the small rank predicates (:func:`_is_main`,
:func:`_is_dist`, :func:`_barrier`, :func:`_cleanup_distributed`) used by the
loss, fit loop and CLI. They live in their own module so both ``loss`` and
``training`` can depend on them without creating an import cycle.
"""

from __future__ import annotations

import os
import socket
import subprocess

import torch
import torch.distributed as dist

# =============================================================================
# Distributed (DDP) bootstrap
# =============================================================================
#
# Designed to be launched with ``srun`` on SLURM:
#
#     srun python -m parnassus.torch_delphes.tune_cms_fullsim ...
#
# Each ``srun`` task becomes one DDP rank. We pin one GPU per rank using
# ``SLURM_LOCALID`` (the within-node rank), which gives the requested
# ``NxN/4x4`` mapping (4 tasks per node, 4 GPUs per node) automatically.
#
# When SLURM env vars are absent the script falls back to single-process
# execution, so plain ``python -m ...`` invocations keep working.


def _init_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize the torch.distributed process group from launcher env vars.

    Returns
    -------
    (rank, world_size, local_rank, device)
        ``rank`` and ``world_size`` are global; ``local_rank`` is the
        within-node rank used to pick the CUDA device. ``device`` is
        ``cuda:local_rank`` if CUDA is available, else ``cpu``.
    """

    has_slurm_multi = (
        "SLURM_PROCID" in os.environ
        and int(os.environ.get("SLURM_NTASKS", "1")) > 1
    )
    # srun exports PMI/PMIx rank env vars; a plain shell in an allocation
    # generally does not. Requiring this in auto-mode avoids unintended
    # multi-rank init when users run without srun.
    has_launcher_rank_env = any(
        k in os.environ for k in ("PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK")
    )

    should_init = (has_slurm_multi and has_launcher_rank_env)
    if should_init:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

        # Pick a master address from the SLURM nodelist (first node).
        if "MASTER_ADDR" not in os.environ:
            nodelist = os.environ.get("SLURM_NODELIST", socket.gethostname())
            try:
                hosts = (
                    subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
                    .decode()
                    .splitlines()
                )
                os.environ["MASTER_ADDR"] = hosts[0] if hosts else socket.gethostname()
            except Exception:
                os.environ["MASTER_ADDR"] = socket.gethostname()
        os.environ.setdefault("MASTER_PORT", "29500")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            device = torch.device(f"cuda:{local_rank}")
        else:
            backend = "gloo"
            device = torch.device("cpu")

        # Passing ``device_id`` explicitly silences NCCL's "Guessing device
        # ID based on global rank" warning (which can actually hang when
        # rank-to-GPU mapping is heterogeneous, e.g. multi-node SLURM with
        # ``--gpu-bind``) and lets NCCL eagerly bind the communicator to
        # the correct GPU on this rank.
        pg_kwargs: dict = {"backend": backend, "rank": rank, "world_size": world_size}
        if backend == "nccl":
            pg_kwargs["device_id"] = device
        dist.init_process_group(**pg_kwargs)
        return rank, world_size, local_rank, device

    # Single-process fallback.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return 0, 1, 0, device


def _is_dist() -> bool:
    """True if a process group has been initialized."""
    return dist.is_available() and dist.is_initialized()


def _is_main(rank: int) -> bool:
    """True on rank 0 (the only rank that prints / writes files)."""
    return rank == 0


def _barrier() -> None:
    """No-op when not running under DDP."""
    if _is_dist():
        dist.barrier()


def _cleanup_distributed() -> None:
    """Tear down the process group."""
    if _is_dist():
        dist.destroy_process_group()
