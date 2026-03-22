"""TorchDelphes Pipeline for Parnassus.

This module provides integration between TorchDelphes detector simulation
and the Parnassus event processing framework. It converts TorchDelphes
tensor outputs to GenEvent objects that can be written using RootWriter.

Key components:
    - TorchDelphesPipeline: Runs detector simulation on HepMC input
    - tensor_to_gen_events: Converts tensor output to GenEvent list
    - get_torch_delphes_accessors: Returns accessors for RootWriter

Example:
    >>> from parnassus.pipelines.torch_delphes import (
    ...     TorchDelphesPipeline,
    ...     tensor_to_gen_events,
    ...     get_torch_delphes_accessors
    ... )
    >>> from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
    >>> from parnassus.writers import RootWriter
    >>>
    >>> # Run detector simulation
    >>> detector = CMSEnergyFlowDefault(debug=False)
    >>> results = detector(stable_particles)
    >>>
    >>> # Convert to GenEvent list
    >>> events = tensor_to_gen_events(
    ...     truth_tensor=all_particles,
    ...     pflow_tensor=results["EFlowObject"]
    ... )
    >>>
    >>> # Write using RootWriter
    >>> accessor_store = AccessorStore()
    >>> accessor_store.update_from_dict(get_torch_delphes_accessors())
    >>> writer_config = WriterConfig(file_path="output.root", accessor_store=accessor_store)
    >>> writer = RootWriter(writer_config)
    >>> writer.write(events)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from parnassus.configs.accessors import Accessor, AccessorStore, ParticleAccessor
from parnassus.configs.scheme import GenEvent, GenParticleCollection
from parnassus.configs.writer import WriterConfig

# Import tensor column indices
from parnassus.torch_delphes.tensor_utils import (
    ETA,
    EVENT_NUMBER,
    MASS,
    PHI,
    PID,
    PT,
    STATUS,
    X,
    Y,
    Z,
)
from parnassus.writers import RootWriter


def _fix_neutral_hadron_pid(pdg_ids: np.ndarray) -> np.ndarray:
    """Fix PID=0 (used by EFlowMerger for neutral hadrons) to a valid PDG ID.

    TorchDelphes EFlowMerger sets PID=0 for neutral hadron candidates from HCal,
    matching C++ Delphes convention. However, PID=0 is not a valid PDG ID and
    will cause issues with pid_to_class(). We map PID=0 → PID=130 (K_L^0),
    which is a common neutral hadron.

    Args:
        pdg_ids: Array of PDG IDs, may contain zeros

    Returns
    -------
        Array with PID=0 replaced by PID=130
    """
    pdg_ids = pdg_ids.copy()
    pdg_ids[pdg_ids == 0] = 130  # K_L^0 (neutral kaon)
    return pdg_ids


def tensor_to_gen_particle_collection(
    tensor: torch.Tensor, name: str, fix_neutral_hadrons: bool = False
) -> dict[int, GenParticleCollection]:
    """Convert a particle tensor to a dictionary of GenParticleCollection per event.

    Groups particles by EVENT_NUMBER and creates a GenParticleCollection for each
    unique event.

    Args:
        tensor: Particle tensor of shape (N, N_FEATURES) with columns defined by CMAP.
            Must contain EVENT_NUMBER column for grouping.
        name: Name for the GenParticleCollection (e.g., "truth", "pflow")
        fix_neutral_hadrons: If True, replace PID=0 with PID=130 (K_L^0).
            Should be True for pflow/EFlowObject tensors.

    Returns
    -------
        Dictionary mapping event_number (int) to GenParticleCollection
    """
    # Convert to numpy for easier manipulation
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    # Get unique event numbers
    event_numbers = tensor[:, EVENT_NUMBER].astype(np.int32)
    unique_events = np.unique(event_numbers)

    result: dict[int, GenParticleCollection] = {}

    for event_num in unique_events:
        mask = event_numbers == event_num
        event_tensor = tensor[mask]

        # Extract particle properties
        pt = event_tensor[:, PT].astype(np.float32)
        eta = event_tensor[:, ETA].astype(np.float32)
        phi = event_tensor[:, PHI].astype(np.float32)
        mass = event_tensor[:, MASS].astype(np.float32)
        pdg_id = event_tensor[:, PID].astype(np.int32)
        vx = event_tensor[:, X].astype(np.float32)
        vy = event_tensor[:, Y].astype(np.float32)
        vz = event_tensor[:, Z].astype(np.float32)
        status = event_tensor[:, STATUS].astype(np.int32)

        # Fix PID=0 if requested (for EFlowObject neutral hadrons)
        if fix_neutral_hadrons:
            pdg_id = _fix_neutral_hadron_pid(pdg_id)

        # Create GenParticleCollection
        # Note: class_id and charge are auto-computed from pdg_id in __post_init__
        collection = GenParticleCollection(
            name=name,
            pt=pt,
            eta=eta,
            phi=phi,
            mass=mass,
            pdg_id=pdg_id,
            vx=vx,
            vy=vy,
            vz=vz,
            status=status,
        )

        result[int(event_num)] = collection

    return result


def tensor_to_gen_events(
    truth_tensor: torch.Tensor,
    pflow_tensor: torch.Tensor,
) -> list[GenEvent]:
    """Convert TorchDelphes output tensors to a list of GenEvent objects.

    This function takes the tensor outputs from TorchDelphes and converts them
    to GenEvent objects suitable for use with RootWriter.

    Args:
        truth_tensor: Tensor containing all HepMC particles (the "Particle" branch).
            Shape: (N_truth, N_FEATURES)
        pflow_tensor: Tensor containing particle flow candidates (the "EFlowObject" branch).
            Shape: (N_pflow, N_FEATURES)

    Returns
    -------
        List of GenEvent objects, one per unique event number found in the tensors.

    Example:
        >>> # After running TorchDelphes
        >>> results = detector(stable_particles)
        >>> events = tensor_to_gen_events(
        ...     truth_tensor=all_particles,
        ...     pflow_tensor=results["EFlowObject"]
        ... )
        >>> print(f"Created {len(events)} events")
    """
    # Convert tensors to per-event GenParticleCollections
    truth_collections = tensor_to_gen_particle_collection(
        truth_tensor, name="truth", fix_neutral_hadrons=False
    )
    pflow_collections = tensor_to_gen_particle_collection(
        pflow_tensor, name="pflow", fix_neutral_hadrons=True
    )

    # Get all unique event numbers from both tensors
    all_event_nums = set(truth_collections.keys()) | set(pflow_collections.keys())

    # Create GenEvent for each event number
    events: list[GenEvent] = []

    for event_num in sorted(all_event_nums):
        # Get collections, or create empty ones if missing
        if event_num in truth_collections:
            truth_particles = truth_collections[event_num]
        else:
            # Empty collection
            truth_particles = GenParticleCollection(
                name="truth",
                pt=np.array([], dtype=np.float32),
                eta=np.array([], dtype=np.float32),
                phi=np.array([], dtype=np.float32),
            )

        if event_num in pflow_collections:
            pflow_particles = pflow_collections[event_num]
        else:
            # Empty collection
            pflow_particles = GenParticleCollection(
                name="pflow",
                pt=np.array([], dtype=np.float32),
                eta=np.array([], dtype=np.float32),
                phi=np.array([], dtype=np.float32),
            )

        # Create GenEvent
        event = GenEvent(
            event_number=event_num,
            truth_particles=truth_particles,
            pflow_particles=pflow_particles,
        )
        events.append(event)

    return events


def get_torch_delphes_accessors() -> dict[str, list[Accessor]]:
    """Get the accessor configuration for TorchDelphes output.

    Returns accessor definitions for Truth and Pflow branches with all
    particle properties (pt, eta, phi, mass, pdg_id, vx, vy, vz, class_id, charge).

    Returns
    -------
        Dictionary mapping branch names to lists of Accessor objects:
        - "Truth": Accessors for truth_particles collection
        - "Pflow": Accessors for pflow_particles collection

    Example:
        >>> accessor_store = AccessorStore()
        >>> accessor_store.update_from_dict(get_torch_delphes_accessors())
        >>> writer_config = WriterConfig(
        ...     file_path="output.root",
        ...     accessor_store=accessor_store
        ... )
    """
    # Define accessors for particle properties
    # These match the attributes of GenParticleCollection
    particle_float_vars = ["pt", "eta", "phi", "mass", "vx", "vy", "vz"]
    particle_int_vars = ["pdg_id", "class_id", "charge", "status"]

    truth_accessors: list[Accessor] = [
        ParticleAccessor(name=var, collection="truth_particles", dtype="float32")
        for var in particle_float_vars
    ] + [
        ParticleAccessor(name=var, collection="truth_particles", dtype="int32")
        for var in particle_int_vars
    ]

    pflow_accessors: list[Accessor] = [
        ParticleAccessor(name=var, collection="pflow_particles", dtype="float32")
        for var in particle_float_vars
    ] + [
        ParticleAccessor(name=var, collection="pflow_particles", dtype="int32")
        for var in particle_int_vars
    ]

    return {
        "Truth": truth_accessors,
        "Pflow": pflow_accessors,
    }


def write_torch_delphes_output(
    output_path: Path | str,
    truth_tensor: torch.Tensor,
    pflow_tensor: torch.Tensor,
) -> None:
    """Convenience function to write TorchDelphes output to a ROOT file.

    This function combines tensor_to_gen_events() and RootWriter into a
    single call for simple use cases.

    Args:
        output_path: Path to output ROOT file
        truth_tensor: Tensor containing all HepMC particles
        pflow_tensor: Tensor containing particle flow candidates

    Example:
        >>> results = detector(stable_particles)
        >>> write_torch_delphes_output(
        ...     output_path="output.root",
        ...     truth_tensor=all_particles,
        ...     pflow_tensor=results["EFlowObject"]
        ... )
    """
    from parnassus.utils.logger import setup_logger

    log = setup_logger()

    # Convert to GenEvent list
    log.info("[green]Converting tensors to GenEvent objects...")
    events = tensor_to_gen_events(truth_tensor, pflow_tensor)
    log.info(f"[green]Created {len(events)} events")

    # Setup accessor store
    accessor_store = AccessorStore()
    accessor_store.update_from_dict(get_torch_delphes_accessors())

    # Create writer config
    writer_config = WriterConfig(
        file_path=Path(output_path) if isinstance(output_path, str) else output_path,
        accessor_store=accessor_store,
    )

    # Write
    log.info(f"[green]Writing to {output_path}...")
    writer = RootWriter(writer_config)
    writer.write(events)
    log.info("[green]Write complete!")


# Export public API
__all__ = [
    "get_torch_delphes_accessors",
    "tensor_to_gen_events",
    "tensor_to_gen_particle_collection",
    "write_torch_delphes_output",
]
