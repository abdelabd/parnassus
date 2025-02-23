from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import awkward as ak
import numpy as np
import numpy.typing as npt
import uproot
from numpy.random import Generator

from .transform import VarTransform, VarTransformConfig

PARTICLE_VARS = [
    "pt",
    "eta",
    "phi",
    "vx",
    "vy",
    "vz",
    "class",
]


def mock_particles(
    num_events: int = 1000,
    num_particles: int = 40,
    rng: Generator | None = None,
) -> dict[str, list[npt.NDArray[np.float32]]]:
    if rng is None:
        rng = np.random.default_rng(42)
    particles = {var: rng.random((num_events, num_particles)) for var in PARTICLE_VARS}
    ind = rng.choice([True, False], size=(num_events, num_particles)).astype(bool)
    data: dict[str, list[npt.NDArray[np.float32]]] = {var: [] for var in PARTICLE_VARS}
    for i in range(num_events):
        for var in PARTICLE_VARS:
            data[var].append(particles[var][i][ind[i]])

    return data


def get_mock_root_file(
    num_events: int = 1000,
    fname: str | None = None,
    ttree_name: str = "evt_tree",
    num_particles: int = 40,
) -> str:
    rng = np.random.default_rng(42)
    truth_particles = mock_particles(num_events=num_events, num_particles=num_particles)
    # create a tempfile in a new folder
    if fname is None:
        fname = NamedTemporaryFile(suffix=".root", dir=mkdtemp()).name  # noqa: SIM115
    else:
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
    with uproot.recreate(fname) as f:
        f[ttree_name] = {
            "truth": ak.zip({var: ak.Array(val) for var, val in truth_particles.items()}),
            "eventNumber": rng.integers(0, num_events * 4, num_events),
        }

    return fname


def get_mock_transforms() -> dict[str, VarTransform]:
    var_transform_dict: dict[str, VarTransform] = {}
    for var in [*PARTICLE_VARS, "met_x", "met_y", "ht", "npart"]:
        cfg = VarTransformConfig(name=var if var != "pt" else "ptrel", mean=0, std=1)
        var_transform_dict[cfg.name] = VarTransform(cfg)
    return var_transform_dict
