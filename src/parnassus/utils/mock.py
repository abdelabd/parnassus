from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from typing import final

import awkward as ak
import numpy as np
import numpy.typing as npt
import pyhepmc
import uproot
from numpy.random import Generator
from torch import Tensor, nn
from torch.export import Dim, export, save

from . import pid_to_class
from .transform import VarTransform, VarTransformConfig

PARTICLE_VARS = [
    "pt",
    "eta",
    "phi",
    "pdgId",
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
    particles: dict[str, npt.NDArray[np.float32 | np.int32]] = {}
    for var in PARTICLE_VARS:
        if var == "pdgId":
            value = rng.choice(
                [-11, 11, -13, 13, 211, -211, 111, 130, 22],
                size=(num_events, num_particles),
                replace=True,
            ).astype(np.int32)
        elif var == "class":
            value = np.array(
                [
                    pid_to_class(particles["pdgId"][i, j])
                    for i in range(num_events)
                    for j in range(num_particles)
                ],
                dtype=np.float32,
            ).reshape(num_events, num_particles)
            # value = rng.integers(0, 5, size=(num_events, num_particles)).astype(np.int32)
        else:
            value = rng.random((num_events, num_particles)).astype(np.float32)
        particles[var] = value
    ind = rng.choice([True, False], size=(num_events, num_particles)).astype(bool)
    data: dict[str, list[npt.NDArray[np.float32]]] = {var: [] for var in PARTICLE_VARS}
    for i in range(num_events):
        for var in PARTICLE_VARS:
            data[var].append(particles[var][i][ind[i]])

    return data


def get_4momentum(
    pt: npt.NDArray[np.float32] | float,
    y: npt.NDArray[np.float32] | float,
    phi: npt.NDArray[np.float32] | float,
    mass: npt.NDArray[np.float32] | float,
) -> npt.NDArray[np.float32]:
    mt = np.sqrt(pt**2 + mass**2)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = mt * np.sinh(y)
    e = mt * np.cosh(y)
    return np.array([px, py, pz, e], dtype=np.float32)


def getParticleHepMC(
    pt: npt.NDArray[np.float32] | float,
    y: npt.NDArray[np.float32] | float,
    phi: npt.NDArray[np.float32] | float,
    pid: npt.NDArray[np.float32] | float,
    status: int = 1,
) -> pyhepmc.GenParticle:
    p = pyhepmc.GenParticle()
    p.momentum = pyhepmc.FourVector(get_4momentum(pt, y, phi, 0))
    p.pid = int(pid)
    p.status = status
    return p


def getVertexHepMC(
    vx: npt.NDArray[np.float32] | float,
    vy: npt.NDArray[np.float32] | float,
    vz: npt.NDArray[np.float32] | float,
) -> pyhepmc.GenVertex:
    v = pyhepmc.GenVertex()
    v.position = pyhepmc.FourVector([vx, vy, vz, 0])
    return v


def getEventHepMC(event_data: list[npt.NDArray[np.float32]], event_number: int) -> pyhepmc.GenEvent:
    event = pyhepmc.GenEvent()
    vtx_dict: dict[int, pyhepmc.GenVertex] = {}
    for pt, y, phi, pid, vx, vy, vz in zip(*event_data, strict=True):
        particle = getParticleHepMC(pt, y, phi, pid)

        vtx = vtx_dict.get(hash((vx, vy, vz)), None)
        if vtx is None:
            vtx = getVertexHepMC(vx, vy, vz)
            particle_in = getParticleHepMC(0.0001, 0, 0, 0, 0)
            event.add_particle(particle_in)
            vtx.add_particle_in(particle_in)
            vtx_dict[hash((vx, vy, vz))] = vtx
        vtx.add_particle_out(particle)
        event.add_particle(particle)
    for vtx in vtx_dict.values():
        event.add_vertex(vtx)
    event.event_number = event_number
    return event


def get_mock_root_file(
    num_events: int = 1000,
    fname: str | None = None,
    ttree_name: str = "evt_tree",
    num_particles: int = 40,
) -> str:
    rng = np.random.default_rng(42)
    truth_particles = mock_particles(num_events=num_events, num_particles=num_particles)
    event_numbers = rng.integers(0, num_events * 4, num_events)
    # create a tempfile in a new folder
    if fname is None:
        fname = NamedTemporaryFile(suffix=".root", dir=mkdtemp()).name  # noqa: SIM115
    else:
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
    with uproot.recreate(fname) as f:
        f[ttree_name] = {
            "truth": ak.zip({var: ak.Array(val) for var, val in truth_particles.items()}),
            "eventNumber": event_numbers,
        }

    return fname


def get_mock_hepmc_file(
    num_events: int = 1000,
    fname: str | None = None,
    num_particles: int = 40,
) -> str:
    rng = np.random.default_rng(42)
    truth_particles = mock_particles(num_events=num_events, num_particles=num_particles)
    event_numbers = rng.integers(0, num_events * 4, num_events)
    # create a tempfile in a new folder
    if fname is None:
        fname = NamedTemporaryFile(suffix=".hepmc", dir=mkdtemp()).name  # noqa: SIM115
    else:
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
    events = [
        getEventHepMC(
            [truth_particles[key][i] for key in ["pt", "eta", "phi", "pdgId", "vx", "vy", "vz"]],
            int(event_numbers[i]),
        )
        for i in range(num_events)
    ]
    with pyhepmc.open(fname, "w") as f:
        for event in events:
            f.write(event)
    return fname


def get_mock_transforms() -> dict[str, VarTransform]:
    var_transform_dict: dict[str, VarTransform] = {}
    for var in [*PARTICLE_VARS, "met_x", "met_y", "ht", "npart"]:
        cfg = VarTransformConfig(name=var if var != "pt" else "ptrel", mean=0, std=1)
        var_transform_dict[cfg.name] = VarTransform(cfg)
    return var_transform_dict


@final
class MockModel(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.net = nn.Identity()
        self.mode = mode

    def forward(
        self, fs_data: Tensor, tr_data: Tensor, mask: Tensor, timestep: Tensor, global_data: Tensor
    ) -> Tensor:
        if self.mode == "evt":
            return (
                self.net(fs_data)
                + tr_data.sum(dim=(1, 2)).view(-1, 1)
                + timestep.view(-1, 1)
                + global_data.sum(dim=-1).view(-1, 1)
                + mask.sum(dim=1).view(-1, 1)
            )
        return (
            self.net(fs_data)
            + tr_data.sum(dim=(1, 2)).view(-1, 1, 1)
            + timestep.view(-1, 1, 1)
            + global_data.sum(dim=-1).view(-1, 1, 1)
            + mask.sum(dim=(1, 2)).view(-1, 1, 1)
        )


def get_mock_input_data(mode: str = "evt") -> dict[str, Tensor]:
    rng = np.random.default_rng(42)
    BS, L = 2, 400
    assert mode in {"evt", "part"}, "Mode should be either evt or part."
    if mode == "evt":
        return {
            "fs_data": Tensor(rng.random((BS, 4))),
            "tr_data": Tensor(rng.random((BS, L, 11))),
            "mask": Tensor(rng.random((BS, L))).bool(),
            "global_data": Tensor(rng.random((BS, 16))),
            "timestep": Tensor(rng.random((BS,))),
        }
    return {
        "fs_data": Tensor(rng.random((BS, L, 11))),
        "tr_data": Tensor(rng.random((BS, L, 11))),
        "mask": Tensor(rng.random((BS, L, 2))).bool(),
        "global_data": Tensor(rng.random((BS, 20))),
        "timestep": Tensor(rng.random((BS,))),
    }


def get_mock_model_file(fname: str | None = None, mode: str = "evt") -> str:
    if fname is None:
        fname = NamedTemporaryFile(suffix=".pt", dir=mkdtemp()).name  # noqa: SIM115
    else:
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
    model = MockModel(mode=mode)
    mock_data = get_mock_input_data(mode=mode)
    batch = Dim("batch", min=1, max=2048)
    program = export(
        model,
        (
            mock_data["fs_data"],
            mock_data["tr_data"],
            mock_data["mask"],
            mock_data["timestep"],
            mock_data["global_data"],
        ),
        dynamic_shapes={
            "fs_data": {0: batch},
            "tr_data": {0: batch},
            "mask": {0: batch},
            "global_data": {0: batch},
            "timestep": {0: batch},
        },
    )
    save(program, fname)
    return fname
