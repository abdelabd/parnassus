from dataclasses import dataclass, field

import awkward as ak
import numpy as np
import numpy.typing as npt
from typing_extensions import override

from parnassus.utils import pid_to_class


@dataclass(slots=True)
class GenParticleCollection:
    """Class storing information about a collection of generic particles.

    This class represents a collection of generic particles
    and provides methods to access and manipulate their properties.

    Attributes
    ----------
        name (str): The name of the particle collection.
        pt (np.ndarray): The transverse momentum of the particles.
        eta (np.ndarray): The pseudorapidity of the particles.
        phi (np.ndarray): The azimuthal angle of the particles.
        mass (np.ndarray): The mass of the particles.
        pdg_id (np.ndarray): The PDG ID of the particles.
        class_id (np.ndarray): The class ID of the particles.
        n (int): The number of particles in the collection.
        charge (np.ndarray): The charge of the particles.
    """

    # Properties
    name: str
    num_particles: int = field(init=False)
    pt: npt.NDArray[np.float32]
    eta: npt.NDArray[np.float32]
    phi: npt.NDArray[np.float32]
    mass: npt.NDArray[np.float32] | None = None
    pdg_id: npt.NDArray[np.int32] | None = None
    particle_jet_idx: npt.NDArray[np.int32] | None = None
    vx: npt.NDArray[np.float32] | None = None
    vy: npt.NDArray[np.float32] | None = None
    vz: npt.NDArray[np.float32] | None = None

    # Additional properties
    class_id: npt.NDArray[np.int32] | None = None
    charge: npt.NDArray[np.int32] | None = None
    status: npt.NDArray[np.int32] | None = None

    def __post_init__(self):
        self.num_particles = len(self.pt)
        if self.mass is None:
            self.mass = np.zeros_like(self.pt)
        if self.pdg_id is not None:
            self.class_id = np.array([pid_to_class(el) for el in self.pdg_id], dtype=np.int32)
            self.charge = np.array([np.sign(el) for el in self.pdg_id], dtype=np.int32)
        for key in self.__slots__:
            if key in {"name", "num_particles"}:
                continue
            attr = self.__getattribute__(key)
            if attr is None:
                continue
            attr_len = len(attr)
            assert attr_len == self.num_particles, (
                f"Assumed length of each features be {self.num_particles}, got"
                f" {attr_len} for {key} attribute"
            )

    def __len__(self):
        return self.num_particles

    @override
    def __repr__(self) -> str:
        return f"{self.name}"

    def get4vecs(self) -> npt.NDArray[np.float32]:
        assert self.mass is not None
        return np.stack([self.pt, self.eta, self.phi, self.mass], axis=-1)

    def get4vecs_awkward(self) -> ak.Array:
        return ak.Array(
            {
                "px": self.pt * np.cos(self.phi),
                "py": self.pt * np.sin(self.phi),
                "pz": self.pt * np.sinh(self.eta),
                "E": self.pt * np.cosh(self.eta),
            },
            with_name="Momentum4D",
        )

    def __getitem__(self, idx: int):
        assert idx < self.num_particles, f"Index {idx} out of range"


@dataclass(slots=True)
class GenJetCollection:
    """Class storing information about generic jet collection."""

    # Jet properties
    name: str
    num_jets: int = field(init=False)
    pt: npt.NDArray[np.float32]
    eta: npt.NDArray[np.float32]
    phi: npt.NDArray[np.float32]
    mass: npt.NDArray[np.float32] | None = None

    jec: npt.NDArray[np.float32] | None = None
    d2: npt.NDArray[np.float32] | None = None
    c2: npt.NDArray[np.float32] | None = None

    def __post_init__(self):
        self.num_jets = len(self.pt)
        for key in self.__slots__:
            if key in {"name", "num_jets"}:
                continue
            attr = self.__getattribute__(key)
            if attr is None:
                continue
            attr_len = len(attr)
            assert attr_len == self.num_jets, (
                f"Assumed length of each features be {self.num_jets}, got"
                f" {attr_len} for {key} attribute"
            )

    def __len__(self):
        return len(self.pt)

    @override
    def __repr__(self) -> str:
        return f"{self.name} with {len(self)} jets"


@dataclass(slots=True)
class GenEvent:
    """Class storing properties of event."""

    # Event properties
    event_number: int

    truth_particles: GenParticleCollection
    pflow_particles: GenParticleCollection

    truth_jets: GenJetCollection | None = None
    pflow_jets: GenJetCollection | None = None

    # Event features
    truth_ht: np.float32 = field(init=False)
    truth_met_x: np.float32 = field(init=False)
    truth_met_y: np.float32 = field(init=False)

    pflow_ht: np.float32 = field(init=False)
    pflow_met_x: np.float32 = field(init=False)
    pflow_met_y: np.float32 = field(init=False)

    def __post_init__(self):
        self.truth_ht = np.sum(self.truth_particles.pt)
        self.truth_met_x = np.sum(self.truth_particles.pt * np.cos(self.truth_particles.phi))
        self.truth_met_y = np.sum(self.truth_particles.pt * np.sin(self.truth_particles.phi))

        self.pflow_ht = np.sum(self.pflow_particles.pt)
        self.pflow_met_x = np.sum(self.pflow_particles.pt * np.cos(self.pflow_particles.phi))
        self.pflow_met_y = np.sum(self.pflow_particles.pt * np.sin(self.pflow_particles.phi))

    @override
    def __repr__(self) -> str:
        return (
            f"Event {self.event_number} with {len(self.truth_particles)} truth_particles"
            f" and {len(self.pflow_particles)} pflow particles"
            f" {len(self.truth_jets) if self.truth_jets else 0} truth jets"
            f" {len(self.pflow_jets) if self.pflow_jets else 0} pflow jets"
        )
