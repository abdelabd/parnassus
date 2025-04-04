import multiprocessing as mp
import os
import sys
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from io import TextIOWrapper
from typing import Any, final

import awkward as ak
import energyflow as ef
import fastjet as fj
import numpy as np
from typing_extensions import override

from parnassus.configs.accessors import Accessor, JetAccessor
from parnassus.configs.pipeline import JetClusteringConfig
from parnassus.configs.scheme import (
    GenEvent,
    GenJetCollection,
    GenParticleCollection,
)
from parnassus.utils.logger import ProgressBar

from .base import GenPipeline


@final
class Jet:
    def __init__(self, fj_jet: fj.PseudoJet, R: float, calc_substructure: bool = False):
        self.fj_jet = fj_jet
        self.R = R
        self.nconstituents = len(self.constituents())
        self.constituents_pt = np.array([c.pt() for c in self.constituents()])
        self.constituents_eta = np.array([c.eta() for c in self.constituents()])
        self.constituents_phi = np.array([
            c.phi() if c.phi() <= np.pi else c.phi() - 2 * np.pi for c in self.constituents()
        ])
        self.constituents_m = np.array([c.m() for c in self.constituents()])
        self.constituents_idx = np.array([c.user_index() for c in self.constituents()])
        self.pt_order_constituents()

        self.dR_matrix = None
        self.ecf = {0: 1, 1: -1, 2: -1, 3: -1}
        self.substructure = {"c2": np.nan, "d2": np.nan}

        if calc_substructure:
            self.calc_substructure()

    def __getattr__(self, name: str):
        if name in {
            "pt",
            "eta",
            "phi",
            "phi_std",
            "e",
            "m",
            "constituents",
            "px",
            "py",
            "pz",
            "E",
        }:
            return getattr(self.fj_jet, name)
        if name in self.__dict__:
            return getattr(self, name)
        raise AttributeError(f"'Jet' object has no attribute '{name}'")

    def pt_order_constituents(self):
        idx = np.argsort(self.constituents_pt)[::-1]
        self.constituents_pt = self.constituents_pt[idx]
        self.constituents_eta = self.constituents_eta[idx]
        self.constituents_phi = self.constituents_phi[idx]
        self.constituents_m = self.constituents_m[idx]
        self.constituents_idx = self.constituents_idx[idx]

    def calc_substructure(self):
        d2_calc = ef.D2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)
        c2_calc = ef.C2(measure="hadr", beta=1, coords="ptyphim", reg=1e-31)

        pt_eta_phi_m = np.stack(
            [
                self.constituents_pt,
                self.constituents_eta,
                self.constituents_phi,
                self.constituents_m,
            ],
            axis=1,
        )

        self.substructure["d2"] = d2_calc.compute(pt_eta_phi_m)
        self.substructure["c2"] = c2_calc.compute(pt_eta_phi_m)


def get_cluster_sequence(
    jet_definition: fj.JetDefinition, four_vectors: ak.Array, user_indices: list[int] | None = None
) -> fj.ClusterSequence:
    pj_array: list[fj.PseudoJet] = []

    for i, part in enumerate(four_vectors):
        pj = fj.PseudoJet(part.px.item(), part.py.item(), part.pz.item(), part.E.item())
        if user_indices is not None:
            pj.set_user_index(user_indices[i])
        else:
            pj.set_user_index(i)
        pj_array.append(pj)

    return fj.ClusterSequence(pj_array, jet_definition)


def cluster_jets(particles: GenParticleCollection, config: JetClusteringConfig) -> list[Jet]:
    ak_4vecs = particles.get4vecs_awkward()
    cs = get_cluster_sequence(
        config.jet_definition, ak_4vecs, user_indices=list(range(len(particles)))
    )
    jets = cs.inclusive_jets(config.min_pt)
    jets = fj.sorted_by_pt(jets)
    jets = [Jet(j, 0.5, calc_substructure=True) for j in jets]
    return [j for j in jets if j.nconstituents >= config.nconst_min]


def convert_to_jet_collection(name: str, jets: list[Jet]) -> GenJetCollection:
    return GenJetCollection(
        name=name,
        pt=np.array([jet.pt() for jet in jets]),
        eta=np.array([jet.eta() for jet in jets]),
        phi=np.array([jet.phi() for jet in jets]),
        d2=np.array([jet.substructure["d2"] for jet in jets]),
        c2=np.array([jet.substructure["c2"] for jet in jets]),
        particle_idx=[jet.constituents_idx for jet in jets],
    )


def process_events(
    event_list: list[GenEvent], config: JetClusteringConfig
) -> list[GenJetCollection]:
    jets: list[GenJetCollection] = []
    for event in event_list:
        if config.collection == "truth":
            evt_jets = cluster_jets(event.truth_particles, config)
        elif config.collection == "pflow":
            evt_jets = cluster_jets(event.pflow_particles, config)
        else:
            evt_jets = []
        jets.append(convert_to_jet_collection(config.name, evt_jets))
    return jets


@contextmanager
def stdout_redirected(to: str = os.devnull):
    """Import os.

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    # assert that Python and C stdio write using the same file descriptor
    # assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to: TextIOWrapper):
        _ = sys.stdout.close()  # + implicit flush()
        _ = os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def process_events_wrapper(args: Iterable[Any]):
    with stdout_redirected():
        return process_events(*args)


@final
class JetClusteringPipeline(GenPipeline):
    @override
    def __init__(self, config: JetClusteringConfig):
        self.config = config

    @override
    def get_accessors(self) -> dict[str, list[Accessor]]:
        return {
            self.config.name: [
                JetAccessor("pt", self.config.name),
                JetAccessor("eta", self.config.name),
                JetAccessor("phi", self.config.name),
                JetAccessor("d2", self.config.name),
                JetAccessor("c2", self.config.name),
            ]
        }

    @override
    def process(self, events: Sequence[GenEvent]):
        n_events = len(events)
        batch_size = 2000
        n_batches = n_events // batch_size
        n_batches += 1 if n_events % batch_size != 0 else 0

        input_batched_data = [
            (events[i * batch_size : (i + 1) * batch_size], self.config) for i in range(n_batches)
        ]
        n_events_in_batch = (len(data[0]) for data in input_batched_data)
        jets: list[GenJetCollection] = []
        with mp.Pool(processes=self.config.num_processes) as pool, ProgressBar() as progress:
            task = progress.add_task(f"[green]Cluster {self.config.name} jets", total=n_events)
            for jets_ in pool.imap(process_events_wrapper, input_batched_data):
                jets.extend(jets_)
                progress.update(task, advance=next(n_events_in_batch))

        for i in range(n_events):
            events[i].jets[self.config.name] = jets[i]
