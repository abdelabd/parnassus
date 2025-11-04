import itertools
import math
from pathlib import Path

import awkward as ak

# FastJet (for clustering)
import fastjet as fj
import matplotlib.pyplot as plt
import numpy as np
import pyhepmc
from scipy import stats

# HepMC3 Python bindings
import pythia8mc
from joblib import Parallel, delayed
from tqdm import tqdm

from parnassus.pipelines.cluster import get_cluster_sequence

# DUT
from parnassus.pythia import HepMC3Generator, Pythia8ToHepMC3

# Helper functions #############################
MZ = 91.1876
GAMMA_Z = 2.4952

# How many samples to generate to compare with benchmark
N_JOBS = 10
N_EVENTS_PARALLEL = int(1e3) # must be in {1e4, 1e3, 1e2} by default
N_EVENTS_SINGLE = int(1e2)  # must be in {1e4, 1e3, 1e2} by default

# For test thresholds
MAX_KS_MAP = {int(1e4): 5e-2, int(1e3): 1.5e-1, int(1e2): 3e-1}
MIN_KS_P_MAP = {int(1e4): 5e-2, int(1e3): 5e-2, int(1e2): 1e-2}

PLOT_HISTS = True # Set to True to produce histograms during tests

def draw(
    event_dict_py: dict,
    event_dict_c: dict,
    key: str,
    fig_dir: str,
    bins: int = 60,
    range_: tuple | None = None,
    xlabel: str | None = None,
):
    x_py = event_dict_py.get(key, [])
    x_c = event_dict_c.get(key, [])
    print(f"{key}: {len(x_py)} Parnassus.Pythia entries, {len(x_c)} C++ entries")
    if len(x_py) == 0:
        print(f"[warn] no Parnassus.Pythia data for {key}")
        return
    if len(x_c) == 0:
        print(f"[warn] no C++ data for {key}")
        return
    plt.figure()
    plt.hist(x_c, bins=bins, range=range_, histtype="step", color="blue", label="C++ Pythia8", density=True)
    plt.hist(x_py, bins=bins, range=range_, histtype="step", color="orange", label="Parnassus.Pythia", density=True)
    plt.xlabel(xlabel or key)
    plt.ylabel("Events")
    plt.title(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / f"{key}.png")
    plt.close()

def full_draw(event_dict_py: dict, event_dict_c: dict, fig_dir: str) -> None: # Draws all histograms from event_dict_py into fig_dir.
    Path(fig_dir).mkdir(exist_ok=True, parents=True)

    # ---------- Plots ----------
    # Core Higgs/ZZ
    draw(event_dict_py, event_dict_c, "m_4l", fig_dir, bins=60, range_=(50, 200), xlabel="m(4ℓ) [GeV]")
    draw(event_dict_py, event_dict_c, "mZ1", fig_dir, bins=60, range_=(40, 120), xlabel="m(Z1) [GeV]")
    draw(event_dict_py, event_dict_c, "mZ2", fig_dir, bins=60, range_=(12, 120), xlabel="m(Z2) [GeV]")

    # Jets / VBF
    draw(event_dict_py, event_dict_c, "j1_pt", fig_dir, bins=60, range_=(0, 300), xlabel="pT(j1) [GeV]")
    draw(event_dict_py, event_dict_c, "j2_pt", fig_dir, bins=60, range_=(0, 200), xlabel="pT(j2) [GeV]")
    draw(event_dict_py, event_dict_c, "mjj", fig_dir, bins=60, range_=(0, 3000), xlabel="m(jj) [GeV]")
    draw(event_dict_py, event_dict_c, "detajj", fig_dir, bins=60, range_=(0, 10), xlabel="Δη(jj)")
    draw(event_dict_py, event_dict_c, "phijj", fig_dir, bins=60, range_=(-math.pi, math.pi), xlabel="Δφ(jj)")
    draw(event_dict_py, event_dict_c, "zeppenfeld_eta_star", fig_dir, bins=60, range_=(-5, 5), xlabel="η*(H)")

    # Leptons
    draw(event_dict_py, event_dict_c, "l1_pt", fig_dir, bins=60, range_=(0, 150), xlabel="pT(ℓ1) [GeV]")
    draw(event_dict_py, event_dict_c, "l2_pt", fig_dir, bins=60, range_=(0, 100), xlabel="pT(ℓ2) [GeV]")
    draw(event_dict_py, event_dict_c, "l3_pt", fig_dir, bins=60, range_=(0, 60), xlabel="pT(ℓ3) [GeV]")
    draw(event_dict_py, event_dict_c, "l4_pt", fig_dir, bins=60, range_=(0, 40), xlabel="pT(ℓ4) [GeV]")
    draw(event_dict_py, event_dict_c, "min_DR_ll", fig_dir, bins=60, range_=(0, 5), xlabel="min ΔR(ℓ,ℓ)")

# ---------- Kinematics ----------
def pt(px: float, py: float) -> float:
    return math.hypot(px, py)


def eta(px: float, py: float, pz: float) -> float:
    p = math.sqrt(px * px + py * py + pz * pz)
    if p == abs(pz):
        return float("inf") if pz >= 0 else -float("inf")
    return 0.5 * math.log((p + pz) / (p - pz))


def phi(px: float, py: float) -> float:
    return math.atan2(py, px)


def inv_mass(p4s: list[tuple[float, float, float, float]]) -> float:
    E = sum(p[3] for p in p4s)
    px = sum(p[0] for p in p4s)
    py = sum(p[1] for p in p4s)
    pz = sum(p[2] for p in p4s)
    m2 = E * E - (px * px + py * py + pz * pz)
    return math.sqrt(max(m2, 0.0))


def deltaR(
    a: tuple[float, float, float, float, float], b: tuple[float, float, float, float, float]
) -> float:
    # a,b are (px,py,pz,E,eta,phi)
    dphi = (a[5] - b[5] + math.pi) % (2 * math.pi) - math.pi
    return math.hypot(a[4] - b[4], dphi)


def as_p4(
    px: float, py: float, pz: float, E: float
) -> tuple[float, float, float, float, float, float]:
    return (px, py, pz, E, eta(px, py, pz), phi(px, py))


# ---------- Object ID ----------
NEUTRINOS = {12, 14, 16, -12, -14, -16}


def is_final(particle: pyhepmc.GenParticle) -> bool:
    # HepMC3 "status==1" is final state for Pythia
    return particle.status == 1


def is_lep(particle: pyhepmc.GenParticle) -> bool:
    return is_final(particle) and abs(particle.pid) in {11, 13}


def is_neutrino(particle: pyhepmc.GenParticle) -> bool:
    return is_final(particle) and particle.pid in NEUTRINOS


# ---------- Pairing (global chi^2 with BW scale) ----------
def best_ossf_pairs_min_chi2(
    leps_p4: list[tuple[float, float, float, float, float, float]], leps_id: list[int]
) -> tuple[tuple[int, int], tuple[int, int], float, float] | None:
    """leps_p4: list of 4-vectors (px,py,pz,E,eta,phi) for 4 leptons (selected & sorted)
    leps_id: matching PDG IDs (11,-11,13,-13)
    Returns: ((i,j),(k,l), m1, m2) with the pair closer to mZ listed first.
    """
    idx = [0, 1, 2, 3]
    best = None

    def mll(
        a: tuple[float, float, float, float, float, float],
        b: tuple[float, float, float, float, float, float],
    ) -> float:
        return inv_mass([
            (leps_p4[a][0], leps_p4[a][1], leps_p4[a][2], leps_p4[a][3]),
            (leps_p4[b][0], leps_p4[b][1], leps_p4[b][2], leps_p4[b][3]),
        ])

    for i in idx:
        for j in idx:
            if j <= i:
                continue
            if leps_id[i] != -leps_id[j]:  # OSSF
                continue
            rem = [k for k in idx if k not in {i, j}]
            if leps_id[rem[0]] != -leps_id[rem[1]]:
                continue
            mA, mB = mll(i, j), mll(rem[0], rem[1])
            # chi^2 with BW width as scale
            chi2 = ((mA - MZ) ** 2 + (mB - MZ) ** 2) / (GAMMA_Z**2)
            if abs(mB - MZ) < abs(mA - MZ):
                cand = (chi2, (rem[0], rem[1]), (i, j), mB, mA)
            else:
                cand = (chi2, (i, j), (rem[0], rem[1]), mA, mB)
            if best is None or cand[0] < best[0]:
                best = cand
    if best is None:
        return None
    _, p1, p2, m1, m2 = best
    return p1, p2, m1, m2


# ---------- Jet clustering ----------
def build_jet_inputs(
    final_particles: list[pyhepmc.GenParticle],
    exclude_p4: list[tuple[float, float, float, float, float, float]],
    R: float = 0.4,
    ptmin: float = 30.0,
    etamax: float = 4.5,
) -> list[tuple[float, float, float, float, float, float]]:
    """final_particles: iterable of HepMC particles (status==1)
    exclude_p4: list of lepton p4 to be excluded from clustering (ΔR>0.4)
    Returns list of jets, each as (px,py,pz,E,eta,phi)
    """
    # Keep visible particles: drop neutrinos; keep photons/hadrons
    cand = []
    for p in final_particles:
        if not is_final(p):
            continue
        if is_neutrino(p):  # drop nu, nubar
            continue
        p4 = as_p4(p.momentum.px, p.momentum.py, p.momentum.pz, p.momentum.e)
        # Overlap removal: don't feed the selected leptons to the jet finder
        if any(deltaR(p4, lp4) < R for lp4 in exclude_p4):
            continue
        # Feed all remaining to clustering; pT/eta cuts applied after clustering
        cand.append((p.momentum.px, p.momentum.py, p.momentum.pz, p.momentum.e))

    if not cand:
        return []

    a = np.array(cand)
    jetdef = fj.JetDefinition(fj.antikt_algorithm, R)
    four_vectors = ak.Array(
        {"px": a[..., 0], "py": a[..., 1], "pz": a[..., 2], "E": a[..., 3]},
        with_name="Momentum4D",
    )
    sequence = get_cluster_sequence(jetdef, four_vectors)
    jets = sequence.inclusive_jets()  # list of PseudoJets

    out = []
    for j in jets:
        px, py, pz, E = j.px(), j.py(), j.pz(), j.e()
        p4 = as_p4(px, py, pz, E)
        if pt(px, py) < ptmin:
            continue
        if abs(p4[4]) > etamax:
            continue
        out.append(p4)
    # sort by pT desc
    out.sort(key=lambda v: pt(v[0], v[1]), reverse=True)
    return out


def inspect_hepmc_HZZ4l(fpath_hepmc: str, args: dict) -> None: # Inspects HepMC3 file with H->ZZ->4l VBF-like events; produces histograms and summary statistics.
    plot_vars = [
        "m_4l",
        "mZ1",
        "mZ2",
        "j1_pt",
        "j2_pt",
        "j1_eta",
        "j2_eta",
        "mjj",
        "detajj",
        "phijj",
        "zeppenfeld_eta_star",
        "l1_pt",
        "l2_pt",
        "l3_pt",
        "l4_pt",
        "l1_eta",
        "l2_eta",
        "l3_eta",
        "l4_eta",
        "min_DR_ll",
    ]
    evt_dict = {k: np.full((args["max_events"],), np.nan) for k in plot_vars}

    cut_vars = ["fail_nlep>=4", "fail_njets>=2", "fail_pairing", "fail_mZ2_window"]
    counts = {k: np.zeros((args["max_events"],)) for k in cut_vars}
    n_events = np.zeros((args["max_events"],))

    def process_event(event_idx: int, event: pyhepmc.GenEvent) -> None:
        n_events[event_idx] = 1
        final_parts = [p for p in event.particles if is_final(p)]

        # --- leptons (e, mu), baseline selection ---
        leptons = []
        for p in final_parts:
            if not is_lep(p):
                continue
            px, py, pz, E = p.momentum.px, p.momentum.py, p.momentum.pz, p.momentum.e
            p4 = as_p4(px, py, pz, E)
            if pt(px, py) < args["lep_pt"]:
                continue
            if abs(p4[4]) > args["lep_eta"]:
                continue
            leptons.append((p.pid, p4))

        if len(leptons) < 4:
            counts["fail_nlep>=4"][event_idx] = 1
            return

        # sort leptons by pT
        leptons.sort(key=lambda t: pt(t[1][0], t[1][1]), reverse=True)
        # take 4 leading (analysis usually requires exactly 4 isolated; here we keep the top 4)
        leptons = leptons[:4]
        lep_ids = [x[0] for x in leptons]
        lep_p4s = [x[1] for x in leptons]

        # Pairing
        pair = best_ossf_pairs_min_chi2(lep_p4s, lep_ids)
        if pair is None:
            counts["fail_pairing"][event_idx] = 1
            return
        _, _, mZ1, mZ2 = pair
        # Off-shell threshold (optional, classic 12 GeV)
        if mZ2 < args["min_mll2"]:
            counts["fail_mZ2_window"][event_idx] = 1
            return

        # Higgs 4l
        p4_4l = tuple(np.sum(np.array([(p[0], p[1], p[2], p[3]) for p in lep_p4s]), axis=0))
        m_4l = inv_mass([(p[0], p[1], p[2], p[3]) for p in lep_p4s])

        # --- jets via anti-kt on visible final state, excluding the leptons ---
        jets = build_jet_inputs(
            final_parts,
            exclude_p4=lep_p4s,
            R=args["R"],
            ptmin=args["jet_pt"],
            etamax=args["jet_eta"],
        )
        if len(jets) < 2:
            counts["fail_njets>=2"][event_idx] = 1
            # still fill purely leptonic histos
            evt_dict["m_4l"][event_idx] = m_4l
            evt_dict["mZ1"][event_idx] = mZ1
            evt_dict["mZ2"][event_idx] = mZ2
            for idx, p4 in enumerate(lep_p4s):
                evt_dict[f"l{idx + 1}_pt"][event_idx] = pt(p4[0], p4[1])
                evt_dict[f"l{idx + 1}_eta"][event_idx] = p4[4]
            evt_dict["min_DR_ll"][event_idx] = min(
                deltaR(lep_p4s[a], lep_p4s[b]) for a in range(4) for b in range(a + 1, 4)
            )
            return

        # Leading jets and VBF variables
        j1, j2 = jets[0], jets[1]
        mjj = inv_mass([(j1[0], j1[1], j1[2], j1[3]), (j2[0], j2[1], j2[2], j2[3])])
        detajj = abs(j1[4] - j2[4])
        dphijj = (j1[5] - j2[5] + math.pi) % (2 * math.pi) - math.pi
        etaH = eta(p4_4l[0], p4_4l[1], p4_4l[2])  # could use rapidity; eta is fine here
        zepp = etaH - 0.5 * (j1[4] + j2[4])

        # Fill histos
        evt_dict["m_4l"][event_idx] = m_4l
        evt_dict["mZ1"][event_idx] = mZ1
        evt_dict["mZ2"][event_idx] = mZ2
        evt_dict["j1_pt"][event_idx] = pt(j1[0], j1[1])
        evt_dict["j2_pt"][event_idx] = pt(j2[0], j2[1])
        evt_dict["j1_eta"][event_idx] = j1[4]
        evt_dict["j2_eta"][event_idx] = j2[4]
        evt_dict["mjj"][event_idx] = mjj
        evt_dict["detajj"][event_idx] = detajj
        evt_dict["phijj"][event_idx] = dphijj
        evt_dict["zeppenfeld_eta_star"][event_idx] = zepp
        for idx, p4 in enumerate(lep_p4s):
            evt_dict[f"l{idx + 1}_pt"][event_idx] = pt(p4[0], p4[1])
            evt_dict[f"l{idx + 1}_eta"][event_idx] = p4[4]
        evt_dict["min_DR_ll"][event_idx] = min(
            deltaR(lep_p4s[a], lep_p4s[b]) for a in range(4) for b in range(a + 1, 4)
        )

    # Reader handles plain or gz
    with pyhepmc.open(fpath_hepmc, "r") as f:
        if args["debug"]:
            for i, evt in enumerate(f):
                if i < 1000 // args["n_jobs"]:
                    process_event(i, evt)
        else:
            Parallel(n_jobs=args["n_jobs"], prefer="threads")(
                itertools.starmap(
                    delayed(process_event), tqdm(enumerate(f), total=args["max_events"])
                )
            )

    # Aggregate results
    for k in evt_dict:
        evt_dict[k] = evt_dict[k][~np.isnan(evt_dict[k])]  # trim to filled
    for k, v in counts.items():
        counts[k] = int(np.sum(v))
    n_events = int(np.sum(n_events))

    # ---------- Summary ----------
    print(f"Events total: {n_events}")
    n_skipped = 0
    for k, v in counts.items():
        print(f"{k}: {v}")
        n_skipped += v
    print(f"\n\nEvents selected: {n_events - n_skipped}\n\n")

    return evt_dict, counts, n_events

# Testing function #################################
def test_hepmc3_generator(): # Tests HepMC3Generator end-to-end: top-level configuration, parallelization, and merging of hepmc files.
    # Tests HepMC3Generator.__init__()
        # ._is_hadronization_on()
    # Tests HepMC3Generator.generate()
        # ._write_single_job()
        # ._gen_hepmc_single_job()
        # ._merge_hepmc_files()
            # ._append_hepmc_file()
    # So that is all the methods of HepMC3Generator covered

    fpath_benchmark = f"src/parnassus/tests/benchmark_data/HZZ4l/{N_EVENTS_PARALLEL}/events.hepmc"
    assert Path(fpath_benchmark).exists(), f"""Benchmark file {fpath_benchmark} does not exist. N_EVENTS_PARALLEL must be a number for which benchmark data exists. 
                                                    \nCheck src/parnassus/tests/benchmark_data/HZZ4l. You can generate your own benchmark data via tests/HZZ4l.cc OR download it from the Google Drive link: https://drive.google.com/drive/folders/1W-V_rU6lRmtuaOclj3gYB1qJSn4J11qM?usp=sharing"""
    args = {
        "R": 0.4,
        "jet_pt": 30.0,
        "jet_eta": 4.5,
        "lep_pt": 10.0,
        "lep_eta": 2.5,
        "min_mll2": 12.0,
        "max_events": N_EVENTS_PARALLEL,
        "n_jobs": N_JOBS,
        "debug": False,
    }
    generator = HepMC3Generator(
        cmnd_file="src/parnassus/tests/HZZ4l.cmnd",
        output_dir=f"src/parnassus/tests/data_out/HZZ4l/{N_EVENTS_PARALLEL}/HepMC3Generator",
        log_dir=f"src/parnassus/tests/logs/HZZ4l/{N_EVENTS_PARALLEL}/HepMC3Generator",
    )
    fpath_merged = generator.generate(
        n_events=args["max_events"], max_workers=args["n_jobs"], debug=args["debug"]
    )
    print(f"Wrote to file {fpath_merged}")

    print("Inspecting hepmc file and generating histograms")
    evt_dict_out, counts_out, n_events_out = inspect_hepmc_HZZ4l(fpath_merged, args)
    evt_dict_bench, counts_bench, n_events_bench = inspect_hepmc_HZZ4l(fpath_benchmark, args)
    if PLOT_HISTS:
        full_draw(evt_dict_out, evt_dict_bench, fig_dir=f"src/parnassus/tests/figures/HZZ4l/{N_EVENTS_PARALLEL}/HepMC3Generator")

    # Check number of events generated
    print(f"HepMC3Generator: n_events_out: {n_events_out}, n_events_bench: {n_events_bench}")
    assert(n_events_out == n_events_bench), "Mismatch in total event count > 1e-5 relative"

    # Compare histograms with bin-independent KS test
    # Kolmogorov-Smirnov statistic = maximum absolute distance between CDFs
    # small KS means similar distributions
    # Large p-value means high probability that samples are drawn from same distribution (i.e. the null hypothesis is not rejected)
    # So we want small KS and large p-value
    print("HepMC3Generator: Comparing histograms...")
    for (k_out, v_out), (k_bench, v_bench) in zip(evt_dict_out.items(), evt_dict_bench.items()):
        ks_diff = stats.ks_2samp(v_out, v_bench)
        print(f"\n{k_out}: KS statistic = {ks_diff.statistic}, p-value = {ks_diff.pvalue}")
        assert(ks_diff.statistic < MAX_KS_MAP[N_EVENTS_PARALLEL]), f"KS statistic too large for {k_out}"
        assert(ks_diff.pvalue > MIN_KS_P_MAP[N_EVENTS_PARALLEL]), f"KS p-value too small for {k_out}"

    # Testing that KS is indeed a useful metric by comparing two different variables
    ks_test = stats.ks_2samp(evt_dict_out["m_4l"], evt_dict_bench["mZ1"])
    print(f"\nCross-variable KS test (m_4l vs mZ1): KS statistic = {ks_test.statistic}, p-value = {ks_test.pvalue}")
    assert(ks_test.statistic > 0.5), "Cross-variable KS statistic too small (should be very different distributions)"
    assert(ks_test.pvalue < 1e-5), "Cross-variable KS p-value too large (should be very different distributions)"




def test_pythia8_to_hepmc3(): # Tests standalone Pythia8ToHepMC3 end-to-end: validates data output against benchmark hepmc files generated with C++ Pythia8 package; see HZZ4l.cc and HZZ4l.cmnd
    # Tests Pythia8ToHepMC3.__init__()
    # Tests Pythia8ToHepMC3.fill_next_event()
        # ._get_particles()
        # ._get_vertices()
        # ._add_tree()
            # ._topological_sort_vertices()
        # ._check_if_free_particle()
        # ._store_event_info()
    # This leaves Pythia8ToHepMC3._add_color(), which is currently broken due to HepMC3 Python bindings limitations

    fpath_benchmark = f"src/parnassus/tests/benchmark_data/HZZ4l/{N_EVENTS_SINGLE}/events.hepmc"
    assert Path(fpath_benchmark).exists(), f"""Benchmark file {fpath_benchmark} does not exist. N_EVENTS_PARALLEL must be a number for which benchmark data exists. 
                                                    \nCheck src/parnassus/tests/benchmark_data/HZZ4l. You can generate your own benchmark data via tests/HZZ4l.cc, or download it from the Google Drive link: https://drive.google.com/drive/folders/1W-V_rU6lRmtuaOclj3gYB1qJSn4J11qM?usp=sharing"""
    
    ############# Generate events with our interface #############
    pythia = pythia8mc.Pythia()

    # Random seed
    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:seed = 42")

    # Command file for rest of config
    pythia.readFile("src/parnassus/tests/HZZ4l.cmnd")

    if not pythia.init():
        print("test_pythia::test_pythia8_to_hepmc3: Pythia initialization failed!")
        return

    converter = Pythia8ToHepMC3(
        m_hadronization_on=True
    )  # This enables detect_cycles and visit_children
    Path(f"src/parnassus/tests/data_out/HZZ4l/{N_EVENTS_SINGLE}/Pythia8ToHepMC3").mkdir(
        exist_ok=True, parents=True
    )
    fpath_out = f"src/parnassus/tests/data_out/HZZ4l/{N_EVENTS_SINGLE}/Pythia8ToHepMC3/events.hepmc"

    writer = pyhepmc.io.WriterAscii(fpath_out)
    n_written = 0
    idx_event = 0
    while n_written < N_EVENTS_SINGLE:
        if not pythia.next():
            continue  # event failed, try again

        hepmcEvent = converter.fill_next_event(pythia, idx_event + 1)
        writer.write_event(hepmcEvent)
        n_written += 1
        idx_event += 1

        if n_written % 100 == 0:
            print(f"Pythia8ToHepMC3: Generated {n_written} events...")

    pythia.stat()
    writer.close()

    ############# Compare against C++ benchmark #############
    args = {
        "R": 0.4,
        "jet_pt": 30.0,
        "jet_eta": 4.5,
        "lep_pt": 10.0,
        "lep_eta": 2.5,
        "min_mll2": 12.0,
        "max_events": N_EVENTS_SINGLE,
        "n_jobs": N_JOBS,
        "debug": False,
    }
    evt_dict_out, counts_out, n_events_out = inspect_hepmc_HZZ4l(fpath_out, args)

    evt_dict_bench, counts_bench, n_events_bench = inspect_hepmc_HZZ4l(fpath_benchmark, args)
    if PLOT_HISTS:
        full_draw(evt_dict_out, evt_dict_bench, fig_dir=f"src/parnassus/tests/figures/HZZ4l/{N_EVENTS_SINGLE}/Pythia8ToHepMC3")

    # Check number of events generated
    print(f"Pythia8ToHepMC3: n_events_out: {n_events_out}, n_events_bench: {n_events_bench}")
    assert(n_events_out == n_events_bench), "Mismatch in total event count"

    # Compare histograms with bin-independent KS test
    # Kolmogorov-Smirnov statistic = maximum absolute distance between CDFs
    # small KS means similar distributions
    # Large p-value means high probability that samples are drawn from same distribution (i.e. the null hypothesis is not rejected)
    # So we want small KS and large p-value
    print("Pythia8ToHepMC3: Comparing histograms...")
    for (k_out, v_out), (k_bench, v_bench) in zip(evt_dict_out.items(), evt_dict_bench.items()):
        ks_diff = stats.ks_2samp(v_out, v_bench)
        print(f"\n{k_out}: KS statistic = {ks_diff.statistic}, p-value = {ks_diff.pvalue}")
        assert(ks_diff.statistic < MAX_KS_MAP[N_EVENTS_SINGLE]), f"KS statistic too large for {k_out}"
        assert(ks_diff.pvalue > MIN_KS_P_MAP[N_EVENTS_SINGLE]), f"KS p-value too small for {k_out}"

    # Testing that KS is indeed a useful metric by comparing two different variables
    ks_test = stats.ks_2samp(evt_dict_out["m_4l"], evt_dict_bench["mZ1"])
    print(f"\nCross-variable KS test (m_4l vs mZ1): KS statistic = {ks_test.statistic}, p-value = {ks_test.pvalue}")
    assert(ks_test.statistic > 0.5), "Cross-variable KS statistic too small (should be very different distributions)"
    assert(ks_test.pvalue < 1e-5), "Cross-variable KS p-value too large (should be very different distributions)"

    