import pyhepmc

# DUT
from parnassus.pythia import HepMC3Generator

N_EVENTS = 1000
N_JOBS = 10


def main():  # Tests HepMC3Generator end-to-end: top-level configuration, parallelization, and merging of hepmc files.
    generator = HepMC3Generator(
        cmnd_file="pythia_cards/HZZ4l.cmnd",
        output_dir="data/HZZ4l",
        log_dir="logs/Parnassus_Pythia/HZZ4l",
    )
    fpath_merged = generator.generate(n_events=N_EVENTS, max_workers=N_JOBS, debug=False)
    print(f"Wrote to file {fpath_merged}")
    with pyhepmc.open(fpath_merged, "r") as f_merged:
        n_events_merged = sum(1 for _ in f_merged)
        assert n_events_merged == N_EVENTS, (
            f"HepMC3Generator: Merged file has {n_events_merged} events, expected {N_EVENTS}"
        )


if __name__ == "__main__":
    main()
