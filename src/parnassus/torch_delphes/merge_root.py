"""Merge arbitrary ROOT files sharing a TTree schema into one.

A general-purpose companion to ``slurm_scripts/merge_pseudodata.sbatch``, which
merges only the fixed ``cms_pseudodata_seed*.root`` glob inside one
``PARTS_DIR``. This module takes an explicit list of files instead, so it can
combine whatever you point it at -- leftover parts from a failed array, or the
per-species gun samples (muon / electron / photon / pion) into one mixed
training sample.

Concatenation is safe for this schema: the files carry no persisted event-number
branch (event identity is the TTree entry index), so appending entries yields
globally unique indices with no duplication. The merge is **streaming** -- one
input is held in memory at a time, not the whole output -- and the merged entry
count is verified against the sum of the inputs before anything is deleted.

Run with
``python -m parnassus.torch_delphes.merge_root out.root in1.root in2.root ...``
(shell globs work: ``... merged.root parts/*.root``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import awkward as ak
import uproot


def _resolve(paths: list[Path], tree: str) -> list[tuple[Path, int, tuple[str, ...]]]:
    """Open each input once: existence, tree presence, entry count, field set."""
    out: list[tuple[Path, int, tuple[str, ...]]] = []
    for p in paths:
        if not p.exists():
            raise SystemExit(f"input does not exist: {p}")
        with uproot.open(p) as f:
            if tree not in f:
                available = [k.split(";")[0] for k in f.keys()]
                raise SystemExit(
                    f"{p}: no TTree named {tree!r} (found: {sorted(set(available))}). "
                    "Pass --tree to select a different one."
                )
            t = f[tree]
            out.append((p, t.num_entries, tuple(sorted(t.keys()))))
    return out


def merge(
    inputs: list[Path],
    output: Path,
    tree: str = "event_tree",
    delete_inputs: bool = False,
    quiet: bool = False,
) -> int:
    """Concatenate ``inputs`` into ``output``; return the merged entry count.

    Every input must expose the same branch set under ``tree`` -- a partial
    merge would silently produce a file whose branches are defined for only some
    entries, which is far worse than refusing. The usual cause of a mismatch is
    mixing samples generated with and without ``generate_pseudodata --debug``
    (8 branches vs ~210).
    """
    def log(msg: str) -> None:
        if not quiet:
            print(msg)

    if not inputs:
        raise SystemExit("no input files given.")

    resolved = _resolve(inputs, tree)

    # Refuse to write over an input: uproot.recreate would truncate it before we
    # ever read it, destroying data with no way back.
    out_real = output.resolve()
    for p, _n, _f in resolved:
        if p.resolve() == out_real:
            raise SystemExit(f"--output {output} is also an input; refusing to overwrite it.")

    reference = resolved[0][2]
    for p, _n, fields in resolved[1:]:
        if fields != reference:
            only_ref = sorted(set(reference) - set(fields))[:5]
            only_p = sorted(set(fields) - set(reference))[:5]
            raise SystemExit(
                "branch mismatch between inputs:\n"
                f"  {resolved[0][0].name}: {len(reference)} branches\n"
                f"  {p.name}: {len(fields)} branches\n"
                f"  only in {resolved[0][0].name}: {only_ref}\n"
                f"  only in {p.name}: {only_p}\n"
                "All inputs must share one schema (a common cause is mixing files "
                "generated with and without `generate_pseudodata --debug`)."
            )

    src_total = sum(n for _p, n, _f in resolved)
    log(
        f"[merge] {len(resolved)} file(s), {len(reference)} branches, "
        f"{src_total} source events -> {output}"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with uproot.recreate(output) as fout:
        for i, (p, _n, _f) in enumerate(resolved):
            with uproot.open(p) as fin:
                rec = fin[tree].arrays(library="ak")
            payload = {field: rec[field] for field in rec.fields}
            if i == 0:
                fout[tree] = payload
            else:
                fout[tree].extend(payload)
            written += len(rec)
            log(f"[merge]   {p.name}: +{len(rec)} (running total {written})")
            del rec, payload

    with uproot.open(output) as f:
        merged_n = f[tree].num_entries
    if merged_n != src_total:
        raise SystemExit(
            f"[merge] ERROR: merged {merged_n} != source total {src_total}; "
            "inputs left untouched for inspection."
        )
    log(f"[merge] OK: {merged_n} events == sum of inputs")

    if delete_inputs:
        for p, _n, _f in resolved:
            os.remove(p)
        log(f"[merge] deleted {len(resolved)} input file(s)")
    return merged_n


def main() -> None:
    """Entry point for ``python -m parnassus.torch_delphes.merge_root``."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output", type=Path, help="Merged ROOT file to write.")
    parser.add_argument("inputs", type=Path, nargs="+", help="Input ROOT files (globs work).")
    parser.add_argument(
        "--tree", type=str, default="event_tree", help="TTree name (default: event_tree)."
    )
    parser.add_argument(
        "--delete-inputs",
        action="store_true",
        help="Delete the inputs after the entry-count check passes. Off by default; "
        "the merge is verified first, so a failed merge never deletes anything.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only report errors.")
    args = parser.parse_args()
    merge(args.inputs, args.output, args.tree, args.delete_inputs, args.quiet)


if __name__ == "__main__":
    main()
