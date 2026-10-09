"""Download the benchmark datasets and record a verifiable manifest.

Run this first.  It downloads each dataset, builds the canonical splits for
every seed in the unified specification, and writes ``results/MANIFEST.json``
containing a SHA-256 digest per (dataset, seed).

The manifest is the mechanism by which the four workstreams confirm they are
testing on the same bytes.  It is committed to the repository; the data itself
is not, because it is large, freely re-downloadable and derived deterministically
from this code.  A team member whose digests match the committed manifest has
provably identical data.  A mismatch means a different library version, a
different seed, or a corrupted download -- all of which silently invalidate a
cross-workstream comparison and none of which are otherwise visible.

Usage::

    python scripts/00_verify_data.py
    python scripts/00_verify_data.py --datasets mnist cifar10 --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shiftbench.datasets import DATASETS, load_split, split_digest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist", "cifar10"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--check",
        default=None,
        help="Path to an existing manifest to verify against instead of writing one.",
    )
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or os.path.join(results_dir, "MANIFEST.json")

    manifest = {"datasets": {}}
    for name in args.datasets:
        if name not in DATASETS:
            print(f"skipping unknown dataset {name!r}", file=sys.stderr)
            continue
        spec = DATASETS[name]
        entry = {
            "image_shape": list(spec.image_shape),
            "dim": spec.dim,
            "n_classes": spec.n_classes,
            "n_train": spec.n_train,
            "n_val": spec.n_val,
            "n_test": spec.n_test,
            "digests": {},
        }
        for seed in args.seeds:
            t0 = time.perf_counter()
            split = load_split(name, seed=seed)
            digest = split_digest(split)
            entry["digests"][str(seed)] = digest
            print(
                f"{name:14s} seed={seed}  d={split.dim:5d}  "
                f"val={len(split.x_val):6d} test={len(split.x_test):6d}  "
                f"{digest[:16]}...  ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
        manifest["datasets"][name] = entry

    if args.check:
        with open(args.check) as fh:
            reference = json.load(fh)
        bad = []
        for ds, entry in manifest["datasets"].items():
            ref = reference.get("datasets", {}).get(ds)
            if ref is None:
                bad.append(f"{ds}: absent from reference manifest")
                continue
            for seed, digest in entry["digests"].items():
                if ref["digests"].get(seed) != digest:
                    bad.append(f"{ds} seed={seed}: digest mismatch")
        if bad:
            print("\nFAIL - data does not match the reference manifest:")
            for line in bad:
                print(f"  {line}")
            return 1
        print("\nOK - every digest matches the reference manifest.")
        return 0

    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
