#!/usr/bin/env python3
"""Create a deterministic random subset of an existing WhestBench dataset.

This does not generate MLPs or recompute Monte Carlo targets. It loads an
existing dataset split, selects rows by seeded random index sampling, and writes
those existing rows to a local whestbench dataset directory.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from whestbench.dataset import load_dataset, metadata
from whestbench.dataset_io import METADATA_FILE, write_dataset_dir


def _parse_dataset_arg(value: str) -> tuple[str, str | None]:
    if not value.startswith("hf://"):
        return value, None
    body = value[len("hf://") :]
    if "@" not in body:
        return body, None
    repo, revision = body.rsplit("@", 1)
    return repo, revision


def _default_output(dataset: str, split: str, n_mlps: int, seed: int) -> Path:
    safe_dataset = dataset.removeprefix("hf://").replace("/", "__").replace("@", "__")
    return Path(".cache/whestbench") / f"{safe_dataset}-{split}-random-n{n_mlps}-seed{seed}"


def _selection_metadata(
    source_metadata: dict[str, Any],
    *,
    dataset: str,
    split: str,
    seed: int,
    indices: list[int],
) -> dict[str, Any]:
    out = dict(source_metadata)
    out["n_mlps"] = len(indices)
    out["split"] = split
    out.pop("prepared_splits", None)
    out["row_selection"] = {
        "method": "random.sample",
        "source_dataset": dataset,
        "source_split": split,
        "seed": seed,
        "indices": indices,
    }
    return out


def _existing_selection_matches(path: Path, expected: dict[str, Any]) -> bool:
    metadata_path = path / METADATA_FILE
    if not metadata_path.is_file():
        return False
    try:
        actual = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False
    return actual.get("row_selection") == expected.get("row_selection")


def _existing_selection_matches_request(
    path: Path, *, dataset: str, split: str, n_mlps: int, seed: int
) -> bool:
    metadata_path = path / METADATA_FILE
    if not metadata_path.is_file():
        return False
    try:
        actual = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False
    selection = actual.get("row_selection")
    return (
        isinstance(selection, dict)
        and selection.get("source_dataset") == dataset
        and selection.get("source_split") == split
        and selection.get("seed") == seed
        and isinstance(selection.get("indices"), list)
        and len(selection["indices"]) == n_mlps
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hf://aicrowd/arc-whestbench-public-2026")
    parser.add_argument("--split", default="mini")
    parser.add_argument("--n-mlps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.n_mlps <= 0:
        raise SystemExit("--n-mlps must be positive")

    output = args.output or _default_output(args.dataset, args.split, args.n_mlps, args.seed)
    if output.exists() and not args.force:
        if _existing_selection_matches_request(
            output,
            dataset=args.dataset,
            split=args.split,
            n_mlps=args.n_mlps,
            seed=args.seed,
        ):
            print(output)
            return
        raise SystemExit(f"{output} already exists with a different selection; pass --force")

    repo_or_path, revision = _parse_dataset_arg(args.dataset)
    ds = load_dataset(repo_or_path, revision=revision, split=args.split)
    if args.n_mlps > len(ds):
        raise SystemExit(f"--n-mlps={args.n_mlps} exceeds split size {len(ds)}")

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(ds)), args.n_mlps))
    selected = ds.select(indices)
    selected_metadata = _selection_metadata(
        metadata(ds),
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        indices=indices,
    )

    if output.exists():
        if _existing_selection_matches(output, selected_metadata):
            print(output)
            return
        if not args.force:
            raise SystemExit(f"{output} already exists with a different selection; pass --force")
        shutil.rmtree(output)

    write_dataset_dir(selected, output_dir=output, split=args.split, metadata=selected_metadata)
    print(output)


if __name__ == "__main__":
    main()
