#!/usr/bin/env python3
"""Split a local WhestBench dataset into one-MLP dataset directories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/i/e/.cache/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE", "/i/e/.cache/huggingface/datasets")

from whestbench.dataset import load_dataset, metadata
from whestbench.dataset_io import METADATA_FILE, write_dataset_dir


def _selection_metadata(
    source_metadata: dict[str, Any],
    *,
    split: str,
    index: int,
) -> dict[str, Any]:
    out = dict(source_metadata)
    out["n_mlps"] = 1
    out["split"] = split
    out.pop("prepared_splits", None)
    out["row_selection"] = {
        "method": "single-row-split",
        "source_dataset": "local",
        "source_split": split,
        "index": index,
    }
    return out


def _existing_matches(path: Path, index: int) -> bool:
    metadata_path = path / METADATA_FILE
    if not metadata_path.is_file():
        return False
    try:
        actual = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False
    selection = actual.get("row_selection")
    return isinstance(selection, dict) and selection.get("index") == index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-mlps", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        raise SystemExit(f"--dataset must be a local directory: {args.dataset}")
    if args.n_mlps is not None and args.n_mlps <= 0:
        raise SystemExit("--n-mlps must be positive")

    ds = load_dataset(str(args.dataset), split=args.split)
    n_mlps = min(args.n_mlps or len(ds), len(ds))
    source_metadata = metadata(ds)
    args.output_root.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for index in range(n_mlps):
        output = args.output_root / f"mlp-{index:06d}"
        if output.exists() and _existing_matches(output, index) and not args.force:
            paths.append(output)
            continue
        if output.exists():
            if not args.force:
                raise SystemExit(f"{output} already exists with different metadata; pass --force")
            shutil.rmtree(output)
        write_dataset_dir(
            ds.select([index]),
            output_dir=output,
            split=args.split,
            metadata=_selection_metadata(source_metadata, split=args.split, index=index),
        )
        paths.append(output)

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
