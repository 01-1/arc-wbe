"""Load the Fly truth bank for offline research gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

BANK_DIR = Path(__file__).resolve().parent


def load_bank(bank_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ``(seeds, truths, metadata)`` from a truth-bank directory."""

    root = Path(bank_dir) if bank_dir is not None else BANK_DIR
    arrays = np.load(root / "truth_bank.npz")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return arrays["seeds"], arrays["truths"], metadata
