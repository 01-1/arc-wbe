#!/usr/bin/env python3
"""Regenerate `history/` from ESTIMATOR_HISTORY.md plus the recovered pre-refactor file.

`ESTIMATOR_HISTORY.md` is canonical: agents append to it (see AGENTS.md). The
`history/` tree published in the public repository is derived, and this script
rebuilds it, so the two cannot drift. Run from the repository root.

The 2026-06-28 commit 88ea3f0 ("Move and prune estimator history") cut
docs/how-to/estimator-history.md from 690 lines to 121. Those 681 lines are
recovered here from `88ea3f0^` and split by shape: warmup-round sections
(256x8, 6.8e10) go to 00-warmup-round.md, while the three already on the
phase-1 256x32 / 2.72e11 shape are appended to the winning-checkpoints file.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "ESTIMATOR_HISTORY.md"
OUT = REPO / "history"
PRUNE_COMMIT = "88ea3f0^:docs/how-to/estimator-history.md"

FILES = {
    "Current Estimator": "01-current-estimator.md",
    "Winning Checkpoints": "02-winning-checkpoints.md",
    "Rejected Or Guarded Ideas": "03-rejected-and-guarded-ideas.md",
    "2026-07-10 Gaussian QMC/LHS closures": "04-qmc-lhs-closures.md",
    "2026-08-10 Block-scaling closure and leaderboard calibration": "05-block-scaling-and-leaderboard.md",
    "Benchmarking Notes": "06-benchmarking-notes.md",
}

# Recovered sections already on the phase-1 shape; merged into the checkpoints file.
POST_WARMUP = {
    "Current Estimator": "Route as of 2026-06-28 (pre-refactor snapshot)",
    "Deep Hadamard First-Covariance Route": None,
    "32x256 Budget Retargeting": None,
}


def split_sections(text):
    lines = text.splitlines(keepends=True)
    bounds = [(i, l[3:].strip()) for i, l in enumerate(lines) if l.startswith("## ")]
    out = {}
    for n, (start, title) in enumerate(bounds):
        end = bounds[n + 1][0] if n + 1 < len(bounds) else len(lines)
        out[title] = "".join(lines[start:end]).rstrip() + "\n"
    return out, "".join(lines[: bounds[0][0]]), [t for _, t in bounds]


def main():
    OUT.mkdir(exist_ok=True)
    for stale in OUT.glob("*.md"):
        stale.unlink()

    current, preamble, order = split_sections(SRC.read_text())

    recovered = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "blob", PRUNE_COMMIT],
        capture_output=True, text=True, check=True).stdout
    rec, _, rec_order = split_sections(re.sub(r"\A# Estimator History\n+", "", recovered))
    warm = [t for t in rec_order if t not in POST_WARMUP]
    post = [t for t in rec_order if t in POST_WARMUP]

    (OUT / "00-warmup-round.md").write_text(
        "# Warmup round (256x8) — recovered\n\n"
        "The warmup-round estimator history, recovered from\n"
        "`docs/how-to/estimator-history.md` at commit `88ea3f0^`. That commit cut the\n"
        "file from 690 lines to 121. Verified against all commits on every branch: it\n"
        "is the only content-removing refactor, and the surviving history is otherwise\n"
        "purely additive.\n\n"
        "Shape here is width 256, depth 8 at a `6.8e10` budget. Three sections of the\n"
        "recovered file had already moved to the phase-1 256x32 / `2.72e11` shape and\n"
        "live in [`02-winning-checkpoints.md`](02-winning-checkpoints.md) instead.\n\n"
        "---\n\n" + "\n".join(rec[t] for t in warm))

    for title in order:
        body = re.sub(r"\A## ", "# ", current[title])
        if title == "Rejected Or Guarded Ideas":
            body = add_lane_index(body)
        if title == "Winning Checkpoints":
            body = body.rstrip() + (
                "\n\n## Pre-refactor checkpoints (recovered)\n\n"
                "Recovered from `docs/how-to/estimator-history.md` at `88ea3f0^`. These\n"
                "predate the 2026-06-28 refactor but were already on the phase-1 shape,\n"
                "so they belong with the checkpoints rather than the warmup round.\n")
            for t in post:
                sec = rec[t]
                if POST_WARMUP[t]:
                    sec = sec.replace(f"## {t}\n", f"## {POST_WARMUP[t]}\n", 1)
                body += "\n" + sec.replace("## ", "### ", 1)
        (OUT / FILES[title]).write_text(body)

    index = ["# Estimator history\n\n",
             "The decision-useful history for the estimator: the current route, benchmark\n"
             "checkpoints that changed direction, and rejected ideas likely to be retried.\n"
             "The estimator itself is not published while Phase 2 is live.\n\n",
             "## Contents\n\n",
             "- **[Summary](SUMMARY.md)** — start here: the arc, the results, and which numbers to trust\n",
             "- [Warmup round (256x8) — recovered](00-warmup-round.md)\n"]
    index += [f"- [{t}]({FILES[t]})\n" for t in order]
    index.append("\nGenerated from `ESTIMATOR_HISTORY.md` by `scripts/build_history_split.py`.\n"
                 "Edit that file, not these.\n")
    (OUT / "README.md").write_text("".join(index))

    apply_publication_rewrites()

    for f in sorted(OUT.glob("*.md")):
        print(f"{f.stat().st_size:>7}  {f.name}")
    print("\nNote: SUMMARY.md is hand-written and not regenerated.")


def apply_publication_rewrites():
    """Fix references that only resolve in the private working repo."""
    a = OUT / "05-block-scaling-and-leaderboard.md"
    a.write_text(a.read_text().replace(
        "[`GATE_REAUDIT_CORRECTED_SCORING.md`](GATE_REAUDIT_CORRECTED_SCORING.md)",
        "[`GATE_REAUDIT.md`](../GATE_REAUDIT.md)"))
    b = OUT / "06-benchmarking-notes.md"
    b.write_text(b.read_text().replace("[`AGENTS.md`](AGENTS.md)", "`AGENTS.md` (not published)"))
    r = OUT / "README.md"
    r.write_text(r.read_text().rstrip() + """

## Path conventions

These files were written against the private working repository, so artifact
paths in the prose do not all match the published layout:

- `paired_fly_logs/fingerprint_theory/...` is [`gates/`](../gates/) there.
- `paired_fly_logs/*.log` are the raw Fly run logs, which are **not** published:
  they carry presigned object-store URLs and machine identifiers. The
  measurements taken from them are in
  [`analysis/block_ladder/ladder_per_mlp_mse.csv`](../analysis/block_ladder/ladder_per_mlp_mse.csv).
- `estimator.py` and `AGENTS.md` are not published while Phase 2 is live.
""")


def add_lane_index(body):
    lines = body.splitlines(keepends=True)
    entries = [(i + 1, m.group(1).rstrip("."))
               for i, l in enumerate(lines)
               if (m := re.match(r"^- \*\*(.+?)\.?\*\*", l))]
    head_end = next(i for i, l in enumerate(lines) if l.startswith("- **"))
    toc = [f"\n\n## Closed lanes ({len(entries)})\n\n",
           "Each entry below is a measured rejection, not an untried idea. Line numbers\n"
           "are into this file.\n\n"]
    toc += [f"- L{ln}: {t}\n" for ln, t in entries]
    return "".join(lines[:head_end]).rstrip() + "".join(toc) + "\n" + "".join(lines[head_end:])


if __name__ == "__main__":
    sys.exit(main())
