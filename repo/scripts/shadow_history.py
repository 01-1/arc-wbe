#!/usr/bin/env python3
"""Repo-local shadow history that snapshots Git-visible files."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SHADOW_DIR = ".shadow-history"
DEFAULT_INTERVAL_SECONDS = 5
DEFAULT_QUIET_SECONDS = 60
DEFAULT_MIN_COMMIT_SECONDS = 120


def run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def repo_root() -> Path:
    result = run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd())
    return Path(result.stdout.strip()).resolve()


def shadow_root(root: Path, shadow_dir: str) -> Path:
    path = Path(shadow_dir)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def git_visible_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    names = result.stdout.split(b"\0")
    return sorted(Path(name.decode("utf-8", "surrogateescape")) for name in names if name)


def ensure_shadow_repo(shadow: Path) -> None:
    shadow.mkdir(parents=True, exist_ok=True)
    if not (shadow / ".git").exists():
        run(["git", "init"], cwd=shadow, capture=False)


def remove_stale_files(shadow: Path, wanted: set[Path]) -> None:
    for path in sorted(shadow.rglob("*"), reverse=True):
        rel = path.relative_to(shadow)
        if rel.parts and rel.parts[0] == ".git":
            continue
        if path.is_file() or path.is_symlink():
            if rel not in wanted:
                path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def copy_visible_files(root: Path, shadow: Path, files: list[Path]) -> None:
    wanted = set(files)
    remove_stale_files(shadow, wanted)
    for rel in files:
        source = root / rel
        target = shadow / rel
        if not source.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def commit_if_changed(shadow: Path, *, message: str | None = None) -> bool:
    run(["git", "add", "-A"], cwd=shadow)
    diff = run(["git", "diff", "--cached", "--quiet"], cwd=shadow, check=False)
    if diff.returncode == 0:
        return False
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    commit_message = message or f"snapshot {stamp}"
    run(
        [
            "git",
            "-c",
            "user.name=shadow-history",
            "-c",
            "user.email=shadow-history@local",
            "commit",
            "-m",
            commit_message,
        ],
        cwd=shadow,
        capture=False,
    )
    return True


def snapshot(root: Path, shadow: Path, *, message: str | None = None) -> bool:
    ensure_shadow_repo(shadow)
    files = git_visible_files(root)
    shadow_rel = shadow.relative_to(root) if shadow.is_relative_to(root) else None
    if shadow_rel is not None:
        files = [path for path in files if not path.parts[: len(shadow_rel.parts)] == shadow_rel.parts]
    copy_visible_files(root, shadow, files)
    return commit_if_changed(shadow, message=message)


def current_tree_fingerprint(root: Path, shadow: Path) -> str:
    files = git_visible_files(root)
    shadow_rel = shadow.relative_to(root) if shadow.is_relative_to(root) else None
    parts: list[str] = []
    for rel in files:
        if shadow_rel is not None and rel.parts[: len(shadow_rel.parts)] == shadow_rel.parts:
            continue
        path = root / rel
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        parts.append(f"{rel.as_posix()}\0{stat.st_mtime_ns}\0{stat.st_size}")
    return "\n".join(parts)


def watch(args: argparse.Namespace, root: Path, shadow: Path) -> None:
    last_fingerprint = current_tree_fingerprint(root, shadow)
    last_change_seen = time.monotonic()
    last_commit = 0.0

    if args.initial:
        changed = snapshot(root, shadow)
        print("initial snapshot committed" if changed else "initial snapshot unchanged", flush=True)
        last_commit = time.monotonic()
        last_fingerprint = current_tree_fingerprint(root, shadow)

    print(
        f"watching {root} -> {shadow} "
        f"(quiet={args.quiet_seconds}s, min-commit={args.min_commit_seconds}s)",
        flush=True,
    )
    while True:
        time.sleep(args.interval_seconds)
        fingerprint = current_tree_fingerprint(root, shadow)
        now = time.monotonic()
        if fingerprint != last_fingerprint:
            last_fingerprint = fingerprint
            last_change_seen = now
            continue
        if now - last_change_seen < args.quiet_seconds:
            continue
        if now - last_commit < args.min_commit_seconds:
            continue
        changed = snapshot(root, shadow)
        last_commit = now
        if changed:
            print(f"snapshot committed at {datetime.now().isoformat(timespec='seconds')}", flush=True)


def show_status(root: Path, shadow: Path) -> None:
    print(f"repo:   {root}")
    print(f"shadow: {shadow}")
    if not (shadow / ".git").exists():
        print("shadow repo: not initialized")
        return
    result = run(["git", "log", "--oneline", "-5"], cwd=shadow, check=False)
    print("recent snapshots:")
    print(result.stdout.rstrip() or "  none")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shadow-dir",
        default=os.environ.get("SHADOW_HISTORY_DIR", DEFAULT_SHADOW_DIR),
        help="shadow Git repo path, relative to the repo root by default",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snap = subparsers.add_parser("snapshot", help="copy Git-visible files and commit if changed")
    snap.add_argument("-m", "--message", help="snapshot commit message")

    watch_parser = subparsers.add_parser("watch", help="run a polling background-service loop")
    watch_parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    watch_parser.add_argument("--quiet-seconds", type=float, default=DEFAULT_QUIET_SECONDS)
    watch_parser.add_argument("--min-commit-seconds", type=float, default=DEFAULT_MIN_COMMIT_SECONDS)
    watch_parser.add_argument("--no-initial", dest="initial", action="store_false")
    watch_parser.set_defaults(initial=True)

    subparsers.add_parser("status", help="show shadow history location and recent snapshots")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = repo_root()
    shadow = shadow_root(root, args.shadow_dir)

    if args.command == "snapshot":
        changed = snapshot(root, shadow, message=args.message)
        print("snapshot committed" if changed else "snapshot unchanged")
    elif args.command == "watch":
        watch(args, root, shadow)
    elif args.command == "status":
        show_status(root, shadow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
