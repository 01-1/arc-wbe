#!/usr/bin/env python3
"""Run many one-MLP WhestBench jobs on Fly.io EWR Machines."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import select
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
from statistics import median
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SOURCE_DATASET = "hf://aicrowd/arc-whestbench-public-2026@v1-phase1"
DEFAULT_SEED = 20260624
FLY_BIN_DIR = Path("/i/.fly/bin")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _ensure_fly_on_path() -> None:
    if shutil.which("fly") is None and (FLY_BIN_DIR / "fly").is_file():
        os.environ["PATH"] = f"{FLY_BIN_DIR}:{os.environ.get('PATH', '')}"


def _quote(value: object) -> str:
    import shlex

    return shlex.quote(str(value))


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
    timeout: float | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print("+ " + " ".join(_quote(part) for part in _redact_command(cmd)), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    try:
        return subprocess.run(
            cmd,
            cwd=cwd or REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(
            cmd,
            124,
            output + f"\ntimed out after {timeout:.1f}s\n",
            "",
        )


def _run_until_machine_id(
    cmd: list[str],
    *,
    dry_run: bool = False,
    timeout: float | None = None,
    quiet: bool = False,
) -> tuple[subprocess.CompletedProcess[str], float | None]:
    if not quiet:
        print("+ " + " ".join(_quote(part) for part in _redact_command(cmd)), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "Machine ID: dry-run-machine\n", ""), 0.0

    started_at = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    deadline = time.monotonic() + timeout if timeout is not None else None
    machine_id_elapsed: float | None = None
    returncode = 124
    try:
        while deadline is None or time.monotonic() < deadline:
            remaining = 1.0 if deadline is None else max(0.0, min(1.0, deadline - time.monotonic()))
            readable, _, _ = select.select([proc.stdout], [], [], remaining)
            if not readable:
                if proc.poll() is not None:
                    returncode = proc.returncode or 0
                    break
                continue
            line = proc.stdout.readline()
            if line == "":
                if proc.poll() is not None:
                    returncode = proc.returncode or 0
                    break
                continue
            lines.append(line)
            if _machine_id_from_output(line):
                machine_id_elapsed = time.monotonic() - started_at
                returncode = 0
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                break
        else:
            lines.append(f"\ntimed out after {timeout:.1f}s waiting for Machine ID\n")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    return subprocess.CompletedProcess(cmd, returncode, "".join(lines), ""), machine_id_elapsed


def _redact_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for index, part in enumerate(redacted[:-1]):
        if part == "--dataset-url":
            redacted[index + 1] = "<dataset-url>"
        elif part == "--estimator-url":
            redacted[index + 1] = "<estimator-url>"
        elif part == "--script-url":
            redacted[index + 1] = "<script-url>"
        elif part == "--bank-url":
            redacted[index + 1] = "<bank-url>"
        elif part == "--payload-url":
            redacted[index + 1] = "<payload-url>"
    return redacted


def _prepare_base_dataset(args: argparse.Namespace) -> Path:
    from scripts.cloud_whest_common import ensure_randomized_dataset

    if args.dataset is not None:
        dataset = args.dataset.resolve()
        if not dataset.is_dir():
            raise SystemExit(f"--dataset must be a directory: {dataset}")
        return dataset
    return ensure_randomized_dataset(
        n_mlps=args.n_mlps,
        seed=args.seed,
        source_dataset=args.source_dataset,
        split=args.split,
        force=args.force_dataset,
        uv_cache_dir=args.uv_cache_dir,
    )


def _split_dataset(args: argparse.Namespace, dataset: Path, fingerprint: str) -> list[Path]:
    output_root = REPO_ROOT / ".cache" / "whestbench" / f"fly-one-mlp-{fingerprint}-n{args.n_mlps}"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/split_whest_dataset.py",
        "--dataset",
        str(dataset),
        "--split",
        args.split,
        "--output-root",
        str(output_root),
        "--n-mlps",
        str(args.n_mlps),
    ]
    if args.force_dataset:
        cmd.append("--force")
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = args.uv_cache_dir
    env.setdefault("HF_HOME", "/i/e/.cache/huggingface")
    env.setdefault("HF_DATASETS_CACHE", "/i/e/.cache/huggingface/datasets")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stdout)
    paths = sorted(path.resolve() for path in output_root.glob("mlp-*") if path.is_dir())
    if len(paths) != args.n_mlps:
        raise SystemExit(f"expected {args.n_mlps} one-MLP datasets, got {len(paths)}")
    return paths


def _prepare_archives(one_mlp_dirs: list[Path], fingerprint: str) -> Path:
    archive_root = REPO_ROOT / ".cache" / "whestbench" / f"fly-archives-{fingerprint}-n{len(one_mlp_dirs)}"
    archive_root.mkdir(parents=True, exist_ok=True)
    for index, dataset in enumerate(one_mlp_dirs):
        archive = archive_root / f"mlp-{index:06d}.tar.gz"
        if archive.is_file():
            continue
        tmp = archive.with_suffix(".tar.gz.tmp")
        if tmp.exists():
            tmp.unlink()
        with tarfile.open(tmp, mode="w:gz", compresslevel=1) as tar:
            tar.add(dataset, arcname=dataset.name)
        tmp.rename(archive)
    return archive_root


def _object_key(args: argparse.Namespace, fingerprint: str, index: int) -> str:
    prefix = args.object_prefix.rstrip("/")
    return f"{prefix}/{fingerprint}/n{args.n_mlps}/mlp-{index:06d}.tar.gz"


def _estimator_key(args: argparse.Namespace, digest: str) -> str:
    prefix = args.estimator_object_prefix.rstrip("/")
    return f"{prefix}/{digest}/estimator.py"


def _research_script_key(args: argparse.Namespace, digest: str, source: Path) -> str:
    prefix = args.research_script_object_prefix.rstrip("/")
    return f"{prefix}/{digest}/{source.name}"


def _bank_object_key(args: argparse.Namespace, digest: str, source: Path) -> str:
    prefix = args.bank_object_prefix.rstrip("/")
    return f"{prefix}/{digest}/{source.name}"


def _payload_object_key(args: argparse.Namespace, digest: str) -> str:
    prefix = args.payload_object_prefix.rstrip("/")
    return f"{prefix}/{digest}/payload.tar.gz"


def _object_bucket(args: argparse.Namespace) -> str:
    bucket = args.object_bucket or os.environ.get("WHEST_OBJECT_STORE_BUCKET")
    if not bucket:
        raise SystemExit("Set --object-bucket or WHEST_OBJECT_STORE_BUCKET")
    return bucket


def _object_endpoint(args: argparse.Namespace) -> str | None:
    return (
        args.object_endpoint_url
        or os.environ.get("WHEST_OBJECT_STORE_ENDPOINT_URL")
        or os.environ.get("AWS_ENDPOINT_URL_S3")
    )


def _estimator_bucket(args: argparse.Namespace) -> str:
    if args.estimator_object_bucket:
        return args.estimator_object_bucket
    if args.dry_run and not (args.object_bucket or os.environ.get("WHEST_OBJECT_STORE_BUCKET")):
        return "dry-run-bucket"
    return _object_bucket(args)


def _estimator_endpoint(args: argparse.Namespace) -> str | None:
    return args.estimator_object_endpoint_url or _object_endpoint(args)


def _research_script_bucket(args: argparse.Namespace) -> str:
    if args.research_script_object_bucket:
        return args.research_script_object_bucket
    if args.dry_run and not (args.object_bucket or os.environ.get("WHEST_OBJECT_STORE_BUCKET")):
        return "dry-run-bucket"
    return _object_bucket(args)


def _research_script_endpoint(args: argparse.Namespace) -> str | None:
    return args.research_script_object_endpoint_url or _object_endpoint(args)


def _bank_bucket(args: argparse.Namespace) -> str:
    if args.bank_object_bucket:
        return args.bank_object_bucket
    if args.dry_run and not (args.object_bucket or os.environ.get("WHEST_OBJECT_STORE_BUCKET")):
        return "dry-run-bucket"
    return _object_bucket(args)


def _bank_endpoint(args: argparse.Namespace) -> str | None:
    return args.bank_object_endpoint_url or _object_endpoint(args)


def _payload_bucket(args: argparse.Namespace) -> str:
    if args.payload_object_bucket:
        return args.payload_object_bucket
    if args.dry_run and not (args.object_bucket or os.environ.get("WHEST_OBJECT_STORE_BUCKET")):
        return "dry-run-bucket"
    return _object_bucket(args)


def _payload_endpoint(args: argparse.Namespace) -> str | None:
    return args.payload_object_endpoint_url or _object_endpoint(args)


def _object_url(args: argparse.Namespace, key: str) -> str:
    if not args.object_base_url:
        raise SystemExit("--object-base-url is required so workers can read uploaded archives")
    base = args.object_base_url.rstrip("/")
    return f"{base}/{'/'.join(quote(part) for part in key.split('/'))}"


def _estimator_object_url(args: argparse.Namespace, key: str) -> str:
    base_url = args.estimator_object_base_url or args.object_base_url
    if not base_url:
        raise SystemExit(
            "--estimator-object-base-url or --object-base-url is required so workers can read estimator.py"
        )
    base = base_url.rstrip("/")
    return f"{base}/{'/'.join(quote(part) for part in key.split('/'))}"


def _research_script_object_url(args: argparse.Namespace, key: str) -> str:
    base_url = args.research_script_object_base_url or args.object_base_url
    if not base_url:
        raise SystemExit(
            "--research-script-object-base-url or --object-base-url is required so workers can read the script"
        )
    base = base_url.rstrip("/")
    return f"{base}/{'/'.join(quote(part) for part in key.split('/'))}"


def _bank_object_url(args: argparse.Namespace, key: str) -> str:
    base_url = args.bank_object_base_url or args.object_base_url
    if not base_url:
        raise SystemExit("--bank-object-base-url or --object-base-url is required so workers can read the bank")
    base = base_url.rstrip("/")
    return f"{base}/{'/'.join(quote(part) for part in key.split('/'))}"


def _payload_object_url(args: argparse.Namespace, key: str) -> str:
    base_url = args.payload_object_base_url or args.object_base_url
    if not base_url:
        raise SystemExit("--payload-object-base-url or --object-base-url is required so workers can read the payload")
    base = base_url.rstrip("/")
    return f"{base}/{'/'.join(quote(part) for part in key.split('/'))}"


def _presign_url(
    args: argparse.Namespace,
    key: str,
    *,
    bucket: str | None = None,
    endpoint: str | None = None,
) -> str:
    aws = shutil.which("aws")
    if aws is None:
        raise SystemExit("aws CLI is required for --presign-urls")
    object_bucket = (
        "dry-run-bucket"
        if args.dry_run
        and bucket is None
        and not (args.object_bucket or os.environ.get("WHEST_OBJECT_STORE_BUCKET"))
        else bucket or _object_bucket(args)
    )
    cmd = [aws]
    object_endpoint = endpoint if endpoint is not None else _object_endpoint(args)
    if object_endpoint:
        cmd.extend(["--endpoint-url", object_endpoint])
    cmd.extend(
        [
            "s3",
            "presign",
            f"s3://{object_bucket}/{key}",
            "--expires-in",
            str(args.presign_expires_in),
        ]
    )
    proc = _run(cmd, dry_run=args.dry_run, quiet=args.quiet_commands)
    if args.dry_run:
        return f"https://presigned.example.invalid/{quote(key)}"
    if proc.returncode != 0:
        raise SystemExit(proc.stdout or f"could not presign {key}")
    return (proc.stdout or "").strip()


def _object_urls(args: argparse.Namespace, fingerprint: str) -> list[str]:
    keys = [_object_key(args, fingerprint, index) for index in range(args.n_mlps)]
    if args.presign_urls:
        if args.dataset_url_cache:
            cached_urls = _read_dataset_url_cache(args, fingerprint, keys)
            if cached_urls is not None:
                return cached_urls
        urls = [""] * len(keys)
        with ThreadPoolExecutor(max_workers=args.object_concurrency) as pool:
            futures = {pool.submit(_presign_url, args, key): index for index, key in enumerate(keys)}
            for future in as_completed(futures):
                urls[futures[future]] = future.result()
        if args.dataset_url_cache:
            _write_dataset_url_cache(args, fingerprint, keys, urls)
        return urls
    return [_object_url(args, key) for key in keys]


def _read_dataset_url_cache(
    args: argparse.Namespace, fingerprint: str, keys: list[str]
) -> list[str] | None:
    cache = args.dataset_url_cache
    if cache is None or not cache.is_file():
        return None
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("fingerprint") != fingerprint:
        return None
    if payload.get("keys") != keys:
        return None
    expires_at = payload.get("expires_at")
    if not isinstance(expires_at, (int, float)) or time.time() >= float(expires_at) - 300:
        return None
    urls = payload.get("urls")
    if not isinstance(urls, list) or len(urls) != len(keys) or not all(isinstance(url, str) for url in urls):
        return None
    return urls


def _write_dataset_url_cache(
    args: argparse.Namespace, fingerprint: str, keys: list[str], urls: list[str]
) -> None:
    cache = args.dataset_url_cache
    if cache is None:
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "bucket": _object_bucket(args),
        "keys": keys,
        "urls": urls,
        "created_at": time.time(),
        "expires_at": time.time() + args.presign_expires_in,
    }
    cache.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_dataset_urls(args: argparse.Namespace) -> tuple[str, list[str]]:
    if args.dataset_fingerprint:
        return args.dataset_fingerprint, _object_urls(args, args.dataset_fingerprint)
    from scripts.cloud_whest_common import dataset_fingerprint

    base_dataset = _prepare_base_dataset(args)
    fingerprint = dataset_fingerprint(base_dataset)
    one_mlp_dirs = _split_dataset(args, base_dataset, fingerprint)
    archive_root = _prepare_archives(one_mlp_dirs, fingerprint)
    print(f"Prepared one-MLP archives: {archive_root}")
    _upload_archives(args, archive_root, fingerprint)
    return fingerprint, _object_urls(args, fingerprint)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_archive_root(args: argparse.Namespace) -> Path:
    return (args.payload_archive_dir or (REPO_ROOT / ".cache" / "whestbench" / "fly-payloads")).resolve()


def _payload_arcname(source: Path) -> str:
    source = source.resolve()
    try:
        rel = source.relative_to(REPO_ROOT)
    except ValueError:
        rel = Path(source.name)
    return rel.as_posix()


def _iter_payload_sources(args: argparse.Namespace) -> list[Path]:
    sources = [args.payload_manifest, *args.payload_file]
    if args.payload_manifest is None:
        raise SystemExit("--task payload requires --payload-manifest or --payload-url")
    resolved: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        path = source.resolve()
        if not path.exists():
            raise SystemExit(f"payload source does not exist: {source}")
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def _prepare_payload_archive(args: argparse.Namespace) -> tuple[Path, str]:
    manifest = args.payload_manifest.resolve()
    if not manifest.is_file():
        raise SystemExit(f"--payload-manifest must be a file: {args.payload_manifest}")
    sources = _iter_payload_sources(args)
    digest = hashlib.sha256()
    members: list[tuple[Path, str]] = []
    for source in sorted(sources, key=lambda path: path.as_posix()):
        if source.is_dir():
            for child in sorted(path for path in source.rglob("*") if path.is_file()):
                arcname = f"payload/{_payload_arcname(child)}"
                members.append((child, arcname))
        elif source.is_file():
            arcname = "payload/manifest.json" if source == manifest else f"payload/{_payload_arcname(source)}"
            members.append((source, arcname))
        else:
            raise SystemExit(f"payload source must be a file or directory: {source}")
    arcnames = [arcname for _, arcname in members]
    if len(set(arcnames)) != len(arcnames):
        duplicates = sorted({name for name in arcnames if arcnames.count(name) > 1})
        raise SystemExit("payload archive has duplicate paths: " + ", ".join(duplicates))
    for source, arcname in members:
        digest.update(arcname.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(source).encode("ascii"))
        digest.update(b"\0")
    short_digest = digest.hexdigest()[:16]
    archive_root = _payload_archive_root(args) / short_digest
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / "payload.tar.gz"
    tmp = archive.with_suffix(".tar.gz.tmp")
    if tmp.exists():
        tmp.unlink()
    with tarfile.open(tmp, mode="w:gz", compresslevel=1) as tar:
        for source, arcname in members:
            tar.add(source, arcname=arcname)
    tmp.rename(archive)
    return archive, short_digest


def _payload_url(args: argparse.Namespace) -> str:
    if args.payload_url:
        return args.payload_url
    archive, digest = _prepare_payload_archive(args)
    key = _payload_object_key(args, digest)
    bucket = _payload_bucket(args)
    endpoint = _payload_endpoint(args)
    if args.upload_payload:
        _upload_file(args, archive, key, bucket=bucket, endpoint=endpoint)
    if args.presign_urls:
        return _presign_url(args, key, bucket=bucket, endpoint=endpoint)
    return _payload_object_url(args, key)


def _upload_file(
    args: argparse.Namespace,
    source: Path,
    key: str,
    *,
    bucket: str | None = None,
    endpoint: str | None = None,
) -> None:
    bucket = (
        "dry-run-bucket"
        if args.dry_run
        and bucket is None
        and not (args.object_bucket or os.environ.get("WHEST_OBJECT_STORE_BUCKET"))
        else bucket or _object_bucket(args)
    )
    aws = shutil.which("aws")
    if aws is None:
        raise SystemExit("aws CLI is required for upload")
    cmd = [aws]
    object_endpoint = endpoint if endpoint is not None else _object_endpoint(args)
    if object_endpoint:
        cmd.extend(["--endpoint-url", object_endpoint])
    cmd.extend(["s3", "cp", str(source), f"s3://{bucket}/{key}", "--only-show-errors"])
    proc = _run(cmd, dry_run=args.dry_run, quiet=args.quiet_commands)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SystemExit(proc.stdout or f"upload failed for {source}")


def _read_estimator_url_cache(args: argparse.Namespace, key: str) -> str | None:
    cache = args.estimator_url_cache
    if cache is None or not cache.is_file():
        return None
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("key") != key:
        return None
    expires_at = payload.get("expires_at")
    if not isinstance(expires_at, (int, float)) or time.time() >= float(expires_at) - 300:
        return None
    url = payload.get("url")
    return url if isinstance(url, str) else None


def _write_estimator_url_cache(args: argparse.Namespace, key: str, url: str) -> None:
    cache = args.estimator_url_cache
    if cache is None:
        return
    payload = {
        "key": key,
        "url": url,
        "created_at": time.time(),
        "expires_at": time.time() + args.presign_expires_in,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(cache)


def _estimator_url(args: argparse.Namespace) -> str:
    if args.estimator_url:
        return args.estimator_url
    estimator = args.estimator.resolve()
    if not estimator.is_file():
        raise SystemExit(f"--estimator must be a file: {estimator}")
    digest = _file_sha256(estimator)[:16]
    key = _estimator_key(args, digest)
    bucket = _estimator_bucket(args)
    endpoint = _estimator_endpoint(args)
    if args.presign_urls:
        cached_url = _read_estimator_url_cache(args, key)
        if cached_url is not None:
            return cached_url
    if args.upload_estimator:
        _upload_file(args, estimator, key, bucket=bucket, endpoint=endpoint)
    if args.presign_urls:
        url = _presign_url(args, key, bucket=bucket, endpoint=endpoint)
        _write_estimator_url_cache(args, key, url)
        return url
    return _estimator_object_url(args, key)


def _research_script_url(args: argparse.Namespace) -> str:
    if args.research_script_url:
        return args.research_script_url
    script = args.research_script.resolve()
    if not script.is_file():
        raise SystemExit(f"--research-script must be a file: {script}")
    digest = _file_sha256(script)[:16]
    key = _research_script_key(args, digest, script)
    bucket = _research_script_bucket(args)
    endpoint = _research_script_endpoint(args)
    if args.upload_research_script:
        _upload_file(args, script, key, bucket=bucket, endpoint=endpoint)
    if args.presign_urls:
        return _presign_url(args, key, bucket=bucket, endpoint=endpoint)
    return _research_script_object_url(args, key)


def _bank_url(args: argparse.Namespace) -> str:
    if args.bank_url:
        return args.bank_url
    bank = args.bank.resolve()
    if not bank.is_file():
        raise SystemExit(f"--bank must be a file: {bank}")
    digest = _file_sha256(bank)[:16]
    key = _bank_object_key(args, digest, bank)
    bucket = _bank_bucket(args)
    endpoint = _bank_endpoint(args)
    if args.upload_bank:
        _upload_file(args, bank, key, bucket=bucket, endpoint=endpoint)
    if args.presign_urls:
        return _presign_url(args, key, bucket=bucket, endpoint=endpoint)
    return _bank_object_url(args, key)


def _truth_seed_for_index(args: argparse.Namespace, index: int) -> int:
    label = f"{args.truth_seed_label}:{index}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(label).digest()[:8], "big") & ((1 << 63) - 1)
    forbidden = {11, 22, 33}
    while seed in forbidden:
        label = f"{args.truth_seed_label}:rehash:{index}:{seed}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(label).digest()[:8], "big") & ((1 << 63) - 1)
    return seed


def _upload_archives(args: argparse.Namespace, archive_root: Path, fingerprint: str) -> None:
    if not args.upload:
        return
    bucket = _object_bucket(args)
    aws = shutil.which("aws")
    if aws is None:
        raise SystemExit("aws CLI is required for --upload")
    endpoint = _object_endpoint(args)
    def upload_one(index: int) -> None:
        archive = archive_root / f"mlp-{index:06d}.tar.gz"
        key = _object_key(args, fingerprint, index)
        cmd = [aws]
        if endpoint:
            cmd.extend(["--endpoint-url", endpoint])
        cmd.extend(["s3", "cp", str(archive), f"s3://{bucket}/{key}", "--only-show-errors"])
        proc = _run(cmd, dry_run=args.dry_run, quiet=args.quiet_commands)
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0:
            raise SystemExit(proc.stdout or f"upload failed for {archive}")
    with ThreadPoolExecutor(max_workers=args.object_concurrency) as pool:
        futures = {pool.submit(upload_one, index): index for index in range(args.n_mlps)}
        for future in as_completed(futures):
            future.result()


def _prepare_context(args: argparse.Namespace) -> Path:
    context = args.context_dir or (REPO_ROOT / ".cache" / "whestbench" / "fly-context-code")
    context = context.resolve()
    if context.exists():
        shutil.rmtree(context)
    context.mkdir(parents=True)

    shutil.copy2(REPO_ROOT / "pyproject.toml", context / "pyproject.toml")
    shutil.copy2(REPO_ROOT / "uv.lock", context / "uv.lock")
    for name in ("estimator_covariance.py", "local_engine.py"):
        source = REPO_ROOT / name
        if source.is_file():
            shutil.copy2(source, context / name)
    scripts_dir = context / "scripts"
    scripts_dir.mkdir()
    for name in (
        "cloud_whest_common.py",
        "fly_bank_gate_entrypoint.py",
        "fly_object_entrypoint.py",
        "fly_truth_entrypoint.py",
        "remote_whest_run.py",
        "whest_with_residual_multiplier.py",
    ):
        source = REPO_ROOT / "scripts" / name
        if source.is_file():
            shutil.copy2(source, scripts_dir / name)
    shutil.copy2(REPO_ROOT / "cloud" / "fly-whest.Dockerfile", context / "Dockerfile")
    (context / "fly.toml").write_text(
        f'app = "{args.app}"\nprimary_region = "{args.region}"\n',
        encoding="utf-8",
    )
    (context / ".dockerignore").write_text(
        ".git\n.cache\n.venv\n__pycache__\n*.pyc\n",
        encoding="utf-8",
    )
    return context


def _build_image(args: argparse.Namespace, context: Path, image_label: str) -> str:
    image = f"registry.fly.io/{args.app}:{image_label}"
    proc = _run(
        [
            "fly",
            "deploy",
            ".",
            "--app",
            args.app,
            "--dockerfile",
            "Dockerfile",
            "--build-only",
            "--push",
            "--remote-only",
            "--image-label",
            image_label,
            "--yes",
        ],
        cwd=context,
        dry_run=args.dry_run,
        quiet=args.quiet_commands,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        raise SystemExit(f"fly deploy failed with {proc.returncode}")
    return image


def _entrypoint_args(
    args: argparse.Namespace,
    *,
    dataset_url: str | None,
    estimator_url: str | None,
    script_url: str | None,
    bank_url: str | None,
    payload_url: str | None,
    index: int,
    done_sentinel: str,
) -> list[str]:
    cmd = ["--task", args.task]
    if args.task == "truth":
        if script_url is None:
            raise SystemExit("--task truth requires a research script URL")
        cmd.extend(
            [
                "--script-url",
                script_url,
                "--mlp-index",
                str(index),
                "--seed",
                str(_truth_seed_for_index(args, index)),
                "--truth-width",
                str(args.truth_width),
                "--truth-depth",
                str(args.truth_depth),
                "--truth-target-seconds",
                str(args.truth_target_seconds),
                "--truth-chunk-pairs",
                str(args.truth_chunk_pairs),
                "--truth-min-pairs",
                str(args.truth_min_pairs),
            ]
        )
    elif args.task == "bank":
        if script_url is None or estimator_url is None or bank_url is None:
            raise SystemExit("--task bank requires research script, estimator, and bank URLs")
        cmd.extend(
            [
                "--script-url",
                script_url,
                "--estimator-url",
                estimator_url,
                "--bank-url",
                bank_url,
                "--shard-index",
                str(index),
                "--shard-count",
                str(args.n_mlps),
                "--flop-budget",
                str(args.flop_budget),
                "--setup-seed",
                str(args.bank_setup_seed),
            ]
        )
        if args.mode:
            cmd.extend(["--mode", args.mode])
    elif args.task == "payload":
        if payload_url is None:
            raise SystemExit("--task payload requires a payload URL")
        cmd.extend(
            [
                "--payload-url",
                payload_url,
                "--shard-index",
                str(index),
                "--shard-count",
                str(args.n_mlps),
            ]
        )
    else:
        if dataset_url is None or estimator_url is None:
            raise SystemExit("--task whest requires dataset and estimator URLs")
        cmd.extend(
            [
                "--dataset-url",
                dataset_url,
                "--estimator-url",
                estimator_url,
                "--split",
                args.split,
                "--flop-budget",
                str(args.flop_budget),
                "--wall-time-limit",
                str(args.wall_time_limit),
                "--residual-wall-time-multiplier",
                str(args.residual_wall_time_multiplier),
                "--max-threads",
                str(args.max_threads),
                "--runner",
                args.worker_runner,
                "--format",
                args.format,
                "--detail",
                args.detail,
            ]
        )
        if args.mode:
            cmd.extend(["--mode", args.mode])
    cmd.extend(
        [
        "--done-sentinel",
        done_sentinel,
        "--linger-after-result",
        str(args.linger_after_result),
        ]
    )
    return cmd


def _machine_command(
    args: argparse.Namespace,
    *,
    image: str,
    dataset_url: str | None,
    estimator_url: str | None,
    script_url: str | None,
    bank_url: str | None,
    payload_url: str | None,
    index: int,
    run_id: str,
    machine_name: str,
    done_sentinel: str,
) -> list[str]:
    cmd = [
        "fly",
        "machine",
        "run",
        "--app",
        args.app,
        "--region",
        args.region,
        "--vm-size",
        args.vm_size,
        "--name",
        machine_name,
        "--rm",
        "--restart",
        "no",
        "--skip-dns-registration",
    ]
    if args.detach_launch:
        cmd.append("--detach")
    return cmd + [
        image,
        "--",
        *_entrypoint_args(
            args,
            dataset_url=dataset_url,
            estimator_url=estimator_url,
            script_url=script_url,
            bank_url=bank_url,
            payload_url=payload_url,
            index=index,
            done_sentinel=done_sentinel,
        ),
    ]


def _fly_api_token(args: argparse.Namespace) -> str:
    token = os.environ.get("FLY_API_TOKEN")
    if token:
        return token.strip()
    proc = _run(["fly", "auth", "token"], dry_run=args.dry_run, quiet=True, timeout=15)
    if args.dry_run:
        return "dry-run-token"
    if proc.returncode != 0 or not proc.stdout.strip():
        raise SystemExit(proc.stdout or "could not get Fly API token")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(("FlyV1 ", "FlyV1.", "fm", "fo")):
            return line
    return lines[-1]


def _guest_from_vm_size(vm_size: str, memory_mb: int | None = None) -> dict[str, object]:
    match = re.fullmatch(r"(shared|performance)-cpu-(\d+)x", vm_size)
    if not match:
        raise SystemExit(f"--launch-method api only supports cpu vm sizes like shared-cpu-8x, got {vm_size!r}")
    cpu_kind, cpus_text = match.groups()
    cpus = int(cpus_text)
    memory_by_size = {
        "shared-cpu-1x": 256,
        "shared-cpu-2x": 512,
        "shared-cpu-4x": 1024,
        "shared-cpu-8x": 2048,
    }
    return {
        "cpu_kind": cpu_kind,
        "cpus": cpus,
        "memory_mb": memory_mb if memory_mb is not None else memory_by_size.get(vm_size, max(256, cpus * 256)),
    }


def _run_machine_api(
    args: argparse.Namespace,
    *,
    image: str,
    dataset_url: str | None,
    estimator_url: str | None,
    script_url: str | None,
    bank_url: str | None,
    payload_url: str | None,
    index: int,
    machine_name: str,
    done_sentinel: str,
) -> tuple[subprocess.CompletedProcess[str], float | None]:
    if args.dry_run:
        return subprocess.CompletedProcess(["fly-api"], 0, "Machine ID: dry-run-machine\n", ""), 0.0

    payload = {
        "name": machine_name,
        "region": args.region,
        "config": {
            "image": image,
            "init": {
                "cmd": _entrypoint_args(
                    args,
                    dataset_url=dataset_url,
                    estimator_url=estimator_url,
                    script_url=script_url,
                    bank_url=bank_url,
                    payload_url=payload_url,
                    index=index,
                    done_sentinel=done_sentinel,
                )
            },
            "guest": _guest_from_vm_size(args.vm_size, args.vm_memory_mb),
            "auto_destroy": True,
            "restart": {"policy": "no"},
            "dns": {"skip_registration": True},
        },
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.machines.dev/v1/apps/{quote(args.app)}/machines",
        data=body,
        headers={
            "Authorization": f"Bearer {_fly_api_token(args)}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    started_at = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.machine_run_timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(["fly-api"], exc.code, text, ""), time.monotonic() - started_at
    except OSError as exc:
        return subprocess.CompletedProcess(["fly-api"], 1, str(exc), ""), time.monotonic() - started_at
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {}
    machine_id = parsed.get("id") if isinstance(parsed, dict) else None
    if not isinstance(machine_id, str):
        return subprocess.CompletedProcess(["fly-api"], 1, text, ""), time.monotonic() - started_at
    return (
        subprocess.CompletedProcess(["fly-api"], 0, f"Machine ID: {machine_id}\n{text}", ""),
        time.monotonic() - started_at,
    )


def _machine_id_from_output(output: str) -> str | None:
    for line in output.splitlines():
        if line.strip().startswith("Machine ID:"):
            return line.split(":", 1)[1].strip()
    return None


def _machine_id_from_name(args: argparse.Namespace, name: str) -> str | None:
    proc = _run(
        ["fly", "machines", "list", "--app", args.app, "--json"],
        dry_run=args.dry_run,
        timeout=15,
        quiet=args.quiet_commands,
    )
    if args.dry_run:
        return "dry-run-machine"
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        machines = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    for machine in machines:
        if machine.get("name") == name:
            machine_id = machine.get("id")
            if isinstance(machine_id, str):
                return machine_id
    return None


def _wait_for_log_sentinel(
    args: argparse.Namespace, machine_id: str, done_sentinel: str
) -> subprocess.CompletedProcess[str]:
    cmd = ["fly", "logs", "--app", args.app, "--machine", machine_id]
    if not args.quiet_commands:
        print("+ " + " ".join(_quote(part) for part in cmd), flush=True)
    if args.dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")

    sentinel_bytes = f"{done_sentinel} returncode=".encode("utf-8")
    result_json_bytes = b"WHEST_RESULT_JSON "
    result_json_b64_done_bytes = b"WHEST_RESULT_JSON_B64_DONE "
    fly_exit_bytes = b"Main child exited normally with code:"
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    chunks: list[bytes] = []
    deadline = time.monotonic() + args.result_wait_timeout
    try:
        while time.monotonic() < deadline:
            remaining = max(0.0, min(1.0, deadline - time.monotonic()))
            readable, _, _ = select.select([proc.stdout], [], [], remaining)
            if not readable:
                if proc.poll() is not None:
                    break
                continue
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                if proc.poll() is not None:
                    break
                continue
            chunks.append(chunk)
            combined_tail = b"".join(chunks[-2:])
            if (
                sentinel_bytes in chunk
                or sentinel_bytes in combined_tail
                or result_json_bytes in chunk
                or result_json_bytes in combined_tail
                or result_json_b64_done_bytes in chunk
                or result_json_b64_done_bytes in combined_tail
                or fly_exit_bytes in chunk
                or fly_exit_bytes in combined_tail
            ):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                output = b"".join(chunks).decode("utf-8", errors="replace")
                return subprocess.CompletedProcess(
                    cmd,
                    _sentinel_returncode(output, done_sentinel),
                    output,
                    "",
                )
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
    return subprocess.CompletedProcess(
        cmd,
        124,
        b"".join(chunks).decode("utf-8", errors="replace")
        or f"timed out waiting for log sentinel on {machine_id}\n",
        "",
    )


def _sentinel_returncode(output: str, done_sentinel: str) -> int:
    for line in output.splitlines():
        marker = "returncode="
        if done_sentinel not in line or marker not in line:
            continue
        value = line.rsplit(marker, 1)[1].strip().split()[0]
        try:
            return int(value)
        except ValueError:
            return 1
    if "WHEST_RESULT_JSON " in output or "WHEST_RESULT_JSON_B64_DONE " in output:
        return 0
    for line in output.splitlines():
        marker = "Main child exited normally with code:"
        if marker not in line:
            continue
        value = line.rsplit(marker, 1)[1].strip().split()[0]
        try:
            return int(value)
        except ValueError:
            return 1
    return 124


def _clean_fly_log_content(line: str) -> str:
    line = ANSI_RE.sub("", line).strip()
    if "[info]" in line:
        return line.split("[info]", 1)[1].strip()
    return line


def _extract_json_objects(output: str) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    collecting = False
    depth = 0
    lines: list[str] = []
    for raw_line in output.splitlines():
        line = _clean_fly_log_content(raw_line)
        if not collecting:
            if not line.startswith("{"):
                continue
            collecting = True
            lines = []
            depth = 0
        lines.append(line)
        depth += line.count("{") - line.count("}")
        if collecting and depth <= 0:
            text = "\n".join(lines)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, dict):
                    objects.append(parsed)
            collecting = False
            lines = []
            depth = 0
    return objects


def _whest_results_from_output(output: str) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    b64_chunks: dict[int, str] = {}
    b64_expected: int | None = None
    for raw_line in output.splitlines():
        line = _clean_fly_log_content(raw_line)
        marker = "WHEST_RESULT_JSON "
        if marker not in line:
            chunk_marker = "WHEST_RESULT_JSON_B64_CHUNK "
            done_marker = "WHEST_RESULT_JSON_B64_DONE "
            if chunk_marker in line:
                parts = line.split(chunk_marker, 1)[1].strip().split(" ", 1)
                if len(parts) == 2:
                    try:
                        b64_chunks[int(parts[0])] = parts[1].strip()
                    except ValueError:
                        pass
            elif done_marker in line:
                try:
                    b64_expected = int(line.split(done_marker, 1)[1].strip().split()[0])
                except ValueError:
                    pass
            continue
        text = line.split(marker, 1)[1].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            results.append(parsed)
    if b64_expected is not None and len(b64_chunks) >= b64_expected:
        try:
            encoded = "".join(b64_chunks[index] for index in range(b64_expected))
            parsed = json.loads(base64.b64decode(encoded).decode("utf-8"))
        except (KeyError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            results.append(parsed)
    for obj in _extract_json_objects(output):
        result = obj.get("results")
        if isinstance(result, dict):
            results.append(result)
    return results


def _numeric(result: dict[str, object], key: str) -> float | None:
    value = result.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _fmt_sci(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3e}"


def _mean_numeric(whest_results: list[dict[str, object]], key: str) -> float | None:
    values = [value for result in whest_results if (value := _numeric(result, key)) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _sum_numeric(whest_results: list[dict[str, object]], key: str) -> float | None:
    values = [value for result in whest_results if (value := _numeric(result, key)) is not None]
    if not values:
        return None
    return sum(values)


def _scaled_whest_values(
    args: argparse.Namespace, whest_results: list[dict[str, object]]
) -> dict[str, float] | None:
    residual_compute_scale = args.residual_compute_scale
    if residual_compute_scale is None:
        return None

    scores: list[float] = []
    multipliers: list[float] = []
    utilizations: list[float] = []
    effective_compute: list[float] = []
    residual_compute: list[float] = []
    for result in whest_results:
        final_layer_mse = _numeric(result, "final_layer_mse")
        flops = _numeric(result, "mlp_flops_used")
        measured_residual = _numeric(result, "mlp_residual_compute")
        if final_layer_mse is None or flops is None or measured_residual is None:
            continue
        scaled_residual = measured_residual * residual_compute_scale
        effective = flops + scaled_residual
        utilization = effective / args.flop_budget
        failed = (_numeric(result, "n_failed_mlps") or 0.0) > 0.0
        multiplier = 1.0 if failed else max(0.1, utilization)
        residual_compute.append(scaled_residual)
        effective_compute.append(effective)
        utilizations.append(utilization)
        multipliers.append(multiplier)
        scores.append(final_layer_mse * multiplier)

    if not scores:
        return None
    return {
        "score_mean": sum(scores) / len(scores),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_multiplier_mean": sum(multipliers) / len(multipliers),
        "compute_utilization_mean": sum(utilizations) / len(utilizations),
        "effective_compute_mean": sum(effective_compute) / len(effective_compute),
        "residual_compute_mean": sum(residual_compute) / len(residual_compute),
        "residual_compute_scale": residual_compute_scale,
    }


def _print_whest_aggregate(args: argparse.Namespace, whest_results: list[dict[str, object]]) -> None:
    if args.task == "truth":
        truth_results = [result for result in whest_results if result.get("task") == "truth"]
        if not truth_results:
            print("Truth aggregate: no results")
            return
        samples = [int(result["sample_count"]) for result in truth_results if isinstance(result.get("sample_count"), int)]
        wall_times = [
            float(result["wall_time_s"])
            for result in truth_results
            if isinstance(result.get("wall_time_s"), (int, float))
        ]
        flops = [int(result["flops"]) for result in truth_results if isinstance(result.get("flops"), int)]
        print(
            "\n".join(
                [
                    f"Truth aggregate ({len(truth_results)} returned MLPs)",
                    f"sample_count_min={min(samples) if samples else 'n/a'}",
                    f"sample_count_max={max(samples) if samples else 'n/a'}",
                    f"sample_count_mean={sum(samples) / len(samples):.3f}" if samples else "sample_count_mean=n/a",
                    f"wall_time_mean_s={sum(wall_times) / len(wall_times):.3f}" if wall_times else "wall_time_mean_s=n/a",
                    f"flops_mean={sum(flops) / len(flops):.3e}" if flops else "flops_mean=n/a",
                ]
            )
        )
        return
    if args.task == "bank":
        bank_results = [result for result in whest_results if result.get("task") == "bank"]
        if not bank_results:
            print("Bank aggregate: no results")
            return
        records = [
            record
            for result in bank_results
            for record in result.get("records", [])
            if isinstance(record, dict)
        ]
        failures = [
            failure
            for result in bank_results
            for failure in result.get("failures", [])
            if isinstance(failure, dict)
        ]
        all_mses = [
            float(record["all_layers_mse"])
            for record in records
            if isinstance(record.get("all_layers_mse"), (int, float))
        ]
        final_mses = [
            float(record["final_layer_mse"])
            for record in records
            if isinstance(record.get("final_layer_mse"), (int, float))
        ]
        print(
            "\n".join(
                [
                    f"Bank aggregate ({len(bank_results)} returned shards)",
                    f"records={len(records)} failures={len(failures)}",
                    f"all_layers_mse_mean={sum(all_mses) / len(all_mses):.6e}" if all_mses else "all_layers_mse_mean=n/a",
                    f"final_layer_mse_mean={sum(final_mses) / len(final_mses):.6e}" if final_mses else "final_layer_mse_mean=n/a",
                ]
            )
        )
        return
    if args.task == "payload":
        payload_results = [result for result in whest_results if result.get("task") == "payload"]
        if not payload_results:
            print("Payload aggregate: no results")
            return
        print(f"Payload aggregate ({len(payload_results)} returned shards)")
        return
    if not whest_results:
        print("WhestBench aggregate: no results")
        return

    failure_totals: dict[str, int] = {}
    for result in whest_results:
        breakdown = result.get("failure_breakdown")
        if not isinstance(breakdown, dict):
            continue
        for key, value in breakdown.items():
            if isinstance(value, int) and value:
                failure_totals[key] = failure_totals.get(key, 0) + value

    if failure_totals:
        failures = ", ".join(f"{key}={value}" for key, value in sorted(failure_totals.items()))
    else:
        failures = "none"

    one_mlp_scores = [
        value for result in whest_results if (value := _numeric(result, "mlp_adjusted_final_layer_score")) is not None
    ]
    score_min = min(one_mlp_scores) if one_mlp_scores else None
    score_max = max(one_mlp_scores) if one_mlp_scores else None
    scaled = _scaled_whest_values(args, whest_results)
    score_label = (
        "score_mean adjusted_final_layer_score_scaled"
        if scaled is not None
        else "score_mean adjusted_final_layer_score"
    )

    lines = [
        f"WhestBench aggregate ({len(whest_results)} returned MLPs)",
        f"{score_label}: {_fmt_sci(scaled['score_mean'] if scaled else _mean_numeric(whest_results, 'adjusted_final_layer_score'))}",
        f"score_min mlp_adjusted_final_layer_score: {_fmt_sci(scaled['score_min'] if scaled else score_min)}",
        f"score_max mlp_adjusted_final_layer_score: {_fmt_sci(scaled['score_max'] if scaled else score_max)}",
        f"final_layer_mse_mean: {_fmt_sci(_mean_numeric(whest_results, 'final_layer_mse'))}",
        f"all_layers_mse_mean: {_fmt_sci(_mean_numeric(whest_results, 'all_layers_mse'))}",
        f"score_multiplier_mean: {_fmt_sci(scaled['score_multiplier_mean'] if scaled else _mean_numeric(whest_results, 'mean_score_multiplier'))}",
        f"compute_utilization_mean: {_fmt_sci(scaled['compute_utilization_mean'] if scaled else _mean_numeric(whest_results, 'mean_compute_utilization'))}",
        f"flops_mean: {_fmt_sci(_mean_numeric(whest_results, 'mlp_flops_used'))} FLOPs",
        f"flops_total_returned: {_fmt_sci(_sum_numeric(whest_results, 'mlp_flops_used'))} FLOPs",
        f"effective_compute_mean: {_fmt_sci(scaled['effective_compute_mean'] if scaled else _mean_numeric(whest_results, 'mlp_effective_compute'))} FLOP-eq",
        f"residual_compute_mean: {_fmt_sci(scaled['residual_compute_mean'] if scaled else _mean_numeric(whest_results, 'mlp_residual_compute'))} FLOP-eq",
        f"residual_compute_scale: {_fmt_sci(scaled['residual_compute_scale'] if scaled else None)}",
        f"residual_compute_mean_measured: {_fmt_sci(_mean_numeric(whest_results, 'mlp_residual_compute'))} FLOP-eq",
        f"residual_wall_time_mean_measured: {_fmt_sci(_mean_numeric(whest_results, 'mlp_residual_wall_time_s'))} s",
        f"failed_mlps_mean: {_fmt_sci(_mean_numeric(whest_results, 'n_failed_mlps'))}",
        f"failures_total: {failures}",
    ]
    print("\n".join(lines))


def _run_machine(
    args: argparse.Namespace,
    image: str,
    dataset_urls: list[str],
    estimator_url: str | None,
    script_url: str | None,
    bank_url: str | None,
    payload_url: str | None,
    run_id: str,
    index: int,
) -> tuple[int, int, str, float | None, float | None, float | None]:
    started_at = time.monotonic()
    machine_name = f"whest-{run_id}-{index:06d}"
    done_sentinel = f"WHEST_RESULT_DONE {run_id}-{index:06d}"
    machine_cmd = _machine_command(
        args,
        image=image,
        dataset_url=dataset_urls[index] if dataset_urls else None,
        estimator_url=estimator_url,
        script_url=script_url,
        bank_url=bank_url,
        payload_url=payload_url,
        index=index,
        run_id=run_id,
        machine_name=machine_name,
        done_sentinel=done_sentinel,
    )
    started: subprocess.CompletedProcess[str] | None = None
    launch_elapsed: float | None = None
    for attempt in range(1, args.machine_run_retries + 1):
        launch_started_at = time.monotonic()
        if args.launch_method == "api":
            started, launch_elapsed = _run_machine_api(
                args,
                image=image,
                dataset_url=dataset_urls[index] if dataset_urls else None,
                estimator_url=estimator_url,
                script_url=script_url,
                bank_url=bank_url,
                payload_url=payload_url,
                index=index,
                machine_name=machine_name,
                done_sentinel=done_sentinel,
            )
        elif args.detach_launch:
            started, machine_id_elapsed = _run_until_machine_id(
                machine_cmd,
                dry_run=args.dry_run,
                timeout=args.machine_run_timeout,
                quiet=args.quiet_commands,
            )
            launch_elapsed = machine_id_elapsed
        else:
            started = _run(
                machine_cmd,
                dry_run=args.dry_run,
                timeout=args.machine_run_timeout,
                quiet=args.quiet_commands,
            )
            launch_elapsed = time.monotonic() - launch_started_at
        output = started.stdout or ""
        if started.returncode == 0:
            break
        retryable = (
            started.returncode == 124
            or "MANIFEST_UNKNOWN" in output
            or "manifest unknown" in output.lower()
        )
        if not retryable:
            break
        if attempt < args.machine_run_retries:
            time.sleep(args.machine_run_retry_delay)
    assert started is not None
    output = started.stdout or ""
    machine_id = _machine_id_from_output(output)
    if not machine_id and "successfully launched" in output.lower():
        machine_id = _machine_id_from_name(args, machine_name)
    if started.returncode != 0 and not machine_id:
        return index, started.returncode, output, None, launch_elapsed, None
    if not machine_id:
        return index, started.returncode, output, None, launch_elapsed, None

    log_started_at = time.monotonic()
    marker_wait = _wait_for_log_sentinel(args, machine_id, done_sentinel)
    log_elapsed = time.monotonic() - log_started_at
    result_elapsed = time.monotonic() - started_at
    output += marker_wait.stdout or ""
    if marker_wait.returncode != 0:
        return index, marker_wait.returncode, output, result_elapsed, launch_elapsed, log_elapsed

    if not args.ignore_destroy_time:
        wait = _run(
            [
                "fly",
                "machine",
                "wait",
                machine_id,
                "--app",
                args.app,
                "--state",
                "destroyed",
                "--wait-timeout",
                args.machine_wait_timeout,
            ],
            dry_run=args.dry_run,
            quiet=args.quiet_commands,
        )
        output += wait.stdout or ""
        return index, wait.returncode, output, result_elapsed, launch_elapsed, log_elapsed
    return index, 0, output, result_elapsed, launch_elapsed, log_elapsed


def _run_machines(
    args: argparse.Namespace,
    image: str,
    dataset_urls: list[str],
    estimator_url: str | None,
    script_url: str | None,
    bank_url: str | None,
    payload_url: str | None,
) -> None:
    if args.skip_run:
        return
    run_started_at = time.monotonic()
    run_id = secrets.token_hex(8)
    results: list[tuple[int, float]] = []
    launch_times: list[tuple[int, float]] = []
    log_times: list[tuple[int, float]] = []
    whest_results: list[dict[str, object]] = []
    result_jsonl = args.result_jsonl
    if result_jsonl is not None:
        result_jsonl.parent.mkdir(parents=True, exist_ok=True)
        result_jsonl.write_text("", encoding="utf-8")
    stopped_early = False
    pool = ThreadPoolExecutor(max_workers=args.launch_concurrency)
    pending: set[object] = set()
    try:
        start_event = threading.Event()

        def run_after_start(index: int) -> tuple[int, int, str, float | None, float | None, float | None]:
            start_event.wait()
            return _run_machine(args, image, dataset_urls, estimator_url, script_url, bank_url, payload_url, run_id, index)

        futures = {
            pool.submit(run_after_start, index): index
            for index in range(args.n_mlps)
        }
        start_event.set()
        failures: list[tuple[int, int, str]] = []
        pending = set(futures)
        stop_reason: str | None = None
        while pending:
            if (
                args.max_result_seconds is not None
                and time.monotonic() - run_started_at >= args.max_result_seconds
            ):
                stop_reason = f"max_result_seconds={args.max_result_seconds:.3f}"
                break
            timeout = None
            if args.max_result_seconds is not None:
                timeout = max(0.0, args.max_result_seconds - (time.monotonic() - run_started_at))
            ready = []
            try:
                for future in as_completed(pending, timeout=timeout):
                    ready.append(future)
                    break
            except FuturesTimeoutError:
                stop_reason = f"max_result_seconds={args.max_result_seconds:.3f}"
                break
            if not ready:
                stop_reason = "no_ready_futures"
                break
            future = ready[0]
            pending.remove(future)
            index, returncode, output, result_elapsed, launch_elapsed, log_elapsed = future.result()
            if result_elapsed is not None:
                results.append((index, result_elapsed))
            if launch_elapsed is not None:
                launch_times.append((index, launch_elapsed))
            if log_elapsed is not None:
                log_times.append((index, log_elapsed))
            if returncode == 0:
                parsed_results = _whest_results_from_output(output)
                whest_results.extend(parsed_results)
                if result_jsonl is not None and parsed_results:
                    with result_jsonl.open("a", encoding="utf-8") as handle:
                        for result in parsed_results:
                            handle.write(json.dumps(result, sort_keys=True) + "\n")
            if args.summary_only and returncode == 0:
                if args.progress:
                    print(
                        f"Fly MLP {index:06d} ok"
                        + (f" result_after={result_elapsed:.3f}s" if result_elapsed is not None else ""),
                        f"launch={launch_elapsed:.3f}s" if launch_elapsed is not None else "",
                        f"log_wait={log_elapsed:.3f}s" if log_elapsed is not None else "",
                        flush=True,
                    )
            else:
                print(f"\n===== Fly MLP {index:06d} returncode={returncode} =====")
                if result_elapsed is not None:
                    print(f"Result available after {result_elapsed:.3f}s")
                print(output, end="")
            if result_elapsed is not None:
                pass
            if returncode != 0:
                failures.append((index, returncode, output))
            if args.min_results is not None and len(results) >= args.min_results:
                stop_reason = f"min_results={args.min_results}"
                break
        if stop_reason and pending:
            for future in pending:
                future.cancel()
            print(f"\nStopping early: {stop_reason}; pending_mlps={len(pending)}", flush=True)
            stopped_early = True
            pool.shutdown(wait=False, cancel_futures=True)
            pool = None
    except KeyboardInterrupt:
        stopped_early = True
        for future in pending:
            future.cancel()
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
            pool = None
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    total_elapsed = time.monotonic() - run_started_at
    _print_whest_aggregate(args, whest_results)
    if not args.no_timing_summary:
        print("\n===== Fly timing summary =====")
        print(f"requested_mlps={args.n_mlps}")
        print(f"completed_mlps={len(results)}")
        print(f"failed_mlps={len(failures)}")
        print(f"pending_mlps={args.n_mlps - len(results) - len(failures)}")
        prepare_timings = getattr(args, "prepare_timings", {})
        if isinstance(prepare_timings, dict):
            for key, value in prepare_timings.items():
                if isinstance(value, (int, float)):
                    print(f"{key}={value:.3f}s")
        print(f"wall_time_to_all_results={total_elapsed:.3f}s")
        if results:
            elapsed = sorted(value for _, value in results)
            print(f"result_elapsed_min={elapsed[0]:.3f}s")
            print(f"result_elapsed_median={median(elapsed):.3f}s")
            for percentile in (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100):
                index = min(len(elapsed) - 1, int((percentile / 100) * (len(elapsed) - 1)))
                print(f"result_elapsed_p{percentile}={elapsed[index]:.3f}s")
            print(f"result_elapsed_max={elapsed[-1]:.3f}s")
            slowest = sorted(results, key=lambda item: item[1], reverse=True)[:5]
            print(
                "slowest_mlps="
                + ", ".join(f"{index:06d}:{elapsed_value:.3f}s" for index, elapsed_value in slowest)
            )
        if launch_times:
            elapsed = sorted(value for _, value in launch_times)
            print(f"launch_elapsed_min={elapsed[0]:.3f}s")
            print(f"launch_elapsed_median={median(elapsed):.3f}s")
            print(f"launch_elapsed_max={elapsed[-1]:.3f}s")
        if log_times:
            elapsed = sorted(value for _, value in log_times)
            print(f"log_wait_min={elapsed[0]:.3f}s")
            print(f"log_wait_median={median(elapsed):.3f}s")
            print(f"log_wait_max={elapsed[-1]:.3f}s")
    if args.no_timing_summary:
        print(f"\ncompleted_mlps={len(results)} failed_mlps={len(failures)} pending_mlps={args.n_mlps - len(results) - len(failures)}")
    if failures:
        failed = ", ".join(f"{index:06d}:{code}" for index, code, _ in failures)
        raise SystemExit(f"Fly machine failures: {failed}")
    if stopped_early:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, help="Existing Fly app used for registry and Machines.")
    parser.add_argument("--task", choices=("whest", "truth", "bank", "payload"), default="whest")
    parser.add_argument("--n-mlps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--dataset", type=Path, help="Use an already staged local baked dataset.")
    parser.add_argument("--dataset-fingerprint", help="Use already-uploaded object-store archives with this dataset fingerprint.")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--context-dir", type=Path)
    parser.add_argument("--estimator", type=Path, default=REPO_ROOT / "estimator.py")
    parser.add_argument("--image-label")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--region", default="ewr")
    parser.add_argument("--vm-size", default="shared-cpu-8x")
    parser.add_argument("--vm-memory-mb", type=int, help="Override Fly Machine guest memory_mb at launch time.")
    parser.add_argument("--launch-concurrency", type=int, default=100)
    parser.add_argument("--machine-wait-timeout", default="10m")
    parser.add_argument("--machine-run-retries", type=int, default=3)
    parser.add_argument("--machine-run-retry-delay", type=float, default=5.0)
    parser.add_argument("--machine-run-timeout", type=float, default=45.0)
    parser.add_argument("--launch-method", choices=("cli", "api"), default="cli")
    parser.add_argument("--detach-launch", action="store_true", help="Pass --detach to fly machine run, then collect results via fly logs.")
    parser.add_argument("--result-wait-timeout", type=float, default=600.0)
    parser.add_argument("--linger-after-result", type=float, default=60.0)
    parser.add_argument("--ignore-destroy-time", action="store_true")
    parser.add_argument("--object-bucket")
    parser.add_argument("--object-prefix", default="whest/fly-mlps")
    parser.add_argument("--estimator-object-prefix", default="whest/estimators")
    parser.add_argument("--estimator-object-bucket")
    parser.add_argument("--research-script", type=Path, default=REPO_ROOT / "scripts" / "fly_truth_entrypoint.py")
    parser.add_argument("--research-script-url")
    parser.add_argument("--research-script-object-prefix", default="whest/research-scripts")
    parser.add_argument("--research-script-object-bucket")
    parser.add_argument("--bank", type=Path, default=REPO_ROOT / "analysis" / "truth_bank" / "truth_bank.npz")
    parser.add_argument("--bank-url")
    parser.add_argument("--bank-object-prefix", default="whest/research-banks")
    parser.add_argument("--bank-object-bucket")
    parser.add_argument("--payload-manifest", type=Path, help="Manifest JSON to include as payload/manifest.json.")
    parser.add_argument(
        "--payload-file",
        type=Path,
        action="append",
        default=[],
        help="File or directory to include in the generic payload archive. May be repeated.",
    )
    parser.add_argument("--payload-url", help="Use an already-uploaded generic payload archive.")
    parser.add_argument("--payload-archive-dir", type=Path)
    parser.add_argument("--payload-object-prefix", default="whest/payloads")
    parser.add_argument("--payload-object-bucket")
    parser.add_argument("--object-base-url")
    parser.add_argument("--estimator-object-base-url")
    parser.add_argument("--research-script-object-base-url")
    parser.add_argument("--bank-object-base-url")
    parser.add_argument("--payload-object-base-url")
    parser.add_argument("--object-endpoint-url")
    parser.add_argument("--estimator-object-endpoint-url")
    parser.add_argument("--research-script-object-endpoint-url")
    parser.add_argument("--bank-object-endpoint-url")
    parser.add_argument("--payload-object-endpoint-url")
    parser.add_argument("--upload", action="store_true", help="Upload one-MLP archives with aws s3 cp.")
    parser.add_argument("--upload-estimator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload-research-script", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--upload-payload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--estimator-url")
    parser.add_argument(
        "--estimator-url-cache",
        type=Path,
        help="Reuse a presigned estimator.py URL for the same estimator digest until it is near expiry.",
    )
    parser.add_argument("--presign-urls", action="store_true", help="Pass presigned HTTPS object URLs to workers.")
    parser.add_argument("--presign-expires-in", type=int, default=3600)
    parser.add_argument(
        "--dataset-url-cache",
        type=Path,
        help="Reuse presigned dataset archive URLs from this JSON cache until they are near expiry.",
    )
    parser.add_argument("--object-concurrency", type=int, default=20)
    parser.add_argument("--flop-budget", type=int, default=272_000_000_000)
    parser.add_argument("--wall-time-limit", type=float, default=60.0)
    parser.add_argument("--residual-wall-time-multiplier", type=float, default=2.0)
    parser.add_argument("--max-threads", type=int, default=8)
    parser.add_argument(
        "--worker-runner",
        choices=("local", "subprocess", "server", "inprocess"),
        default="local",
        help="WhestBench runner used inside each one-MLP Fly worker.",
    )
    parser.add_argument("--mode")
    parser.add_argument("--truth-width", type=int, default=256)
    parser.add_argument("--truth-depth", type=int, default=32)
    parser.add_argument("--truth-target-seconds", type=float, default=60.0)
    parser.add_argument("--truth-chunk-pairs", type=int, default=1024)
    parser.add_argument("--truth-min-pairs", type=int, default=1024)
    parser.add_argument("--truth-seed-label", default="arc-whest-fly-truth-bank-20260706-v1")
    parser.add_argument("--bank-setup-seed", type=int, default=0)
    parser.add_argument("--format", choices=("plain", "json", "rich"), default="plain")
    parser.add_argument("--detail", choices=("raw", "full"), default="raw")
    parser.add_argument("--summary-only", action="store_true", help="Print compact per-machine status and final timing summary.")
    parser.add_argument("--progress", action="store_true", help="Print one status line per completed Fly machine.")
    parser.add_argument("--quiet-commands", action="store_true", help="Do not echo fly/aws commands before running them.")
    parser.add_argument("--no-timing-summary", action="store_true", help="Suppress detailed timing summary.")
    parser.add_argument("--result-jsonl", type=Path, help="Write one parsed WHEST_RESULT_JSON object per line.")
    parser.add_argument("--min-results", type=int, help="Stop once this many MLPs have returned results.")
    parser.add_argument("--max-result-seconds", type=float, help="Stop waiting for results after this many seconds.")
    parser.add_argument(
        "--residual-compute-scale",
        type=float,
        help="Multiply each measured per-MLP residual FLOP-equivalent by this factor before recomputing aggregate score.",
    )
    parser.add_argument("--uv-cache-dir", default="/i/e/.uv-cache")
    args = parser.parse_args(argv)
    if args.n_mlps <= 0:
        raise SystemExit("--n-mlps must be positive")
    if args.task == "truth":
        if args.truth_width <= 0 or args.truth_depth <= 0:
            raise SystemExit("--truth-width and --truth-depth must be positive")
        if args.truth_target_seconds <= 0:
            raise SystemExit("--truth-target-seconds must be positive")
        if args.truth_chunk_pairs <= 0 or args.truth_min_pairs <= 0:
            raise SystemExit("--truth-chunk-pairs and --truth-min-pairs must be positive")
    if args.task == "payload":
        if args.payload_url is None and args.payload_manifest is None:
            raise SystemExit("--task payload requires --payload-manifest or --payload-url")
    if args.min_results is not None and not (0 < args.min_results <= args.n_mlps):
        raise SystemExit("--min-results must be between 1 and --n-mlps")
    if args.residual_compute_scale is not None and args.residual_compute_scale <= 0:
        raise SystemExit("--residual-compute-scale must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    _ensure_fly_on_path()
    if "AWS_REGION" in os.environ and "AWS_DEFAULT_REGION" not in os.environ:
        os.environ["AWS_DEFAULT_REGION"] = os.environ["AWS_REGION"]
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    image_label = args.image_label or "whest-runner"
    prepare_timings: dict[str, float] = {}
    print(
        f"Preparing Fly run: task={args.task} app={args.app} mlps={args.n_mlps} region={args.region}",
        flush=True,
    )
    dataset_urls: list[str] = []
    estimator_url: str | None = None
    script_url: str | None = None
    bank_url: str | None = None
    payload_url: str | None = None
    if args.task == "truth":
        step_started_at = time.monotonic()
        print("Preparing research script URL...", flush=True)
        script_url = _research_script_url(args)
        prepare_timings["prepare_research_script_url_s"] = time.monotonic() - step_started_at
        prepare_timings["prepare_dataset_urls_s"] = 0.0
        prepare_timings["prepare_estimator_url_s"] = 0.0
        print(
            f"Research script URL ready ({prepare_timings['prepare_research_script_url_s']:.3f}s)",
            flush=True,
        )
    elif args.task == "bank":
        step_started_at = time.monotonic()
        print("Preparing estimator URL...", flush=True)
        estimator_url = _estimator_url(args)
        prepare_timings["prepare_estimator_url_s"] = time.monotonic() - step_started_at
        print(f"Estimator URL ready ({prepare_timings['prepare_estimator_url_s']:.3f}s)", flush=True)
        step_started_at = time.monotonic()
        print("Preparing truth bank URL...", flush=True)
        bank_url = _bank_url(args)
        prepare_timings["prepare_bank_url_s"] = time.monotonic() - step_started_at
        print(f"Truth bank URL ready ({prepare_timings['prepare_bank_url_s']:.3f}s)", flush=True)
        step_started_at = time.monotonic()
        print("Preparing research script URL...", flush=True)
        script_url = _research_script_url(args)
        prepare_timings["prepare_research_script_url_s"] = time.monotonic() - step_started_at
        prepare_timings["prepare_dataset_urls_s"] = 0.0
        print(
            f"Research script URL ready ({prepare_timings['prepare_research_script_url_s']:.3f}s)",
            flush=True,
        )
    elif args.task == "payload":
        step_started_at = time.monotonic()
        print("Preparing generic payload URL...", flush=True)
        payload_url = _payload_url(args)
        prepare_timings["prepare_payload_url_s"] = time.monotonic() - step_started_at
        prepare_timings["prepare_dataset_urls_s"] = 0.0
        prepare_timings["prepare_estimator_url_s"] = 0.0
        print(f"Payload URL ready ({prepare_timings['prepare_payload_url_s']:.3f}s)", flush=True)
    else:
        step_started_at = time.monotonic()
        print("Preparing dataset URLs...", flush=True)
        fingerprint, dataset_urls = _prepare_dataset_urls(args)
        prepare_timings["prepare_dataset_urls_s"] = time.monotonic() - step_started_at
        print(
            f"Fly dataset fingerprint: {fingerprint} ({prepare_timings['prepare_dataset_urls_s']:.3f}s)",
            flush=True,
        )
        step_started_at = time.monotonic()
        print("Preparing estimator URL...", flush=True)
        estimator_url = _estimator_url(args)
        prepare_timings["prepare_estimator_url_s"] = time.monotonic() - step_started_at
        print(f"Estimator URL ready ({prepare_timings['prepare_estimator_url_s']:.3f}s)", flush=True)
    if args.dry_run:
        payload: dict[str, object] = {"task": args.task}
        if args.task == "truth":
            payload.update(
                {
                    "script_url": script_url,
                    "truth_width": args.truth_width,
                    "truth_depth": args.truth_depth,
                    "truth_target_seconds": args.truth_target_seconds,
                    "truth_seed_label": args.truth_seed_label,
                    "truth_seeds_first_5": [
                        _truth_seed_for_index(args, index) for index in range(min(5, args.n_mlps))
                    ],
                }
            )
        elif args.task == "bank":
            payload.update(
                {
                    "estimator_url": estimator_url,
                    "bank_url": bank_url,
                    "script_url": script_url,
                    "bank_path": str(args.bank),
                    "bank_setup_seed": args.bank_setup_seed,
                    "shard_count": args.n_mlps,
                }
            )
        elif args.task == "payload":
            payload.update(
                {
                    "payload_url": payload_url,
                    "payload_manifest": str(args.payload_manifest) if args.payload_manifest else None,
                    "payload_files": [str(path) for path in args.payload_file],
                    "shard_count": args.n_mlps,
                }
            )
        else:
            payload.update(
                {
                    "dataset_urls": dataset_urls[: min(3, len(dataset_urls))],
                    "dataset_url_count": len(dataset_urls),
                    "estimator_url": estimator_url,
                }
            )
        print(json.dumps(payload, indent=2))
    image = f"registry.fly.io/{args.app}:{image_label}"
    if not args.skip_build:
        step_started_at = time.monotonic()
        context = _prepare_context(args)
        prepare_timings["prepare_context_s"] = time.monotonic() - step_started_at
        print(f"Prepared Fly context: {context}", flush=True)
        step_started_at = time.monotonic()
        image = _build_image(args, context, image_label)
        prepare_timings["build_image_s"] = time.monotonic() - step_started_at
    else:
        prepare_timings["prepare_context_s"] = 0.0
        prepare_timings["build_image_s"] = 0.0
    args.prepare_timings = prepare_timings
    print(f"Fly image: {image}", flush=True)
    if args.build_only:
        return 0
    _run_machines(args, image, dataset_urls, estimator_url, script_url, bank_url, payload_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
