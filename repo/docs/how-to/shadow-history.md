# Repo-Scale Shadow History

This repository includes a private, repo-local snapshot history for tiny recovery
points that should not become normal Git or jj commits.

The shadow history uses Git's own ignored-file rules:

```bash
git ls-files -co --exclude-standard
```

That means `.gitignore` is the single source of truth. Files ignored by normal
Git are ignored by shadow history too, including `.shadow-history/` itself.

## One-Shot Snapshot

```bash
make shadow-snapshot
```

This creates `.shadow-history/` if needed, copies the current Git-visible repo
contents into it, and commits only when there is an actual diff.

## Watch Mode

```bash
make shadow-watch
```

Watch mode is designed to be run as a background service or terminal session. It
takes an initial snapshot, then polls for repo changes, waits for a quiet period,
and commits another snapshot only if content changed.

Defaults:

```text
SHADOW_INTERVAL_SECONDS=5
SHADOW_QUIET_SECONDS=60
SHADOW_MIN_COMMIT_SECONDS=120
SHADOW_HISTORY_DIR=.shadow-history
```

Example with faster snapshots:

```bash
make shadow-watch SHADOW_QUIET_SECONDS=15 SHADOW_MIN_COMMIT_SECONDS=30
```

## Inspect Or Recover

```bash
make shadow-status
git -C .shadow-history log --oneline
git -C .shadow-history show <snapshot>:path/to/file
git -C .shadow-history diff <old-snapshot> <new-snapshot>
```

To restore a file manually:

```bash
git -C .shadow-history show <snapshot>:path/to/file > path/to/file
```

Keep this history private and disposable. Its job is microscopic recovery; the
normal repository history remains the place for meaningful commits.
