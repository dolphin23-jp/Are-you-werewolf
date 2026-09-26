"""Crash-safe file replacement for checkpoints, manifests and run state.

Every writer used to open ``<destination>.tmp`` and rename it over the target.
That name was fixed, so two writers could interleave in one temporary file, and
nothing was fsynced: after a power loss the rename could land before the data,
leaving an empty or truncated checkpoint where a resumable one was expected.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: the RunPod helpers only copy files
    fcntl = None  # type: ignore[assignment]


@contextmanager
def atomic_write(destination: str | Path) -> Iterator[IO[bytes]]:
    """Yield a binary handle whose contents replace ``destination`` on success.

    The temporary file is unique per call and lives beside the destination so
    the final ``os.replace`` stays on one filesystem. It is removed if writing
    fails, and the destination is left untouched.
    """

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    _fsync_directory(target.parent)


def atomic_write_text(destination: str | Path, text: str) -> None:
    with atomic_write(destination) as handle:
        handle.write(text.encode("utf-8"))


def atomic_write_json(destination: str | Path, payload: Any, **dumps_options: Any) -> None:
    """Write ``payload`` as indented UTF-8 JSON with a trailing newline."""

    options: dict[str, Any] = {"ensure_ascii": False, "indent": 2, **dumps_options}
    atomic_write_text(destination, json.dumps(payload, **options) + "\n")


@contextmanager
def exclusive_lock(path: str | Path) -> Iterator[None]:
    """Hold an advisory lock on ``path`` (created if missing) for the block.

    Used for read-modify-write of shared manifests: two trainers appending to
    one pool used to each rewrite the manifest from their own stale copy, and
    whichever finished last silently dropped the other's generation.
    """

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        yield
        return
    with lock_path.open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def checkpoint_file_name(value: object) -> str:
    """Validate a manifest's checkpoint entry as a bare file name.

    Pools only ever write ``g000123.npz`` beside their manifest. A manifest
    naming ``../x`` or an absolute path made the pool load, and the snapshot
    exporter read and write, files outside the pool directory.
    """

    if not isinstance(value, str):
        raise ValueError("checkpoint must be a string")
    if value in ("", ".", "..") or Path(value).name != value or "\\" in value:
        raise ValueError(f"checkpoint must be a file name inside the pool: {value!r}")
    return value


def _fsync_directory(directory: Path) -> None:
    """Persist the rename itself; best effort where directories can't be opened."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
