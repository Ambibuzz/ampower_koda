"""Atomic replacement for source files the agent writes."""
import os
import stat
import tempfile
from pathlib import Path


def read_bytes(path: str) -> bytes | None:
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        return None


def atomic_write(path: str, content: bytes, *, expected: bytes | None, before_replace=None, on_temporary=None,
                 exclusive: bool = False) -> None:
    """Keep the original intact until a complete, flushed replacement is ready.

    A lost replace acknowledgement is reconciled against the desired bytes.
    Expected bytes reject an observed concurrent edit, including file creation.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    mode = stat.S_IMODE(os.stat(path).st_mode) if os.path.exists(path) else 0o644
    fd, temporary = tempfile.mkstemp(prefix=".koda-write-", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if on_temporary:
                on_temporary(temporary)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        if before_replace:
            before_replace()
        if read_bytes(path) != expected:
            raise ValueError("File changed while preparing the write; read current source and retry.")
        try:
            if exclusive:
                os.link(temporary, path)  # atomic no-overwrite publication of a new copy
            else:
                os.replace(temporary, path)
        except OSError:
            if read_bytes(path) != content:
                raise
    finally:
        try:
            # A failed replacement may leave our temporary copy read-only on
            # Windows after preserving the destination's mode above.
            linked = exclusive and os.path.exists(path) and os.path.samefile(temporary, path)
            if not linked:
                os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
            os.unlink(temporary)
        except FileNotFoundError:
            pass
