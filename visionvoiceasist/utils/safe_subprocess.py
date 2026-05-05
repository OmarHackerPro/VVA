"""Safe subprocess wrapper — never use shell=True with user-influenced data.

Replaces every ``os.system(f'... "{text}"')`` call in the legacy codebase.
Critical for security: text from OCR / Gemini may contain shell metacharacters.
"""

from __future__ import annotations

import logging
import shutil
import subprocess  # noqa: S404 — bandit understands; we use shell=False
from pathlib import Path

log = logging.getLogger(__name__)


def run(
    argv: list[str],
    *,
    timeout: float | None = 30.0,
    capture: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run *argv* with shell=False. Raises FileNotFoundError if executable missing.

    Args:
        argv: List of arguments — argv[0] must be the executable.
        timeout: Kill the process after this many seconds.
        capture: If True, capture stdout/stderr.
        check: If True, raise CalledProcessError on non-zero exit.
    """
    if not argv:
        raise ValueError("argv must not be empty")
    log.debug("subprocess: %s", argv)
    try:
        return subprocess.run(  # noqa: S603 — shell=False is enforced
            argv,
            shell=False,
            timeout=timeout,
            capture_output=capture,
            text=True,
            check=check,
        )
    except FileNotFoundError:
        log.warning("Executable not found: %s", argv[0])
        raise


def which(executable: str) -> Path | None:
    """Locate *executable* on PATH; return None if missing."""
    found = shutil.which(executable)
    return Path(found) if found else None


def have(executable: str) -> bool:
    return which(executable) is not None
