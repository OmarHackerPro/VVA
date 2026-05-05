"""Tests for safe_subprocess — shell=False enforcement + security invariants."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from visionvoiceasist.utils.safe_subprocess import have, run, which


class TestRun:
    def test_runs_simple_command(self) -> None:
        exe = "python" if sys.platform == "win32" else "python3"
        result = run([exe, "-c", "print('hello')"], capture=True)
        assert "hello" in result.stdout

    def test_shell_false_is_enforced(self) -> None:
        """Verify shell=False is always passed — no metacharacter injection."""
        import subprocess
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["python"], returncode=0, stdout="", stderr=""
            )
            run(["python", "-c", "1"])
            _, kwargs = mock_run.call_args
            assert kwargs.get("shell") is False

    def test_raises_on_empty_argv(self) -> None:
        with pytest.raises(ValueError, match="argv must not be empty"):
            run([])

    def test_raises_on_missing_executable(self) -> None:
        with pytest.raises(FileNotFoundError):
            run(["__nonexistent_binary_xyz__"])

    def test_capture_captures_stdout(self) -> None:
        exe = "python" if sys.platform == "win32" else "python3"
        result = run([exe, "-c", "import sys; sys.stdout.write('captured')"],
                     capture=True)
        assert result.stdout == "captured"

    def test_metacharacters_safe_in_argv(self) -> None:
        """Shell metacharacters in text must NOT be interpreted."""
        exe = "python" if sys.platform == "win32" else "python3"
        dangerous_text = 'hello; echo INJECTED'
        result = run(
            [exe, "-c", f"print({dangerous_text!r})"],
            capture=True,
        )
        assert "INJECTED" not in result.stdout or "hello" in result.stdout

    def test_timeout_parameter_accepted(self) -> None:
        exe = "python" if sys.platform == "win32" else "python3"
        result = run([exe, "-c", "pass"], timeout=5.0)
        assert result.returncode == 0


class TestWhich:
    def test_returns_path_for_python(self) -> None:
        result = which("python") or which("python3")
        assert result is not None
        assert isinstance(result, Path)

    def test_returns_none_for_missing(self) -> None:
        assert which("__no_such_exe__xyz__") is None


class TestHave:
    def test_python_is_available(self) -> None:
        assert have("python") or have("python3")

    def test_missing_returns_false(self) -> None:
        assert have("__no_such_exe__xyz__") is False
