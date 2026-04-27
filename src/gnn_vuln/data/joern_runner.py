"""
joern_runner.py
~~~~~~~~~~~~~~~
Subprocess wrapper around the Joern CLI for CPG extraction.

Joern pipeline per function:
    source .c file
        -> joern-parse   -> <name>.bin  (binary CPG)
        -> joern-export  -> <name>.graphml  (graph for GNN)

Requires Joern 1.1+ with joern-parse and joern-export on PATH,
or provide the joern-cli directory explicitly.

Download Joern: https://joern.io/docs/installation/
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Java / environment helpers
# ---------------------------------------------------------------------------

_COMMON_JDK_DIRS_WIN = [
    r"C:\Program Files\Java",
    r"C:\Program Files\Eclipse Adoptium",
    r"C:\Program Files\Microsoft",
    r"C:\Program Files\BellSoft",
]


def _find_java_home() -> Optional[str]:
    """Try to locate a JDK on the system when JAVA_HOME is not set."""
    # Already set
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]

    if sys.platform != "win32":
        return None  # on Unix, rely on PATH

    for base in _COMMON_JDK_DIRS_WIN:
        base_path = Path(base)
        if not base_path.exists():
            continue
        # Pick the newest JDK dir (sorted descending)
        candidates = sorted(base_path.iterdir(), reverse=True)
        for c in candidates:
            if c.is_dir() and (c / "bin" / "java.exe").exists() and (c / "bin" / "javac.exe").exists():
                return str(c)
    return None


def _build_env(java_home: Optional[str] = None) -> dict:
    """
    Build a subprocess environment that has JAVA_HOME and JDK bin on PATH.
    Joern's .bat launcher checks both JAVA_HOME and PATH for java/javac.
    """
    env = os.environ.copy()

    jh = java_home or _find_java_home()
    if jh:
        env["JAVA_HOME"] = jh
        jdk_bin = str(Path(jh) / "bin")
        # Prepend JDK bin so it takes priority over any JRE on PATH
        env["PATH"] = jdk_bin + os.pathsep + env.get("PATH", "")

    return env


# ---------------------------------------------------------------------------
# Executable discovery
# ---------------------------------------------------------------------------

def _exe(joern_cli_dir: Optional[Path], name: str) -> str:
    """
    Return the full path to a Joern CLI tool.
    On Windows, Joern ships .bat wrappers; on Unix, plain executables.
    """
    suffix = ".bat" if sys.platform == "win32" else ""
    if joern_cli_dir:
        candidate = joern_cli_dir / f"{name}{suffix}"
        if candidate.exists():
            return str(candidate)
        candidate_no_suffix = joern_cli_dir / name
        if candidate_no_suffix.exists():
            return str(candidate_no_suffix)
        raise FileNotFoundError(
            f"Could not find '{name}' in {joern_cli_dir}. "
            "Make sure you downloaded Joern and set --joern-cli correctly."
        )
    found = shutil.which(f"{name}{suffix}") or shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"'{name}' not found on PATH. Install Joern or pass --joern-cli."
    )


# ---------------------------------------------------------------------------
# Core Joern calls
# ---------------------------------------------------------------------------

def joern_parse(
    src_path: Path,
    output_bin: Path,
    joern_cli_dir: Optional[Path] = None,
    java_home: Optional[str] = None,
    timeout: int = 120,
) -> None:
    """Run `joern-parse <src_path> --output <output_bin>`."""
    exe = _exe(joern_cli_dir, "joern-parse")
    cmd = [exe, str(src_path), "--output", str(output_bin)]
    env = _build_env(java_home)
    result = subprocess.run(
        cmd, env=env, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd,
            output=result.stdout, stderr=result.stderr,
        )


def joern_export(
    cpg_bin: Path,
    out_dir: Path,
    joern_cli_dir: Optional[Path] = None,
    java_home: Optional[str] = None,
    repr: str = "all",
    fmt: str = "graphml",
    timeout: int = 120,
) -> list[Path]:
    """
    Run `joern-export --repr <repr> --format <fmt> --out <out_dir> <cpg_bin>`.

    Joern v4 syntax: CPG file is a positional arg; output dir must NOT exist.
    Use repr='all' with fmt='graphml' — cpg14+graphml is not supported in v4.
    Returns list of exported files.
    """
    # Joern requires the output dir to not exist yet
    if out_dir.exists():
        shutil.rmtree(out_dir)

    exe = _exe(joern_cli_dir, "joern-export")
    cmd = [
        exe,
        "--repr", repr,
        "--format", fmt,
        "--out", str(out_dir),
        str(cpg_bin),          # positional — must be last
    ]
    env = _build_env(java_home)
    result = subprocess.run(
        cmd, env=env, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd,
            output=result.stdout, stderr=result.stderr,
        )

    # Joern v4 writes 'export.xml' for repr=all
    exported = list(out_dir.glob("*.xml")) or list(out_dir.glob("*.graphml"))
    return exported


# ---------------------------------------------------------------------------
# Higher-level helper: one function -> one CPG file
# ---------------------------------------------------------------------------

def process_function(
    code: str,
    idx: int,
    out_dir: Path,
    joern_cli_dir: Optional[Path] = None,
    java_home: Optional[str] = None,
    tmp_root: Optional[Path] = None,
    fmt: str = "graphml",
) -> Optional[Path]:
    """
    Write a C function to a temp file, run Joern parse + export, copy
    the resulting CPG file to out_dir.

    Returns the destination Path on success, None on any failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        tmp_path = Path(tmp)

        src_file = tmp_path / f"func_{idx}.c"
        src_file.write_text(code, encoding="utf-8")

        cpg_bin = tmp_path / f"func_{idx}.bin"
        export_dir = tmp_path / "export"

        try:
            joern_parse(src_file, cpg_bin, joern_cli_dir, java_home)
            exported = joern_export(cpg_bin, export_dir, joern_cli_dir, java_home, fmt=fmt)
        except subprocess.CalledProcessError as e:
            # Decode stderr so the caller can see what went wrong
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            stdout = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise RuntimeError(
                f"Joern command failed (exit {e.returncode}).\n"
                f"STDOUT: {stdout[-1000:]}\nSTDERR: {stderr[-1000:]}"
            ) from e
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        if not exported:
            return None

        dest = out_dir / f"func_{idx}.{exported[0].suffix.lstrip('.')}"
        shutil.copy2(exported[0], dest)
        return dest
