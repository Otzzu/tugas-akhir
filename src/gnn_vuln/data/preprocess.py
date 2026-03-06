"""
preprocess.py
~~~~~~~~~~~~~
Source-code normalisation utilities before graph extraction.

Common steps used in vulnerability detection research:
  1. Strip comments
  2. Normalise variable / function names (optional)
  3. Remove blank lines
"""

from __future__ import annotations

import re


def strip_comments(code: str, lang: str = "c") -> str:
    """Remove C/C++ (or Java) block and line comments from *code*."""
    if lang in ("c", "cpp", "java"):
        # Remove /* ... */ block comments (including multi-line)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        # Remove // line comments
        code = re.sub(r"//[^\n]*", "", code)
    elif lang == "python":
        # Remove # comments
        code = re.sub(r"#[^\n]*", "", code)
    return code


def normalize_identifiers(code: str) -> str:
    """
    Replace all user-defined identifiers with generic tokens.

    This reduces vocabulary size and improves generalisation —
    a common trick in code-based ML (e.g., Devign, Reveal).

    NOTE: This is a *very* simplified version. For production use,
    integrate a proper language parser (e.g., tree-sitter).
    """
    # Replace string literals
    code = re.sub(r'"[^"]*"', '"STR"', code)
    code = re.sub(r"'[^']*'", "'STR'", code)
    # Replace integer / float literals (but keep operators)
    code = re.sub(r"\b\d+(\.\d+)?\b", "NUM", code)
    return code


def remove_blank_lines(code: str) -> str:
    """Collapse consecutive blank lines to a single blank line."""
    return re.sub(r"\n{3,}", "\n\n", code)


def preprocess(code: str, lang: str = "c", normalize: bool = False) -> str:
    """Full preprocessing pipeline for a single source file string."""
    code = strip_comments(code, lang)
    if normalize:
        code = normalize_identifiers(code)
    code = remove_blank_lines(code)
    return code.strip()
