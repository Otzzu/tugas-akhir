"""Tests for data preprocessing utilities."""

import pytest
from gnn_vuln.data.preprocess import strip_comments, normalize_identifiers, preprocess


def test_strip_c_block_comments():
    code = "int x = 1; /* this is a comment */ int y = 2;"
    result = strip_comments(code, lang="c")
    assert "/*" not in result
    assert "*/" not in result
    assert "int x = 1;" in result


def test_strip_c_line_comments():
    code = "int x = 1; // inline comment\nint y = 2;"
    result = strip_comments(code, lang="c")
    assert "// inline comment" not in result
    assert "int y = 2;" in result


def test_normalize_string_literals():
    code = 'strcpy(buf, "hello world");'
    result = normalize_identifiers(code)
    assert '"STR"' in result
    assert '"hello world"' not in result


def test_normalize_number_literals():
    code = "int x = 42; float y = 3.14;"
    result = normalize_identifiers(code)
    assert "42" not in result
    assert "NUM" in result


def test_preprocess_pipeline():
    code = '''
    // This function has a bug
    int foo(char *buf) {
        /* copy user input */
        strcpy(buf, "hello");
        return 0;
    }
    '''
    result = preprocess(code, lang="c", normalize=False)
    assert "//" not in result
    assert "/*" not in result
    assert "strcpy" in result
