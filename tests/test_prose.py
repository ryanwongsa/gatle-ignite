"""Prose rules for src, examples, scripts and tests: six-line cap, no issue numbers, no em dashes."""

import ast
import inspect
import io
import re
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MAX_LINES = 6

EM_DASH = "\u2014"

# Also matches a three-digit hex colour, which reads as an issue number in this repo.
_ISSUE_REF = re.compile(r"#\d+\b|\bGH-\d+\b|github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+")


def _non_empty(paths):
    return sorted(p for p in paths if p.read_text().strip())


FRAMEWORK_FILES = _non_empty((ROOT / "src" / "gatle_ignite").rglob("*.py"))
TEMPLATE_FILES = _non_empty((ROOT / "src" / "gatle_ignite" / "_template").rglob("*.py-tpl"))
EXAMPLE_FILES = _non_empty((ROOT / "examples").rglob("*.py"))
SCRIPT_FILES = _non_empty((ROOT / "scripts").glob("*.py"))
TEST_FILES = _non_empty((ROOT / "tests").glob("*.py"))
CHECKED_FILES = FRAMEWORK_FILES + TEMPLATE_FILES + EXAMPLE_FILES + SCRIPT_FILES + TEST_FILES
# By suffix, not every file: runs leave binary checkpoints under examples/.
EM_DASH_FILES = TEST_FILES + sorted(
    p
    for tree in ("src", "examples", "scripts")
    for pattern in ("*.py", "*-tpl")
    for p in (ROOT / tree).rglob(pattern)
)


def _source_id(path):
    return str(path.relative_to(ROOT))


def _read(path):
    """A template parses only once `gatle-ignite init` has filled in its project name."""
    return path.read_text().replace("{{project_name}}", "demo")


def _comment_tokens(src):
    """Every own-line comment token.

    Trailing comments are excluded: config/base.py documents each field in one, and
    gen_reference.py renders them into the config reference.
    """
    return [
        tok
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type == tokenize.COMMENT and not tok.line[: tok.start[1]].strip()
    ]


def _comment_blocks(src):
    """Runs of own-line comments, blank lines included. -> [(start_line, comment_lines)]

    A blank line does not end a block, so an essay cannot pass as short paragraphs.
    """
    lines = src.splitlines()
    blocks, current = [], []
    for tok in [*_comment_tokens(src), None]:
        joins = (
            current
            and tok is not None
            and not any(lines[i - 1].strip() for i in range(current[-1].start[0] + 1, tok.start[0]))
        )
        if joins:
            current.append(tok)
            continue
        if current:
            blocks.append((current[0].start[0], len(current)))
        current = [tok] if tok is not None else []
    return blocks


def _string_literals(src):
    """Every non-docstring string constant, with its line. -> [(lineno, text)]"""
    tree = ast.parse(src)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docs.add(id(body[0].value))
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
    ]


def _docstrings(src):
    """Every docstring, named and measured. -> [(name, lineno, length, text)]

    inspect.cleandoc, not textwrap.dedent: a raw docstring's first line has no indent to strip.
    """
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        name = "<module>" if isinstance(node, ast.Module) else node.name
        lineno = 1 if isinstance(node, ast.Module) else node.lineno
        found.append((name, lineno, len(inspect.cleandoc(doc).splitlines()), doc))
    return found


@pytest.mark.parametrize("path", CHECKED_FILES, ids=_source_id)
def test_no_comment_block_runs_past_six_lines(path):
    over = [(line, n) for line, n in _comment_blocks(_read(path)) if n > MAX_LINES]
    listing = "\n  ".join(f"{_source_id(path)}:{line}: {n} lines" for line, n in over)
    assert not over, (
        f"comment blocks longer than {MAX_LINES} lines:\n  {listing}\n"
        "State the rule and stop. How a bug was found belongs in its issue."
    )


@pytest.mark.parametrize("path", CHECKED_FILES, ids=_source_id)
def test_no_docstring_runs_past_six_lines(path):
    over = [(name, line, n) for name, line, n, _ in _docstrings(_read(path)) if n > MAX_LINES]
    listing = "\n  ".join(f"{_source_id(path)}:{line} {name}: {n} lines" for name, line, n in over)
    assert not over, (
        f"docstrings longer than {MAX_LINES} lines:\n  {listing}\n"
        "One line, unless it carries a rule the code cannot show or a list of accepted values."
    )


@pytest.mark.parametrize("path", CHECKED_FILES, ids=_source_id)
def test_no_source_file_cites_an_issue_number(path):
    src = _read(path)
    prose = [(tok.start[0], tok.string) for tok in _comment_tokens(src)]
    prose += [(line, text) for _, line, _, text in _docstrings(src)]
    prose += _string_literals(src)
    hits = [
        f"{_source_id(path)}:{line}: {ref}"
        for line, text in sorted(prose)
        for ref in _ISSUE_REF.findall(text)
    ]
    assert not hits, (
        "issue and PR numbers:\n  " + "\n  ".join(hits) + "\n"
        "A number names nothing a reader of the wheel or the test can open. "
        "State the rule; git blame is the archive."
    )


@pytest.mark.parametrize("path", EM_DASH_FILES, ids=_source_id)
def test_no_file_has_an_em_dash(path):
    hits = [
        f"{_source_id(path)}:{n}"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if EM_DASH in line
    ]
    assert not hits, (
        "em dashes:\n  " + "\n  ".join(hits) + "\n"
        "Use a comma, a colon, parentheses or a full stop instead."
    )
