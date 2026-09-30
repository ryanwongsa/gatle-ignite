"""`gatle-ignite init`. Template files end in `-tpl` so ruff and pytest skip them."""

import importlib.util
import keyword
from importlib.resources import files
from pathlib import Path

from gatle_ignite.dispatch import ConfigError

PLACEHOLDER = "{{project_name}}"

# Not `.gitignore`: git and hatchling would both obey a literal one inside the package.
GITIGNORE_TPL = "gitignore-tpl"


def _validate(name):
    """Reject an unusable name now: later, the failure would not mention init."""
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ConfigError(
            f"{name!r} is not a usable project name: it becomes a directory Python imports.\n"
            "  Use a Python identifier that is not a keyword, e.g. 'my_project'."
        )
    if name == "gatle_ignite":
        raise ConfigError(
            "a project cannot be called 'gatle_ignite': it would shadow the framework it "
            "imports.\n  Pick another name."
        )
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    # A namespace package (origin None) is allowed: init's own output is one, and --force must run.
    if spec is not None and spec.origin is not None:
        # The project root goes on sys.path, so a taken name shadows that module everywhere.
        raise ConfigError(
            f"{name!r} is already importable here ({spec.origin}), so your project and it "
            "would shadow each other. Pick a name that is not taken."
        )


def _walk(node, prefix=Path()):
    for child in sorted(node.iterdir(), key=lambda c: c.name):
        rel = prefix / child.name
        if child.is_dir():
            if child.name != "__pycache__":
                yield from _walk(child, rel)
        else:
            yield rel, child


def _target(rel, name):
    text = str(rel).replace(PLACEHOLDER, name)
    if Path(text).name == GITIGNORE_TPL:
        return (
            str(Path(text).parent / ".gitignore") if Path(text).parent != Path() else ".gitignore"
        )
    return text[: -len("-tpl")] if text.endswith("-tpl") else text


def init_project(name, directory=".", force=False):
    _validate(name)

    root = Path(directory).expanduser().resolve() / name
    source = files("gatle_ignite").joinpath("_template")

    planned = [(root / _target(rel, name), node) for rel, node in _walk(source)]

    existing = [path for path, _ in planned if path.exists()]
    if existing and not force:
        listing = "\n  ".join(str(p) for p in existing)
        raise ConfigError(
            f"these files already exist:\n  {listing}\n"
            "Pass --force to overwrite them, or pick another name."
        )

    for dest, node in planned:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Overwrite, never delete: --force may be run inside a real project.
        dest.write_text(node.read_text(encoding="utf-8").replace(PLACEHOLDER, name), "utf-8")

    _report(root, name, [path for path, _ in planned])
    return 0


def _report(root, name, written):
    """Print the tree that was actually written, rather than one typed out here to rot."""
    print(f"created {root}\n")
    for path in sorted(written):
        print(f"  {path.relative_to(root.parent)}")
    print(
        f"\nIt trains as-is, on synthetic data. Check the wiring before you change anything:\n"
        f"\n"
        f"  cd {name}\n"
        f"  gatle-ignite config --config=configs/{name}_v0.py   # validate, no GPU\n"
        f"  gatle-ignite train  --config=configs/{name}_v0.py   # CPU, seconds\n"
        f"\n"
        f"Then read README.md: it says where a loss, a metric or a sampler goes, and which\n"
        f"builtins mean you do not have to write one.\n"
    )
