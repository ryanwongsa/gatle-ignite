"""The gatle-ignite command line: argument parsing and command dispatch."""

import argparse
import os
import sys

from gatle_ignite import __version__
from gatle_ignite.config.loader import apply_overrides, load_config_from_file
from gatle_ignite.config.validate import check_dispatch, validate_config
from gatle_ignite.dispatch import ConfigError

_DESCRIPTION = """The gatle-ignite command line.

    gatle-ignite init   my_project                      # write a new project, then cd into it
    gatle-ignite train  --config=path/to/config.py [--override k=v ...]
    gatle-ignite eval   --config=path/to/config.py [--ckpt best|latest]
    gatle-ignite config --config=path/to/config.py     # resolve + validate, print, exit

`python -m gatle_ignite <cmd>` is equivalent, which is the form to use under SLURM.
"""


def _add_common(parser):
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a .py defining get_config(). Comma-separated paths run in parallel.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing config field. Repeatable.",
    )


def build_parser():
    # Raw, or argparse reflows _DESCRIPTION's examples into one paragraph, in --help and the docs.
    parser = argparse.ArgumentParser(
        prog="gatle-ignite",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"gatle-ignite {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # No --task flag: every task has the same layout, and the worked tasks live in examples/.
    initialise = sub.add_parser("init", help="Write a new project skeleton.")
    initialise.add_argument(
        "name",
        help="Project name. Must be a valid Python identifier: it becomes a directory "
        "that gets imported, and the prefix is dropped from your config's dotted paths.",
    )
    initialise.add_argument(
        "--dir",
        default=".",
        dest="directory",
        help="Where to create it. Default: the current directory.",
    )
    initialise.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files. Never deletes anything.",
    )

    _add_common(sub.add_parser("train", help="Train a config."))
    _add_common(sub.add_parser("config", help="Resolve and validate a config, then print it."))

    evaluate = sub.add_parser("eval", help="Load a checkpoint and evaluate.")
    _add_common(evaluate)
    evaluate.add_argument(
        "--ckpt",
        default="best",
        choices=("best", "latest"),
        help="Which checkpoint to load: the highest-scoring valid_best_result* (best), "
        "or the newest latest_epoch* (latest).",
    )
    return parser


def _ensure_cwd_importable():
    """A console script leaves the cwd off sys.path, so a config's own paths would not resolve."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _prepare(args, config_path):
    _ensure_cwd_importable()
    cfg = load_config_from_file(config_path)
    apply_overrides(cfg, args.override)
    validate_config(cfg)
    return cfg


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        # init has no --config, so it dispatches before args.config is read.
        if args.command == "init":
            from gatle_ignite.cli.init import init_project

            return init_project(args.name, args.directory, args.force)

        paths = [p.strip() for p in args.config.split(",") if p.strip()]

        if args.command == "config":
            for path in paths:
                print(f"===== {path} =====")
                cfg = _prepare(args, path)
                print(cfg)
                check_dispatch(cfg)
                print("\nall dotted paths resolve.")
            return 0

        if args.command == "eval":
            from gatle_ignite.cli.launch import launch

            # Scoring only paths[0] would report a number for a run the user did not ask for.
            if len(paths) > 1:
                raise ConfigError(
                    f"eval takes one --config, got {len(paths)}: {', '.join(paths)}\n"
                    f"  The comma-separated form is train-only. Run eval once per config."
                )

            cfg = _prepare(args, paths[0])
            cfg.run = True
            cfg.load_from_ckpt = args.ckpt
            launch(cfg)
            return 0

        if len(paths) > 1:
            from gatle_ignite.cli.launch import launch_many

            extra = [f"--override={o}" for o in args.override]
            return launch_many(paths, extra)

        from gatle_ignite.cli.launch import launch

        launch(_prepare(args, paths[0]))
        return 0

    except ConfigError as e:
        # A user error: no traceback through framework internals, just the line to fix.
        print(f"\nconfig error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
