"""Command line interface: `mtg-cube-builder <command>`.

embed        Embed every card: data/embeddings/.
build        Build a set from a list of card names (a text file):
             output/set-<template>.{csv,json}
             and a report.
config       Print the effective build settings as JSON.
report       Render the HTML report of a built set.
packs        Generate booster packs from a built set.
"""

import argparse
from collections.abc import Sequence
import dataclasses
import json
import logging
import pathlib
import sys

from mtg_cube_builder import __version__
from mtg_cube_builder import config as config_lib
from mtg_cube_builder import embeddings
from mtg_cube_builder import errors
from mtg_cube_builder import packs as packs_lib
from mtg_cube_builder import paths as paths_lib
from mtg_cube_builder import report as report_lib

_logger = logging.getLogger(__name__)


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Adds the option naming a settings file."""
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help="JSON file of settings (default: config/build.json if it exists)",
    )


def _add_setting_args(parser: argparse.ArgumentParser) -> None:
    """Adds one option per build setting, named after the setting.

    The parser must use `argument_default=argparse.SUPPRESS`, so that only the
    options actually given override the config file.
    """
    for field in dataclasses.fields(config_lib.BuildConfig):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            dest=field.name,
            type=field.type,
            help=f"{field.metadata['help']} (default: {field.default})",
        )


def _build_parser() -> argparse.ArgumentParser:
    """Returns the argument parser with all commands."""
    parser = argparse.ArgumentParser(
        prog="mtg-cube-builder",
        description="Turn a card list into a set shaped like a real expansion.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="project directory with data/ and "
        "output/ (default: current directory)",
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="command"
    )

    embed = commands.add_parser(
        "embed", help="embed every card (writes data/embeddings/)"
    )
    embed.add_argument(
        "--query",
        metavar="NAME",
        help="print the nearest neighbors of a card from the cached "
        "embeddings and exit",
    )
    embed.add_argument(
        "-k", type=int, default=10, help="neighbors to print with --query"
    )
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--max-length", type=int, default=512)
    embed.add_argument(
        "--device", help="torch device (default: mps if available, else cpu)"
    )

    build = commands.add_parser(
        "build",
        help="build a set from the card list",
        description="Build a set from the card list. Settings come from the "
        "defaults, then the config file, then the options below.",
        argument_default=argparse.SUPPRESS,
    )
    build.add_argument(
        "cards",
        type=pathlib.Path,
        help="text file of the cards to build the set from, one card name "
        "per line",
    )
    _add_config_arg(build)
    _add_setting_args(build)
    build.add_argument(
        "--no-report",
        action="store_true",
        default=False,
        help="do not write the HTML report",
    )

    config = commands.add_parser(
        "config",
        help="print the effective build settings as JSON",
        description="Print the build settings, with the config file applied "
        "to the defaults, as JSON.",
    )
    _add_config_arg(config)

    report = commands.add_parser(
        "report", help="render the HTML report for a built set"
    )
    _add_config_arg(report)
    report.add_argument(
        "--template",
        default=None,
        help="which built set to report on (default: the template setting)",
    )
    report.add_argument(
        "--input",
        type=pathlib.Path,
        help="JSON list of cards to render (default: the built set's JSON)",
    )
    report.add_argument("--out", type=pathlib.Path, help="output HTML path")
    report.add_argument("--title")

    packs = commands.add_parser(
        "packs", help="generate booster packs from a card list"
    )
    packs.add_argument(
        "config",
        nargs="?",
        help="pack config (default: config/pack-config.json)",
    )
    packs.add_argument("--seed", type=int, help="override the config seed")
    packs.add_argument("--out", help="override the config output path")
    packs.add_argument(
        "--text", action="store_true", help="also print the packs as plain text"
    )
    return parser


def _write_report(
    paths: paths_lib.Paths,
    cards: list[dict],
    out_path: pathlib.Path,
    title: str,
    source: str,
) -> None:
    """Writes the HTML report and says where."""
    path = report_lib.write_report(cards, out_path, title=title, source=source)
    _logger.info("Wrote %s", paths.rel(path))


def load_build_config(
    args: argparse.Namespace, paths: paths_lib.Paths
) -> config_lib.BuildConfig:
    """Resolves the build settings: defaults, then the file, then options.

    Args:
      args: The parsed arguments. Any attribute named after a setting and set
        to something other than None overrides the file.
      paths: The project paths.

    Returns:
      The settings.

    Raises:
      errors.MtgCubeBuilderError: If a config file named with --config is missing,
        or a config file is invalid.
    """
    explicit = getattr(args, "config", None)
    path = explicit or paths.build_config
    if explicit:
        paths.require(path, "config file")
    cfg = config_lib.load(path) if path.exists() else config_lib.BuildConfig()
    overrides = {
        name: value
        for name, value in vars(args).items()
        if name in config_lib.setting_names() and value is not None
    }
    return dataclasses.replace(cfg, **overrides)


def _run_build(args: argparse.Namespace, paths: paths_lib.Paths) -> int:
    """Runs the build command."""
    # Imported here because scikit-learn and scipy take a second to load and
    # the other commands do not need them.
    from mtg_cube_builder import builder  # pylint: disable=import-outside-toplevel

    cfg = load_build_config(args, paths)
    result = builder.build_set(paths, cfg, args.cards)
    if not getattr(args, "no_report", False):
        _write_report(
            paths,
            result.cards,
            paths.report_html(cfg.template),
            f"Built set ({cfg.template.upper()} template)",
            paths.built_json(cfg.template).name,
        )
    return 0


def _run_config(args: argparse.Namespace, paths: paths_lib.Paths) -> int:
    """Runs the config command."""
    print(config_lib.to_json(load_build_config(args, paths)), end="")
    return 0


def _run_report(args: argparse.Namespace, paths: paths_lib.Paths) -> int:
    """Runs the report command."""
    template = args.template or load_build_config(args, paths).template
    source = args.input or paths.built_json(template)
    cards = json.loads(pathlib.Path(source).read_text())
    _write_report(
        paths,
        cards,
        args.out or paths.report_html(template),
        args.title or f"Built set ({template.upper()} template)",
        pathlib.Path(source).name,
    )
    return 0


def _run_packs(args: argparse.Namespace, paths: paths_lib.Paths) -> int:
    """Runs the packs command."""
    config_path = args.config or paths.root / "config" / "pack-config.json"
    _, lines = packs_lib.write_packs(
        paths, config_path, seed=args.seed, out=args.out, text=args.text
    )
    for line in lines:
        print(line)
    return 0


def _run_embed(args: argparse.Namespace, paths: paths_lib.Paths) -> int:
    """Runs the embed command."""
    if args.query:
        name, neighbors = embeddings.nearest(paths, args.query, args.k)
        print(f"Nearest to {name}:")
        for similarity, neighbor in neighbors:
            print(f"  {similarity:.3f}  {neighbor}")
    else:
        embeddings.embed_all(
            paths, args.batch_size, args.max_length, args.device
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the command line interface.

    Args:
      argv: The arguments after the program name. Defaults to sys.argv.

    Returns:
      The process exit code: 0 on success and 1 on a failure the user can fix.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout
    )
    paths = paths_lib.Paths(args.root)
    try:
        if args.command == "embed":
            return _run_embed(args, paths)
        if args.command == "build":
            return _run_build(args, paths)
        if args.command == "config":
            return _run_config(args, paths)
        if args.command == "report":
            return _run_report(args, paths)
        return _run_packs(args, paths)
    except errors.MtgCubeBuilderError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
