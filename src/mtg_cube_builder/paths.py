"""Where everything lives.

Small, versionable inputs and results sit at the project root; bulk downloads
go in data/ and generated results in output/ (both gitignored).

  data/cards.jsonl   The card pool: every card, deduped, as one Scryfall
                     object per line. Input.
  data/sets/*.jsonl  Every printing in a set, one Scryfall card object per
                     line, named after the set code. Only the template set
                     is read. Input.
  data/embeddings/   Embeddings of the pool (embed).
  config/build.json  Build settings (optional).
  output/            Built sets, reports and packs.
"""

import dataclasses
import pathlib

from mtg_cube_builder import errors


@dataclasses.dataclass(frozen=True)
class Paths:
    """The files of one project, all derived from its root directory.

    Attributes:
      root: The project directory. Resolved to an absolute path.
    """

    root: pathlib.Path = dataclasses.field(default_factory=pathlib.Path.cwd)

    def __post_init__(self) -> None:
        """Resolves the root to an absolute path."""
        object.__setattr__(self, "root", pathlib.Path(self.root).resolve())

    @property
    def data(self) -> pathlib.Path:
        """Directory for bulk downloads."""
        return self.root / "data"

    @property
    def sets_dir(self) -> pathlib.Path:
        """Directory with one JSONL file per set."""
        return self.data / "sets"

    @property
    def embeddings_dir(self) -> pathlib.Path:
        """Directory with the card embeddings."""
        return self.data / "embeddings"

    @property
    def pool(self) -> pathlib.Path:
        """The card pool: every card from every set, as rows of text."""
        return self.data / "cards.jsonl"

    @property
    def build_config(self) -> pathlib.Path:
        """The optional file of build settings."""
        return self.root / "config" / "build.json"

    @property
    def output(self) -> pathlib.Path:
        """Directory for generated results."""
        return self.root / "output"

    def set_file(self, code: str) -> pathlib.Path:
        """Returns the JSONL file holding the printings of a set."""
        return self.sets_dir / f"{code}.jsonl"

    def built_csv(self, template: str) -> pathlib.Path:
        """Returns the per-slot CSV of a built set."""
        return self.output / f"set-{template}.csv"

    def built_json(self, template: str) -> pathlib.Path:
        """Returns the card list of a built set."""
        return self.output / f"set-{template}.json"

    def report_html(self, template: str) -> pathlib.Path:
        """Returns the HTML report of a built set."""
        return self.output / f"report-{template}.html"

    def rel(self, path: pathlib.Path) -> str:
        """Returns a path relative to the project root when possible."""
        try:
            return str(pathlib.Path(path).resolve().relative_to(self.root))
        except ValueError:
            return str(path)

    def require(self, path: pathlib.Path, what: str, hint: str = "") -> None:
        """Checks that an input file exists.

        Args:
          path: The file that must exist.
          what: What the file is, for the error message.
          hint: How to get the file, appended to the error message.

        Raises:
          errors.MtgCubeBuilderError: If the file is missing.
        """
        if not path.exists():
            message = f"Missing {what}: {self.rel(path)}."
            raise errors.MtgCubeBuilderError(f"{message} {hint}".strip())
