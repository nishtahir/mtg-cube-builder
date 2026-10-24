"""Settings for building a set, and reading them from a JSON file.

Every setting has a default. A config file (`config/build.json`) overrides the
defaults, and command-line options override the file. `mtg-cube-builder config`
prints the effective settings as JSON, ready to save as a config file.
"""

from collections.abc import Mapping
import dataclasses
import json
import pathlib
from typing import Any

from mtg_cube_builder import errors


def _setting(default: Any, help_text: str) -> Any:
    """Declares a setting with its default and a description."""
    return dataclasses.field(default=default, metadata={"help": help_text})


@dataclasses.dataclass
class BuildConfig:
    """Settings for one build.

    Attributes:
      template: Code of the real set whose cards define the slots.
      max_cost: Pairs costlier than this become gaps.
      w_color: Cost of a color mismatch.
      w_type: Cost of a card type mismatch.
      w_cmc: Cost per mana value of difference, capped at three.
      w_rarity: Cost per rarity step of difference.
      w_coh: Reward per unit of cohesion above the list median.
      w_sim: Reward for similarity to the template card.
      w_fill_sim: Retrieval weight on similarity to the template card.
      themes: Number of clusters of the list. 1 means global cohesion.
      theme_alpha: Theme quota is proportional to size ** alpha. 1 is
        proportional and 0 is equal.
      price_step: How fast an over- or under-filled theme's price moves.
      w_balance: Retrieval bonus for under-filled themes.
      max_iter: Maximum rounds of re-assignment.
      knn: Neighbors averaged for cohesion.
    """

    template: str = _setting(
        "fra",
        "Code of the real set whose cards define the slots. Its file must be "
        "in data/sets/.",
    )
    max_cost: float = _setting(
        1.5,
        "A list card costlier than this for a slot is not used and the slot "
        "is filled from the pool instead. Lower uses fewer of your cards but "
        "fits them better.",
    )
    w_color: float = _setting(2.0, "Cost of a color mismatch.")
    w_type: float = _setting(1.5, "Cost of a card type mismatch.")
    w_cmc: float = _setting(
        0.5, "Cost per mana value of difference, capped at three."
    )
    w_rarity: float = _setting(0.4, "Cost per rarity step of difference.")
    w_coh: float = _setting(
        5.0,
        "Reward per unit of cohesion (closeness to the other chosen cards of "
        "the same theme) above the list median.",
    )
    w_sim: float = _setting(
        0.0, "Reward for similarity to the template card. Off by default."
    )
    w_fill_sim: float = _setting(
        0.0,
        "Weight of similarity to the template card when filling gaps from "
        "the pool. Off by default.",
    )
    themes: int = _setting(
        8, "Number of themes to cluster the list into. 1 means global cohesion."
    )
    theme_alpha: float = _setting(
        0.5,
        "A theme's share of the set is proportional to its size to this "
        "power. 1 is proportional and 0 is equal.",
    )
    price_step: float = _setting(
        0.5, "How fast an over- or under-filled theme's price moves."
    )
    w_balance: float = _setting(
        0.3, "Bonus for filling gaps with cards of under-filled themes."
    )
    max_iter: int = _setting(30, "Maximum rounds of re-assignment.")
    knn: int = _setting(
        10, "Number of nearest neighbors averaged to measure cohesion."
    )


def setting_names() -> list[str]:
    """Returns the names of all settings, in declaration order."""
    return [f.name for f in dataclasses.fields(BuildConfig)]


def from_dict(values: Mapping[str, Any], source: str = "config") -> BuildConfig:
    """Builds a config from settings read from a file.

    Args:
      values: Setting names and values. Settings that are missing keep their
        default.
      source: Where the values came from, for error messages.

    Returns:
      The config.

    Raises:
      errors.MtgCubeBuilderError: If a name is not a setting, or a value has the
        wrong type. An integer is accepted for a decimal setting.
    """
    fields = {f.name: f for f in dataclasses.fields(BuildConfig)}
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise errors.MtgCubeBuilderError(
            f"Unknown setting(s) in {source}: {', '.join(unknown)}. "
            f"Valid settings: {', '.join(fields)}."
        )
    checked = {}
    for name, value in values.items():
        expected = fields[name].type
        if (
            expected is float
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            value = float(value)
        if type(value) is not expected:
            raise errors.MtgCubeBuilderError(
                f"Setting {name!r} in {source} must be a "
                f"{expected.__name__}, not {value!r}."
            )
        checked[name] = value
    return BuildConfig(**checked)


def load(path: pathlib.Path) -> BuildConfig:
    """Reads a config file.

    Args:
      path: A JSON file holding an object of settings. An empty file means
        no settings.

    Returns:
      The config.

    Raises:
      errors.MtgCubeBuilderError: If the file is not a JSON object, or holds an
        unknown setting or a value of the wrong type.
    """
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if not text.strip():
        return BuildConfig()  # An empty file sets nothing.
    try:
        values = json.loads(text)
    except json.JSONDecodeError as e:
        raise errors.MtgCubeBuilderError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(values, dict):
        raise errors.MtgCubeBuilderError(
            f"{path} must hold a JSON object of settings."
        )
    return from_dict(values, str(path))


def to_json(cfg: BuildConfig) -> str:
    """Renders a config as JSON, one setting per line."""
    return json.dumps(dataclasses.asdict(cfg), indent=2) + "\n"


def describe() -> list[tuple[str, Any, str]]:
    """Returns (name, default, description) for every setting."""
    return [
        (f.name, f.default, f.metadata["help"])
        for f in dataclasses.fields(BuildConfig)
    ]
