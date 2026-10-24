"""The list of cards to build a set from: a text file of card names.

The file has one card name per line. Blank lines and lines starting with `#`
are ignored. Names are matched against the card pool, ignoring case and curly
quotes; a multi-faced card can be named by its full name ("Fire // Ice") or by
either face ("Fire").
"""

import difflib
import pathlib
from typing import Any

from mtg_cube_builder import errors
from mtg_cube_builder import paths as paths_lib

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_MAX_REPORTED = 20  # Unmatched names listed in an error.


def normalize(name: str) -> str:
    """Returns a card name in the form used to compare names."""
    return " ".join(name.translate(_QUOTES).split()).casefold()


def read_names(path: pathlib.Path) -> list[str]:
    """Reads the card names in a file.

    Args:
      path: A text file with one card name per line.

    Returns:
      The names in file order. Blank lines, `#` comments and repeated names
      (ignoring case) are dropped.
    """
    names: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                names.setdefault(normalize(name), name)
    return list(names.values())


def _suggestion(name: str, known: list[str]) -> str:
    """Returns a hint naming the closest known card, or nothing."""
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.8)
    return f" (did you mean {close[0]!r}?)" if close else ""


def resolve(
    names: list[str], cards: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Finds the pool card for each name.

    Args:
      names: Card names.
      cards: The pool's cards.

    Returns:
      The matching cards in the order of `names`, each card once.

    Raises:
      errors.MtgCubeBuilderError: If any name matches no card. The message lists
        them, with the closest card name where there is a good guess.
    """
    by_name = {normalize(c["name"]): c for c in cards}
    by_face: dict[str, dict[str, Any]] = {}
    for card in cards:
        for face in card.get("card_faces", []):
            by_face.setdefault(normalize(face["name"]), card)
    found: dict[str, dict[str, Any]] = {}
    unmatched = []
    for name in names:
        card = by_name.get(normalize(name)) or by_face.get(normalize(name))
        if card is None:
            unmatched.append(name)
        else:
            found.setdefault(card["oracle_id"], card)
    if unmatched:
        known = [c["name"] for c in cards]
        lines = [
            f"  {name}{_suggestion(name, known)}"
            for name in unmatched[:_MAX_REPORTED]
        ]
        more = len(unmatched) - _MAX_REPORTED
        if more > 0:
            lines.append(f"  ... and {more} more")
        raise errors.MtgCubeBuilderError(
            f"{len(unmatched)} card name(s) are not in the card pool:\n"
            + "\n".join(lines)
        )
    return list(found.values())


def load(
    paths: paths_lib.Paths,
    cards_file: pathlib.Path | str,
    pool_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reads a card list file and finds its cards in the pool.

    Args:
      paths: The project paths.
      cards_file: The text file of card names.
      pool_cards: The pool's cards.

    Returns:
      The pool cards named in the file.

    Raises:
      errors.MtgCubeBuilderError: If the file is missing, or a name matches no
        card.
    """
    path = pathlib.Path(cards_file)
    paths.require(path, "card list", "It should have one card name per line.")
    return resolve(read_names(path), pool_cards)
