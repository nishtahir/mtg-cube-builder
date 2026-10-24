"""The card pool: every card from every set, one Scryfall object each.

`data/cards.jsonl` is an input. It has one line per unique card (by oracle id):
a Scryfall card object reduced to the fields the package reads, plus two fields
computed across all of the card's printings:

  rarities: Every rarity the card has been printed at.
  paper: Whether it has a real paper printing, that is, one that is not
    digital-only and not in a joke, memorabilia, token or minigame set.

The pool is what gaps are filled from, what `embed` embeds, and where the names
in a card list are looked up. `experiments/build_pool.py` builds it from set
files.
"""

import json
from typing import Any

from mtg_cube_builder import cardinfo
from mtg_cube_builder import paths as paths_lib

Pool = dict[str, dict[str, Any]]


def read_cards(paths: paths_lib.Paths) -> list[dict[str, Any]]:
    """Reads the pool file.

    Args:
      paths: The project paths.

    Returns:
      One card object per line.

    Raises:
      errors.MtgCubeBuilderError: If the pool file is missing.
    """
    paths.require(
        paths.pool,
        "card pool",
        "It holds one card object per line; see experiments/build_pool.py.",
    )
    with open(paths.pool, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fill_pool(cards: list[dict[str, Any]]) -> Pool:
    """Returns the cards gaps can be filled from, keyed by oracle id.

    Args:
      cards: The pool's cards, as read by `read_cards`.

    Returns:
      For each card, its name, colors, cmc, cat, type_line and rarity (see
      `cardinfo.attrs`), `rarities` (every rarity it was printed at) and `card`
      (the full pool object). Cards without a paper printing and basic lands
      are left out.
    """
    pool: Pool = {}
    for card in cards:
        if not card.get("paper") or cardinfo.is_basic_land(card):
            continue
        pool[card["oracle_id"]] = {
            **cardinfo.attrs(card),
            "rarities": card["rarities"],
            "card": card,
        }
    return pool
