"""Helpers for reading Scryfall card objects."""

import dataclasses
from typing import Any, Self

# Layouts that are not real cards: tokens, emblems, art cards and so on.
SKIP_LAYOUTS = frozenset(
    {
        "token",
        "double_faced_token",
        "emblem",
        "art_series",
        "vanguard",
        "scheme",
        "planar",
    }
)

_RARITY_RANK = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}
_DEFAULT_RARITY_RANK = 2  # Special and bonus cards count as rare.
# Checked in order, so an artifact creature is a creature.
_TYPE_CATEGORIES = (
    "Land",
    "Creature",
    "Planeswalker",
    "Battle",
    "Instant",
    "Sorcery",
    "Artifact",
    "Enchantment",
)
_SHARED_COLOR_COST = 0.5


@dataclasses.dataclass(frozen=True)
class Card:
    """The attributes a slot or a candidate is matched on.

    Attributes:
      oracle_id: Scryfall's identifier for the card across printings.
      name: The card name.
      colors: Color letters in sorted order, such as "BR". Empty if colorless.
      cmc: Mana value.
      cat: The main card type: one of the type categories, or "Other".
      rarity: The rarity of this printing.
    """

    oracle_id: str
    name: str
    colors: str
    cmc: int
    cat: str
    rarity: str

    @classmethod
    def from_scryfall(cls, card: dict[str, Any], oracle_id: str) -> Self:
        """Builds a Card from a Scryfall object."""
        facts = attrs(card)
        return cls(
            oracle_id=oracle_id,
            name=facts["name"],
            colors=facts["colors"],
            cmc=facts["cmc"],
            cat=facts["cat"],
            rarity=facts["rarity"],
        )


def face0(card: dict[str, Any]) -> dict[str, Any]:
    """Returns the front face of a card (the card itself if it has one)."""
    return (card.get("card_faces") or [card])[0]


def oracle_id_of(card: dict[str, Any]) -> str | None:
    """Returns the oracle id of a card, looking at its first face if needed."""
    return card.get("oracle_id") or (card.get("card_faces") or [{}])[0].get(
        "oracle_id"
    )


def is_basic_land(card: dict[str, Any]) -> bool:
    """Returns whether the card is a basic land."""
    return "Basic Land" in card.get("type_line", "")


def category(type_line: str) -> str:
    """Returns the main card type from a type line, or "Other".

    Args:
      type_line: A type line, such as "Legendary Artifact Creature — Robot".

    Returns:
      The first of Land, Creature, Planeswalker, Battle, Instant, Sorcery,
      Artifact and Enchantment that the type line contains. An artifact
      creature is therefore a creature.
    """
    main_types = type_line.split("—")[0]
    for name in _TYPE_CATEGORIES:
        if name in main_types:
            return name
    return "Other"


def attrs(card: dict[str, Any]) -> dict[str, Any]:
    """Returns the attributes of a card that slots are matched on.

    Args:
      card: A Scryfall card object.

    Returns:
      A dict with name, colors, cmc, cat, type_line and rarity. Multi-faced
      cards are described by their front face.
    """
    front = face0(card)
    colors = card.get("colors")
    if colors is None:
        colors = front.get("colors", [])
    type_line = front.get("type_line") or card.get("type_line", "")
    category_name = category(type_line)
    return {
        "name": card["name"],
        "colors": "".join(sorted(colors)),
        "cmc": int(card.get("cmc", 0)),
        "cat": category_name,
        "type_line": card.get("type_line", ""),
        "rarity": card["rarity"],
    }


def rank(rarity: str) -> int:
    """Returns a rarity as a number so rarities can be compared."""
    return _RARITY_RANK.get(rarity, _DEFAULT_RARITY_RANK)


def color_cost(a: str, b: str) -> float:
    """Returns how badly two color sets differ.

    Args:
      a: Color letters, such as "WU".
      b: Color letters.

    Returns:
      0 for identical sets, 0.5 if they share a color, and 1 otherwise.
    """
    if a == b:
        return 0.0
    return _SHARED_COLOR_COST if set(a) & set(b) else 1.0
