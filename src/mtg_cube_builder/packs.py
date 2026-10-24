"""Generates booster packs from a card list using a JSON config.

Config keys:
  seed                   Int or null. Fixes the shuffle for reproducible boxes.
  packs                  Number of packs to generate.
  cards                  Path to the Scryfall card list, relative to the
                         project root.
  output                 Where to write the generated packs (JSON).
  allowDuplicatesInPack  Default false.
  maxCopiesPerBox        Most times one card may appear across all packs.
                         Null means unlimited.
  exclude                Global filter; matching cards never appear in a pack.
  slots                  Slots filled in order for every pack:
    name           Label for the slot.
    count          Cards drawn for the slot.
    filter         Which cards qualify.
    rarityWeights  Optional {rarity: weight}; otherwise every card is equally
                   likely.

Filter keys (a card must satisfy every key that is present):
  rarity        List of rarities.
  typeIncludes  The type line contains any of these strings.
  typeExcludes  The type line contains none of these strings.
  colors        List of W/U/B/R/G, "C" (colorless) or "M" (multicolor); any
                match qualifies.
  cmcMin, cmcMax
  names         List of exact card names.
"""

import collections
import json
import logging
import pathlib
import random
from typing import Any

from mtg_cube_builder import errors
from mtg_cube_builder import paths as paths_lib

_logger = logging.getLogger(__name__)
Card = dict[str, Any]


def _card_colors(card: Card) -> list[str]:
    """Returns the color letters of a card, looking at faces if needed."""
    colors = card.get("colors")
    if colors is None:
        colors = sorted(
            {
                c
                for face in card.get("card_faces", [])
                for c in face.get("colors", [])
            }
        )
    return colors


def matches(card: Card, card_filter: dict[str, Any] | None) -> bool:
    """Returns whether a card satisfies every key of a filter.

    Args:
      card: A Scryfall card object.
      card_filter: A filter as described in the module docstring. An empty or
        missing filter matches every card.
    """
    if not card_filter:
        return True
    if "rarity" in card_filter and card["rarity"] not in card_filter["rarity"]:
        return False
    type_line = card.get("type_line", "")
    if "typeIncludes" in card_filter and not any(
        t in type_line for t in card_filter["typeIncludes"]
    ):
        return False
    if "typeExcludes" in card_filter and any(
        t in type_line for t in card_filter["typeExcludes"]
    ):
        return False
    if "colors" in card_filter:
        colors = _card_colors(card)
        tags = set(colors)
        if not colors:
            tags.add("C")
        if len(colors) > 1:
            tags.add("M")
        if not tags & set(card_filter["colors"]):
            return False
    if "cmcMin" in card_filter and card.get("cmc", 0) < card_filter["cmcMin"]:
        return False
    if "cmcMax" in card_filter and card.get("cmc", 0) > card_filter["cmcMax"]:
        return False
    return not (
        "names" in card_filter and card["name"] not in card_filter["names"]
    )


def _draw(
    rng: random.Random,
    pool: list[Card],
    weights: dict[str, float] | None,
    in_pack: set[str],
    copies: collections.Counter,
    allow_duplicates: bool,
    max_copies: int | None,
) -> Card | None:
    """Picks one card from a pool, honoring the duplicate limits.

    Returns:
      The card, or None if no card in the pool is allowed.
    """
    candidates = [
        c
        for c in pool
        if (allow_duplicates or c["id"] not in in_pack)
        and (max_copies is None or copies[c["id"]] < max_copies)
    ]
    if not candidates:
        return None
    if weights:
        card_weights = [weights.get(c["rarity"], 0) for c in candidates]
        if sum(card_weights) > 0:
            return rng.choices(candidates, card_weights)[0]
    return rng.choice(candidates)


def _summarize(card: Card) -> dict[str, Any]:
    """Returns the fields of a card that are written to the pack file."""
    return {
        "name": card["name"],
        "rarity": card["rarity"],
        "type_line": card.get("type_line"),
        "mana_cost": card.get("mana_cost"),
        "id": card["id"],
    }


def generate(
    config: dict[str, Any], cards: list[Card], rng: random.Random
) -> list[dict[str, Any]]:
    """Generates the packs of a box.

    Args:
      config: The pack configuration.
      cards: The cards to draw from.
      rng: Source of randomness.

    Returns:
      One {"pack": number, "cards": [...]} dict per pack.

    Raises:
      errors.MtgCubeBuilderError: If a slot runs out of eligible cards.
    """
    exclude = config.get("exclude")
    if exclude:
        cards = [c for c in cards if not matches(c, exclude)]
    allow_duplicates = config.get("allowDuplicatesInPack", False)
    max_copies = config.get("maxCopiesPerBox")
    slots = config["slots"]
    pools = [
        [c for c in cards if matches(c, slot.get("filter"))] for slot in slots
    ]

    copies: collections.Counter = collections.Counter()
    packs = []
    for number in range(1, config["packs"] + 1):
        in_pack: set[str] = set()
        pack = []
        for slot, pool in zip(slots, pools, strict=True):
            for _ in range(slot["count"]):
                card = _draw(
                    rng,
                    pool,
                    slot.get("rarityWeights"),
                    in_pack,
                    copies,
                    allow_duplicates,
                    max_copies,
                )
                if card is None:
                    raise errors.MtgCubeBuilderError(
                        f"Pack {number}: slot '{slot['name']}' ran out of "
                        f"eligible cards (pool of {len(pool)}). Loosen the "
                        "filter, raise maxCopiesPerBox, or reduce "
                        "packs/count."
                    )
                in_pack.add(card["id"])
                copies[card["id"]] += 1
                pack.append({"slot": slot["name"], **_summarize(card)})
        packs.append({"pack": number, "cards": pack})
    return packs


def write_packs(
    paths: paths_lib.Paths,
    config_path: pathlib.Path | str,
    seed: int | None = None,
    out: pathlib.Path | str | None = None,
    text: bool = False,
) -> tuple[pathlib.Path, list[str]]:
    """Reads a config, generates the packs and writes them.

    Args:
      paths: The project paths.
      config_path: The pack config file.
      seed: Overrides the config's seed.
      out: Overrides the config's output path.
      text: Also return the packs as plain text lines.

    Returns:
      The path written, and the packs as text lines (empty unless `text`).
    """
    config = json.loads(pathlib.Path(config_path).read_text())
    if "cards" not in config:
        raise errors.MtgCubeBuilderError(
            f'{config_path} must name the card list in "cards", for example '
            '"output/set-fra.json".'
        )
    cards = json.loads((paths.root / config["cards"]).read_text())

    if seed is None:
        seed = config.get("seed")
    if seed is None:
        seed = random.SystemRandom().randrange(2**32)
    packs = generate(config, cards, random.Random(seed))

    out_path = pathlib.Path(out or config.get("output", "output/packs.json"))
    if not out_path.is_absolute():
        out_path = paths.root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"seed": seed, "packs": packs}, indent=2, ensure_ascii=False)
    )

    _logger.info(
        "Wrote %d packs (seed %d) to %s", len(packs), seed, paths.rel(out_path)
    )
    lines = []
    if text:
        for pack in packs:
            lines.append(f"\nPack {pack['pack']}")
            lines.extend(
                f"  [{c['rarity'][0].upper()}] {c['name']}"
                for c in pack["cards"]
            )
    return out_path, lines
