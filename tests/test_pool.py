import json

import pytest

from mtg_cube_builder import errors
from mtg_cube_builder import paths as paths_lib
from mtg_cube_builder import pool


def card(name, oracle_id, paper=True, **extra):
    return {
        "oracle_id": oracle_id,
        "name": name,
        "type_line": "Creature",
        "colors": ["G"],
        "cmc": 2.0,
        "rarity": "common",
        "rarities": ["common", "rare"],
        "paper": paper,
        **extra,
    }


def write_pool(root, cards):
    paths = paths_lib.Paths(root)
    paths.pool.parent.mkdir(parents=True, exist_ok=True)
    paths.pool.write_text("".join(json.dumps(c) + "\n" for c in cards))
    return paths


def test_read_cards_reads_one_card_per_line(tmp_path):
    paths = write_pool(tmp_path, [card("Bear", "o1"), card("Elk", "o2")])
    assert [c["name"] for c in pool.read_cards(paths)] == ["Bear", "Elk"]


def test_a_missing_pool_is_a_clear_error(tmp_path):
    with pytest.raises(errors.MtgCubeBuilderError, match="card pool"):
        pool.read_cards(paths_lib.Paths(tmp_path))


def test_fill_pool_leaves_out_unplayable_cards():
    cards = [
        card("Bear", "o1"),
        card("Forest", "o2", type_line="Basic Land — Forest"),
        card("Digital", "o3", paper=False),
    ]
    filled = pool.fill_pool(cards)
    assert [e["name"] for e in filled.values()] == ["Bear"]


def test_fill_pool_entries_carry_what_matching_needs():
    entry = pool.fill_pool([card("Bear", "o1", colors=["G", "B"])])["o1"]
    assert entry["colors"] == "BG"
    assert entry["cmc"] == 2
    assert entry["cat"] == "Creature"
    assert entry["rarities"] == ["common", "rare"]
    assert entry["card"]["name"] == "Bear"
