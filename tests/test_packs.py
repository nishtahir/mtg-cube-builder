import random

from mtg_cube_builder import packs

CARDS = [
    {
        "id": f"c{i}",
        "name": f"Card {i}",
        "rarity": r,
        "type_line": t,
        "cmc": i,
        "colors": c,
    }
    for i, (r, t, c) in enumerate(
        [
            ("common", "Creature", ["W"]),
            ("common", "Instant", ["U"]),
            ("common", "Land", []),
            ("uncommon", "Creature", ["R", "G"]),
            ("rare", "Sorcery", ["B"]),
            ("mythic", "Creature", ["B"]),
        ]
        * 5
    )
]


def test_filters():
    assert packs.matches(CARDS[0], {"rarity": ["common"], "colors": ["W"]})
    assert not packs.matches(CARDS[0], {"typeExcludes": ["Creature"]})
    assert packs.matches(CARDS[2], {"colors": ["C"]})  # colorless tag
    assert packs.matches(CARDS[3], {"colors": ["M"]})  # multicolor tag


def test_generate_respects_slots_and_has_no_duplicates_in_a_pack():
    config = {
        "packs": 3,
        "slots": [
            {"name": "Common", "count": 3, "filter": {"rarity": ["common"]}},
            {
                "name": "Rare",
                "count": 1,
                "filter": {"rarity": ["rare", "mythic"]},
                "rarityWeights": {"rare": 3, "mythic": 1},
            },
        ],
    }
    result = packs.generate(config, CARDS, random.Random(0))
    assert len(result) == 3
    for p in result:
        assert [c["slot"] for c in p["cards"]] == ["Common"] * 3 + ["Rare"]
        ids = [c["id"] for c in p["cards"]]
        assert len(set(ids)) == len(ids)
        assert all(c["rarity"] == "common" for c in p["cards"][:3])


def test_config_without_exclude_keeps_all_cards():
    config = {"packs": 1, "slots": [{"name": "Any", "count": 2, "filter": {}}]}
    assert len(packs.generate(config, CARDS, random.Random(0))[0]["cards"]) == 2
