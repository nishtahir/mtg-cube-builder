import numpy as np

from mtg_cube_builder import cardinfo
from mtg_cube_builder import cluster
from mtg_cube_builder import embeddings
from mtg_cube_builder import paths as paths_lib


def test_attrs_type_category_and_colors():
    card = {
        "name": "Dragon",
        "type_line": "Legendary Artifact Creature — Dragon",
        "cmc": 5.0,
        "colors": ["R", "B"],
        "rarity": "mythic",
    }
    a = cardinfo.attrs(card)
    assert a["cat"] == "Creature"  # creature outranks artifact
    assert a["colors"] == "BR"  # sorted
    assert a["cmc"] == 5


def test_attrs_uses_front_face_for_multiface_cards():
    card = {
        "name": "A // B",
        "type_line": "Instant // Sorcery",
        "cmc": 3.0,
        "rarity": "rare",
        "card_faces": [
            {"type_line": "Instant", "colors": ["U"]},
            {"type_line": "Sorcery", "colors": ["R"]},
        ],
    }
    a = cardinfo.attrs(card)
    assert a["cat"] == "Instant" and a["colors"] == "U"


def test_color_cost_and_rank():
    assert cardinfo.color_cost("W", "W") == 0
    assert cardinfo.color_cost("WU", "U") == 0.5
    assert cardinfo.color_cost("W", "R") == 1
    assert cardinfo.color_cost("", "") == 0
    assert (
        cardinfo.rank("common")
        < cardinfo.rank("uncommon")
        < cardinfo.rank("rare")
        < cardinfo.rank("mythic")
    )
    assert cardinfo.rank("special") == cardinfo.rank("rare")


def test_card_text_scryfall_format():
    bird = {
        "name": "Birds of Paradise",
        "mana_cost": "{G}",
        "type_line": "Creature — Bird",
        "oracle_text": "Flying",
        "power": "0",
        "toughness": "1",
    }
    assert (
        embeddings.card_text(bird)
        == "Birds of Paradise {G}\nCreature — Bird\nFlying\n0/1"
    )
    split = {
        "card_faces": [
            {"name": "Armed", "mana_cost": "{1}{R}", "type_line": "Sorcery"},
            {
                "name": "Dangerous",
                "mana_cost": "{3}{G}",
                "type_line": "Sorcery",
            },
        ]
    }
    assert (
        embeddings.card_text(split).split("\n----\n")[1].startswith("Dangerous")
    )


def test_spherical_kmeans_is_deterministic_and_finds_clusters():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(20, 8)) + 5
    b = rng.normal(size=(20, 8)) - 5
    points = np.vstack([a, b]).astype(np.float64)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    c1, s1 = cluster.spherical_kmeans(points, 2, np.random.default_rng(0))
    c2, s2 = cluster.spherical_kmeans(points, 2, np.random.default_rng(0))
    assert np.allclose(c1, c2) and sorted(s1) == [20, 20]


def test_topk_mean_ignores_masked_entries():
    matrix = np.array([[0.9, 0.5, -np.inf], [-np.inf, -np.inf, -np.inf]])
    mean, n = cluster.topk_mean(matrix, 2)
    assert np.isclose(mean[0], 0.7) and n.tolist() == [2, 0] and mean[1] == 0


def test_paths_layout(tmp_path):
    p = paths_lib.Paths(tmp_path)
    assert p.pool == tmp_path.resolve() / "data" / "cards.jsonl"
    assert p.set_file("eld").parent == p.sets_dir
    assert p.rel(p.pool) == "data/cards.jsonl"
