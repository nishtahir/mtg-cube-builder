"""Tests for experiments/build_pool.py, which produces data/cards.jsonl."""

import importlib.util
import json
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).parent.parent / "experiments" / "build_pool.py"
_spec = importlib.util.spec_from_file_location("build_pool", _SCRIPT)
build_pool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_pool)


def card(name, oracle_id, rarity="common", set_code="aaa", **extra):
    return {
        "id": f"{oracle_id}-{set_code}",
        "oracle_id": oracle_id,
        "name": name,
        "layout": "normal",
        "set": set_code,
        "rarity": rarity,
        "released_at": "2020-01-01",
        "digital": False,
        "type_line": "Creature",
        "cmc": 2.0,
        "image_uris": {"normal": "n", "small": "s", "png": "p"},
        "prices": {"usd": "1.00"},
        **extra,
    }


def build(tmp_path, sets):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir()
    for code, cards in sets.items():
        (sets_dir / f"{code}.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in cards)
        )
    sets_json = tmp_path / "sets.json"
    sets_json.write_text(
        json.dumps(
            [
                {"code": "aaa", "set_type": "expansion"},
                {"code": "bbb", "set_type": "masters"},
                {"code": "jok", "set_type": "funny"},
                {"code": "prm", "set_type": "promo"},
            ]
        )
    )
    out = tmp_path / "cards.jsonl"
    build_pool.build_pool([sets_dir], sets_json, out)
    return {json.loads(line)["name"]: json.loads(line) for line in open(out)}


def test_dedupes_and_collects_rarities(tmp_path):
    cards = build(
        tmp_path,
        {
            "aaa": [card("Bear", "o1", "common", "aaa")],
            "bbb": [
                card("Bear", "o1", "rare", "bbb", released_at="2021-01-01")
            ],
        },
    )
    assert list(cards) == ["Bear"]
    assert cards["Bear"]["rarities"] == ["common", "rare"]
    assert cards["Bear"]["set"] == "bbb"  # The newest paper printing is kept.


def test_slims_the_card_object(tmp_path):
    bear = build(tmp_path, {"aaa": [card("Bear", "o1")]})["Bear"]
    assert "prices" not in bear
    assert bear["image_uris"] == {"normal": "n"}


def test_paper_flag_ignores_digital_and_joke_sets(tmp_path):
    cards = build(
        tmp_path,
        {
            "aaa": [card("Digital", "o1", digital=True), card("Paper", "o2")],
            "jok": [card("Joke", "o3", set_code="jok")],
        },
    )
    assert not cards["Digital"]["paper"]
    assert cards["Paper"]["paper"]
    assert not cards["Joke"]["paper"]


def test_tokens_are_skipped(tmp_path):
    cards = build(
        tmp_path,
        {"aaa": [card("Goblin", "o1", layout="token"), card("Bear", "o2")]},
    )
    assert list(cards) == ["Bear"]


def test_multifaced_card_gets_its_oracle_id_from_a_face(tmp_path):
    two_faced = card("Front // Back", None, layout="transform")
    del two_faced["oracle_id"]
    two_faced["card_faces"] = [
        {"name": "Front", "oracle_id": "o9", "image_uris": {"normal": "f"}},
        {"name": "Back", "oracle_id": "o9", "image_uris": {"normal": "b"}},
    ]
    cards = build(tmp_path, {"aaa": [two_faced]})
    assert cards["Front // Back"]["oracle_id"] == "o9"


def test_a_missing_set_directory_stops_before_writing(tmp_path):
    out = tmp_path / "cards.jsonl"
    with pytest.raises(SystemExit, match="Missing set directory"):
        build_pool.build_pool([tmp_path / "nope"], tmp_path / "x.json", out)
    assert not out.exists()


def test_the_printing_does_not_depend_on_the_order_files_are_read(tmp_path):
    sets = {
        "prm": [card("Bear", "o1", set_code="prm")],
        "aaa": [card("Bear", "o1", set_code="aaa")],
    }
    forward = build(tmp_path / "f", sets)["Bear"]
    backward = build(tmp_path / "b", dict(reversed(sets.items())))["Bear"]
    assert forward == backward
    assert forward["set"] == "aaa"  # Not the promo printing of the same day.
