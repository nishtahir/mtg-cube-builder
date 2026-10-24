import pytest

from mtg_cube_builder import cardlist
from mtg_cube_builder import errors

CARDS = [
    {"oracle_id": "1", "name": "Sol Ring"},
    {
        "oracle_id": "2",
        "name": "Fire // Ice",
        "card_faces": [{"name": "Fire"}, {"name": "Ice"}],
    },
    {"oracle_id": "3", "name": "Fire"},
    {"oracle_id": "4", "name": "Boseiju, Who Endures"},
    {"oracle_id": "5", "name": "Sakura-Tribe Elder"},
]


def test_read_names_skips_blanks_comments_and_repeats(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("Sol Ring\n\n# a comment\n  Fire  \nSOL RING\nIce\n")
    assert cardlist.read_names(path) == ["Sol Ring", "Fire", "Ice"]


def test_names_match_ignoring_case_and_curly_quotes():
    cards = [{"oracle_id": "9", "name": "Jeska's Will"}]
    assert cardlist.resolve(["JESKA’S will"], cards)[0]["oracle_id"] == "9"


def test_a_face_name_finds_the_multifaced_card():
    assert cardlist.resolve(["Ice"], CARDS)[0]["oracle_id"] == "2"


def test_a_full_name_wins_over_a_face_name():
    assert cardlist.resolve(["Fire"], CARDS)[0]["oracle_id"] == "3"


def test_a_card_named_twice_is_returned_once():
    found = cardlist.resolve(["Fire // Ice", "Ice"], CARDS)
    assert [c["oracle_id"] for c in found] == ["2"]


def test_order_follows_the_list():
    found = cardlist.resolve(["Sakura-Tribe Elder", "Sol Ring"], CARDS)
    assert [c["oracle_id"] for c in found] == ["5", "1"]


def test_unmatched_names_are_listed_with_suggestions():
    with pytest.raises(errors.MtgCubeBuilderError) as raised:
        cardlist.resolve(["Boseju, Who Endures", "Not A Card At All"], CARDS)
    message = str(raised.value)
    assert "2 card name(s)" in message
    assert "did you mean 'Boseiju, Who Endures'" in message
    assert "Not A Card At All" in message


def test_a_missing_list_file_is_a_clear_error(tmp_path):
    from mtg_cube_builder import paths as paths_lib

    with pytest.raises(errors.MtgCubeBuilderError, match="card list"):
        cardlist.load(paths_lib.Paths(tmp_path), tmp_path / "cards.txt", CARDS)
