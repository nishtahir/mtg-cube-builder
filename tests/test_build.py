import csv
import json

import pytest

from mtg_cube_builder import builder
from mtg_cube_builder import cli
from mtg_cube_builder import config
from mtg_cube_builder import errors
from mtg_cube_builder import paths as paths_lib


def run(project, **overrides):
    root, template, pool, mine = project
    cfg = config.BuildConfig(
        template="tst", themes=3, max_iter=6, knn=3, **overrides
    )
    result = builder.build_set(paths_lib.Paths(root), cfg, root / "cards.txt")
    return result, template, pool, mine


def test_every_slot_is_filled_with_a_distinct_card(project):
    result, template, pool, mine = run(project)
    assert len(result.rows) == len(template)
    names = [r["card"] for r in result.rows if r["card"]]
    assert len(names) == len(set(names))
    assert {r["source"].split("(")[0] for r in result.rows} <= {
        "list",
        "retrieved",
        "EMPTY",
    }


def test_sources_are_what_they_claim(project):
    result, template, pool, mine = run(project)
    mine_ids = {c["oracle_id"] for c in mine}
    template_ids = {c["oracle_id"] for c in template}
    for r in result.rows:
        if r["source"] == "list":
            assert r["oracle_id"] in mine_ids
        elif r["source"].startswith("retrieved"):
            assert (
                r["oracle_id"] not in mine_ids | template_ids
            )  # never copies the template or reuses a list card


def test_exported_cards_carry_slot_rarity_and_map_position(project):
    result, *_ = run(project)
    by_name = {r["card"]: r for r in result.rows if r["card"]}
    for c in result.cards:
        assert c["rarity"] == by_name[c["name"]]["slot_rarity"]
        assert {
            "printed_rarity",
            "x",
            "y",
            "theme",
            "theme_score",
            "source",
        } <= c.keys()


def test_strict_threshold_pushes_slots_to_retrieval(project):
    loose, *_ = run(project, max_cost=10.0)
    strict, *_ = run(project, max_cost=0.5)
    assert sum(r["source"] == "list" for r in strict.rows) < sum(
        r["source"] == "list" for r in loose.rows
    )


def test_clear_error_when_no_list_card_fits(project):
    with pytest.raises(errors.MtgCubeBuilderError, match="raise --max-cost"):
        run(project, max_cost=-50.0)


def test_cli_build_writes_outputs_and_report(project):
    root, *_ = project
    assert (
        cli.main(
            [
                "--root",
                str(root),
                "build",
                str(root / "cards.txt"),
                "--template",
                "tst",
                "--themes",
                "3",
                "--knn",
                "3",
                "--max-iter",
                "4",
            ]
        )
        == 0
    )
    p = paths_lib.Paths(root)
    assert p.built_csv("tst").exists() and p.report_html("tst").exists()
    cards = json.loads(p.built_json("tst").read_text())
    assert cards and "__DATA__" not in p.report_html("tst").read_text()


def test_cli_reports_expected_failures_as_exit_code_1(project, capsys):
    root, *_ = project
    code = cli.main(
        [
            "--root",
            str(root),
            "build",
            str(root / "cards.txt"),
            "--template",
            "tst",
            "--max-cost",
            "-50",
        ]
    )
    assert code == 1
    assert "raise --max-cost" in capsys.readouterr().err


def test_missing_template_is_a_clear_error(project):
    root, *_ = project
    with pytest.raises(errors.MtgCubeBuilderError, match="template set"):
        builder.build_set(
            paths_lib.Paths(root),
            config.BuildConfig(template="nope"),
            root / "cards.txt",
        )


def test_unknown_card_names_stop_the_build_with_a_suggestion(project):
    root, _, _, mine = project
    names = [c["name"] for c in mine]
    (root / "cards.txt").write_text("\n".join([*names, "Mine Cadr 3"]))
    with pytest.raises(errors.MtgCubeBuilderError, match="Mine Card 3"):
        builder.build_set(
            paths_lib.Paths(root),
            config.BuildConfig(template="tst"),
            root / "cards.txt",
        )


def test_the_card_list_argument_chooses_the_cards(project):
    root, _, _, mine = project
    (root / "other.txt").write_text(
        "".join(c["name"] + "\n" for c in mine[:25])
    )
    assert (
        cli.main(
            [
                "--root",
                str(root),
                "build",
                str(root / "other.txt"),
                "--template",
                "tst",
                "--themes",
                "3",
                "--knn",
                "3",
                "--max-iter",
                "3",
                "--no-report",
            ]
        )
        == 0
    )
    chosen = {
        r["card"]
        for r in csv.DictReader(
            open(paths_lib.Paths(root).built_csv("tst"), encoding="utf-8")
        )
        if r["source"] == "list"
    }
    assert chosen <= {c["name"] for c in mine[:25]}


def test_a_missing_card_list_is_a_clear_error(project, capsys):
    root, *_ = project
    code = cli.main(["--root", str(root), "build", str(root / "nope.txt")])
    assert code == 1
    assert "card list" in capsys.readouterr().err
