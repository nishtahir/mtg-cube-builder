import dataclasses
import json

import pytest

from mtg_cube_builder import cli
from mtg_cube_builder import config
from mtg_cube_builder import errors
from mtg_cube_builder import paths as paths_lib


def resolve(tmp_path, argv):
    args = cli._build_parser().parse_args(["--root", str(tmp_path), *argv])
    return cli.load_build_config(args, paths_lib.Paths(tmp_path))


def write_config(tmp_path, text, name="build.json"):
    (tmp_path / "config").mkdir(exist_ok=True)
    path = tmp_path / "config" / name
    path.write_text(text)
    return path


def test_every_setting_is_documented():
    for field in dataclasses.fields(config.BuildConfig):
        assert field.metadata["help"], field.name


def test_json_round_trips():
    cfg = config.BuildConfig(template="eld", themes=3, max_cost=2.0)
    assert config.from_dict(json.loads(config.to_json(cfg))) == cfg


def test_from_dict_accepts_an_integer_for_a_decimal_setting():
    assert config.from_dict({"max_cost": 2}).max_cost == 2.0


def test_from_dict_rejects_unknown_settings_and_lists_valid_ones():
    with pytest.raises(errors.MtgCubeBuilderError, match="max_cots.*max_cost"):
        config.from_dict({"max_cots": 1.0})


def test_from_dict_rejects_wrong_types():
    with pytest.raises(errors.MtgCubeBuilderError, match="themes"):
        config.from_dict({"themes": "eight"})
    with pytest.raises(errors.MtgCubeBuilderError, match="themes"):
        config.from_dict({"themes": True})


def test_load_reports_invalid_json(tmp_path):
    path = write_config(tmp_path, "{themes: 3}")
    with pytest.raises(errors.MtgCubeBuilderError, match="Invalid JSON"):
        config.load(path)


def test_load_requires_a_json_object(tmp_path):
    path = write_config(tmp_path, "[1, 2]")
    with pytest.raises(errors.MtgCubeBuilderError, match="JSON object"):
        config.load(path)


def test_an_empty_file_means_no_settings(tmp_path):
    assert config.load(write_config(tmp_path, "  \n")) == config.BuildConfig()


def test_defaults_without_a_file(tmp_path):
    assert resolve(tmp_path, ["build", "cards.txt"]) == config.BuildConfig()


def test_file_overrides_defaults(tmp_path):
    write_config(tmp_path, '{"template": "tst", "themes": 3}')
    cfg = resolve(tmp_path, ["build", "cards.txt"])
    assert (cfg.template, cfg.themes) == ("tst", 3)
    assert cfg.max_cost == config.BuildConfig().max_cost


def test_options_override_the_file_but_only_those_given(tmp_path):
    write_config(tmp_path, '{"template": "tst", "themes": 3, "knn": 4}')
    cfg = resolve(tmp_path, ["build", "cards.txt", "--themes", "5"])
    assert cfg.themes == 5  # The option wins.
    assert (cfg.template, cfg.knn) == ("tst", 4)  # The file is kept.


def test_explicit_config_file(tmp_path):
    path = write_config(tmp_path, '{"knn": 7}', name="other.json")
    assert (
        resolve(tmp_path, ["build", "cards.txt", "--config", str(path)]).knn
        == 7
    )


def test_missing_explicit_config_file_is_an_error(tmp_path):
    with pytest.raises(errors.MtgCubeBuilderError, match="config file"):
        resolve(
            tmp_path,
            ["build", "cards.txt", "--config", str(tmp_path / "nope.json")],
        )


def test_report_template_comes_from_the_config(tmp_path):
    write_config(tmp_path, '{"template": "tst"}')
    assert resolve(tmp_path, ["report"]).template == "tst"
    assert resolve(tmp_path, ["report", "--template", "abc"]).template == "abc"


def test_config_command_prints_the_effective_settings(tmp_path, capsys):
    write_config(tmp_path, '{"themes": 3}')
    assert cli.main(["--root", str(tmp_path), "config"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["themes"] == 3 and printed["template"] == "fra"


def test_cards_is_a_required_argument_not_a_setting(capsys):
    assert "cards" not in config.setting_names()
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["build"])
    assert "cards" in capsys.readouterr().err
