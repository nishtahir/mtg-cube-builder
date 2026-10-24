# MTG cube set builder

Fit a list of cards you want to play in your cube and to a predifined set and this app helps build a cohesive set.

## How it works
Every card of the template is a slot with a rarity, color, type and mana value. `mtg-cube-builder` fills the slots with cards from your list via optimal assignment, rewarding cards that cluster into themes, and pads slots with high quality selections from outside your list to balance it. 

The output is a set list that losely follows the distribution of the template with the goal of building new sets that are balanced with cohesive themes present in the provided list.

## Setup

This assumes you use `uv`.

```sh
uv venv                 # Create your virtual environment

uv sync                 # installs the package and the `mtg-cube-builder` command
uv run pytest           # optional: run the tests
```

Embeddings use [nishtahir/modern-BERT-MTG-Commander](https://huggingface.co/nishtahir/modern-BERT-MTG-Commander)
(downloaded on first use; runs on the Mac GPU when available).

## Usage

```sh
uv run mtg-cube-builder embed                       # once: embed the pool -> data/embeddings/
uv run mtg-cube-builder build cards.txt             # your card names -> output/set-<template>.{csv,json}, report-<template>.html
uv run mtg-cube-builder build my-list.txt --themes 5   # any list, any settings
uv run mtg-cube-builder packs config/pack-config.json   # optional: booster packs -> output/packs.json
uv run mtg-cube-builder report                      # optional: re-render the HTML report
```

## Configuration

`build` reads its settings from `config/build.json` if it exists, a JSON object of the settings
below. Every setting is optional; a missing one keeps its default. Command-line options override
the file (`uv run mtg-cube-builder build --themes 5 --max-cost 1.0`), and `--config PATH` uses another
file.

```sh
uv run mtg-cube-builder config                       # print the effective settings as JSON
uv run mtg-cube-builder config > config/build.json   # (re)generate the file with every setting
```

| setting | default | meaning |
|---|---|---|
| `template` | `fra` | Code of the real set whose cards define the slots. Its file must be in data/sets/. |
| `max_cost` | `1.5` | A list card costlier than this for a slot is not used and the slot is filled from the pool instead. Lower uses fewer of your cards but fits them better. |
| `w_color` | `2.0` | Cost of a color mismatch. |
| `w_type` | `1.5` | Cost of a card type mismatch. |
| `w_cmc` | `0.5` | Cost per mana value of difference, capped at three. |
| `w_rarity` | `0.4` | Cost per rarity step of difference. |
| `w_coh` | `5.0` | Reward per unit of cohesion (closeness to the other chosen cards of the same theme) above the list median. |
| `w_sim` | `0.0` | Reward for similarity to the template card. Off by default. |
| `w_fill_sim` | `0.0` | Weight of similarity to the template card when filling gaps from the pool. Off by default. |
| `themes` | `8` | Number of themes to cluster the list into. 1 means global cohesion. |
| `theme_alpha` | `0.5` | A theme's share of the set is proportional to its size to this power. 1 is proportional and 0 is equal. |
| `price_step` | `0.5` | How fast an over- or under-filled theme's price moves. |
| `w_balance` | `0.3` | Bonus for filling gaps with cards of under-filled themes. |
| `max_iter` | `30` | Maximum rounds of re-assignment. |
| `knn` | `10` | Number of nearest neighbors averaged to measure cohesion. |
