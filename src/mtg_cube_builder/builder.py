"""Builds a set shaped like a real expansion out of a card list.

1. Template: every card of a real expansion is a slot with a rarity, color,
   type category and mana value.
2. Fit: cards from the list (the `cards` file) are assigned to slots with an
   optimal assignment. A pair costs more for color, type, mana value and rarity
   mismatches, and less for cohesion: how close the card is (mean of its `knn`
   nearest neighbors by embedding) to the other chosen cards of its theme.
   Cohesion depends on the chosen set, so the assignment is repeated with
   updated cohesion until the chosen set stops changing. A slot is left as a
   gap when no list card fits (cost above `max_cost`).
   Themes: the list is clustered into `themes` groups. Each theme gets a quota
   proportional to size ** theme_alpha, enforced with prices that rise for
   over-filled themes, so one theme cannot take over. `themes=1` gives plain
   global cohesion.
3. Fill: each gap gets a card from the full card pool that matches the slot and
   is close to the cards already chosen from the list, favoring under-filled
   themes.

Similarity to the template card is an optional extra (`w_sim`, `w_fill_sim`),
off by default.
"""

import collections
import csv
import dataclasses
import json
import logging
import pathlib
from typing import Any, NamedTuple

import numpy as np
from scipy import optimize
from sklearn import manifold

from mtg_cube_builder import cardinfo
from mtg_cube_builder import cardlist
from mtg_cube_builder import cluster
from mtg_cube_builder import config as config_lib
from mtg_cube_builder import embeddings
from mtg_cube_builder import errors
from mtg_cube_builder import paths as paths_lib
from mtg_cube_builder import pool as pool_lib

_logger = logging.getLogger(__name__)

_KMEANS_RESTARTS = 8
_REPORT_THEMES = 8  # Themes shown in the report even when themes=1.
_MAX_CMC_DIFF = 3  # Mana value differences beyond this cost no extra.
_RELAX_LEVELS = 4  # Ever looser matching rules for filling a gap.
_ALTERNATIVES = 3  # Runner-up cards listed for each retrieved card.
_TSNE_PERPLEXITY = 30
_BALANCE_TOLERANCE = 0.25  # Allowed deviation of a theme from its quota.
_BALANCE_SLACK = 3  # ... or this many cards, whichever is larger.
_CONVERGED_SWAP_FRACTION = 0.02  # Fraction of chosen cards that may still move.


@dataclasses.dataclass(frozen=True)
class BuildResult:
    """What a build produced.

    Attributes:
      rows: One dict per slot, as written to the CSV.
      cards: The chosen cards as Scryfall objects, with `rarity` set to the
        slot's rarity.
      csv_path: Where the rows were written.
      json_path: Where the cards were written.
    """

    rows: list[dict[str, Any]]
    cards: list[dict[str, Any]]
    csv_path: pathlib.Path
    json_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Problem:
    """Everything the fit and fill steps read.

    Attributes:
      slots: The template's cards, one per slot.
      items: The cards of the list that can fill a slot.
      item_objects: The pool card object of each item, by oracle id.
      slot_emb: Embeddings of the slots, (n_slots, dim).
      item_emb: Embeddings of the items, (n_items, dim).
      all_emb: Embeddings of every card.
      row_of: Row of `all_emb` for each oracle id.
      pool: The pool that gaps are filled from.
      struct_cost: (n_slots, n_items) cost of putting an item in a slot, from
        color, type, mana value and rarity alone.
    """

    slots: list[cardinfo.Card]
    items: list[cardinfo.Card]
    item_objects: dict[str, dict[str, Any]]
    slot_emb: np.ndarray
    item_emb: np.ndarray
    all_emb: np.ndarray
    row_of: dict[str, int]
    pool: pool_lib.Pool
    struct_cost: np.ndarray


@dataclasses.dataclass(frozen=True)
class Themes:
    """The clustering of the list into themes.

    Attributes:
      count: Number of themes the fit balances (1 for global cohesion).
      report_count: Number of themes shown in the report.
      centres: Unit-length centres of the report themes, (report_count, dim).
      report_labels: Report theme of each item.
      labels: Theme of each item for the fit (all zeros if count is 1).
      sizes: Number of items per fit theme.
      share: Target share of the set for each fit theme.
    """

    count: int
    report_count: int
    centres: np.ndarray
    report_labels: np.ndarray
    labels: np.ndarray
    sizes: np.ndarray
    share: np.ndarray


@dataclasses.dataclass(frozen=True)
class Fit:
    """The assignment of list cards to slots.

    Attributes:
      assigned: Item index for each slot that got a list card.
      cost: Final (n_slots, n_items) cost matrix.
      cohesion: Cohesion of every item at the end of the fit.
      chosen: Sorted indices of the items that were assigned.
      gaps: Slots that no list card fits.
    """

    assigned: dict[int, int]
    cost: np.ndarray
    cohesion: np.ndarray
    chosen: np.ndarray
    gaps: list[int]


class Fill(NamedTuple):
    """A card retrieved for a gap."""

    oracle_id: str
    alternatives: list[str]
    level: int


def _load_template(paths: paths_lib.Paths, code: str) -> list[cardinfo.Card]:
    """Returns the template's cards, one per oracle card, minus basic lands."""
    path = paths.set_file(code)
    paths.require(path, f"data for template set {code!r}")
    slots: list[cardinfo.Card] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            card = json.loads(line)
            oracle_id = cardinfo.oracle_id_of(card)
            if (
                not oracle_id
                or oracle_id in seen
                or card.get("layout") in cardinfo.SKIP_LAYOUTS
                or cardinfo.is_basic_land(card)
            ):
                continue
            seen.add(oracle_id)
            slots.append(cardinfo.Card.from_scryfall(card, oracle_id))
    return slots


def _load_items(
    listed: list[dict[str, Any]], row_of: dict[str, int]
) -> tuple[list[cardinfo.Card], dict[str, dict[str, Any]]]:
    """Returns the listed cards that can be placed, and their pool objects.

    Basic lands are left out, and so are cards without an embedding (which
    means the embeddings are out of date); those are logged.
    """
    items = []
    objects = {}
    skipped = []
    for card in listed:
        if cardinfo.is_basic_land(card):
            continue
        oracle_id = card["oracle_id"]
        if oracle_id in row_of:
            items.append(cardinfo.Card.from_scryfall(card, oracle_id))
            objects[oracle_id] = card
        else:
            skipped.append(card["name"])
    if skipped:
        _logger.warning(
            "%d list cards have no embedding and are skipped (run "
            "`mtg-cube-builder embed`): %s",
            len(skipped),
            ", ".join(skipped),
        )
    return items, objects


def _structural_costs(
    slots: list[cardinfo.Card],
    items: list[cardinfo.Card],
    sim: np.ndarray,
    cfg: config_lib.BuildConfig,
) -> np.ndarray:
    """Returns the (n_slots, n_items) cost from attribute mismatches."""
    cost = np.empty((len(slots), len(items)))
    for i, slot in enumerate(slots):
        for j, item in enumerate(items):
            cmc_diff = min(abs(slot.cmc - item.cmc), _MAX_CMC_DIFF)
            rarity_diff = abs(
                cardinfo.rank(slot.rarity) - cardinfo.rank(item.rarity)
            )
            cost[i, j] = (
                cfg.w_color * cardinfo.color_cost(slot.colors, item.colors)
                + cfg.w_type * (slot.cat != item.cat)
                + cfg.w_cmc * cmc_diff
                + cfg.w_rarity * rarity_diff
                - cfg.w_sim * sim[i, j]
            )
    return cost


def load_problem(
    paths: paths_lib.Paths,
    cfg: config_lib.BuildConfig,
    cards_file: pathlib.Path | str,
) -> Problem:
    """Loads the template, the card list, the embeddings and the pool.

    Args:
      paths: The project paths.
      cfg: The build settings.
      cards_file: The text file of card names to build the set from.

    Returns:
      The problem to solve.

    Raises:
      errors.MtgCubeBuilderError: If the template set is not downloaded, or a name
        in the card list is not in the pool.
    """
    cards = pool_lib.read_cards(paths)
    pool = pool_lib.fill_pool(cards)
    listed = cardlist.load(paths, cards_file, cards)
    ids, all_emb = embeddings.load_embeddings(paths)
    row_of = {oracle_id: i for i, oracle_id in enumerate(ids)}
    slots = _load_template(paths, cfg.template)
    items, item_objects = _load_items(listed, row_of)
    _logger.info(
        "template %s: %d slots; list: %d cards",
        cfg.template,
        len(slots),
        len(items),
    )
    slot_emb = all_emb[[row_of[s.oracle_id] for s in slots]]
    item_emb = all_emb[[row_of[c.oracle_id] for c in items]]
    struct_cost = _structural_costs(slots, items, slot_emb @ item_emb.T, cfg)
    return Problem(
        slots,
        items,
        item_objects,
        slot_emb,
        item_emb,
        all_emb,
        row_of,
        pool,
        struct_cost,
    )


def cluster_themes(item_emb: np.ndarray, cfg: config_lib.BuildConfig) -> Themes:
    """Clusters the list into themes and sets each theme's quota.

    Args:
      item_emb: Embeddings of the list cards.
      cfg: The build settings.

    Returns:
      The themes. The best of several k-means restarts is used.
    """
    count = max(1, cfg.themes)
    report_count = count if count > 1 else _REPORT_THEMES
    best_score, centres = None, None
    for seed in range(_KMEANS_RESTARTS):
        candidate, _ = cluster.spherical_kmeans(
            item_emb, report_count, np.random.default_rng(seed)
        )
        score = (item_emb @ candidate.T).max(axis=1).mean()
        if best_score is None or score > best_score:
            best_score, centres = score, candidate
    report_labels = np.argmax(item_emb @ centres.T, axis=1)
    labels = report_labels if count > 1 else np.zeros(len(item_emb), dtype=int)
    sizes = np.bincount(labels, minlength=count)
    share = sizes.astype(float) ** cfg.theme_alpha
    share /= share.sum()
    return Themes(
        count, report_count, centres, report_labels, labels, sizes, share
    )


def cohesion(
    item_emb: np.ndarray, labels: np.ndarray, chosen: np.ndarray, knn: int
) -> np.ndarray:
    """Measures how well each list card fits the chosen cards of its theme.

    Args:
      item_emb: Embeddings of the list cards.
      labels: Theme of each list card.
      chosen: Indices of the currently chosen cards.
      knn: How many nearest neighbors to average.

    Returns:
      For each list card, the mean similarity to its `knn` nearest chosen
      cards of the same theme, excluding itself. Cards of a theme with fewer
      than two chosen cards use half of the same measure over all themes.
    """
    sims = item_emb @ item_emb[chosen].T
    sims[chosen, np.arange(len(chosen))] = -np.inf
    same = np.where(labels[:, None] == labels[chosen][None, :], sims, -np.inf)
    within, n_within = cluster.topk_mean(same, knn)
    overall, _ = cluster.topk_mean(sims, knn)
    return np.where(n_within >= 2, within, 0.5 * overall)


def _assign(
    problem: Problem,
    themes: Themes,
    cfg: config_lib.BuildConfig,
    coh: np.ndarray,
    coh_ref: float,
    price: np.ndarray,
) -> tuple[dict[int, int], np.ndarray]:
    """Solves one round of the assignment; returns it and the cost matrix."""
    n_slots, n_items = problem.struct_cost.shape
    cost = (
        problem.struct_cost
        - cfg.w_coh * (coh - coh_ref)[None, :]
        + price[themes.labels][None, :]
    )
    # Extra columns let a slot take "no card" at max_cost.
    padded = np.hstack([cost, np.full((n_slots, n_slots), cfg.max_cost)])
    slot_idx, col_idx = optimize.linear_sum_assignment(padded)
    return (
        {
            int(i): int(j)
            for i, j in zip(slot_idx, col_idx, strict=True)
            if j < n_items
        },
        cost,
    )


def fit(problem: Problem, themes: Themes, cfg: config_lib.BuildConfig) -> Fit:
    """Assigns list cards to slots, repeating as cohesion changes.

    Args:
      problem: The problem to solve.
      themes: The themes of the list.
      cfg: The build settings.

    Returns:
      The assignment.

    Raises:
      errors.MtgCubeBuilderError: If no list card fits any slot.
    """
    n_items = len(problem.items)
    price = np.zeros(themes.count)
    coh = cohesion(problem.item_emb, themes.labels, np.arange(n_items), cfg.knn)
    coh_ref = float(np.median(coh))  # Fixed, so the scale does not drift.
    _logger.info(
        "%d theme(s), list sizes %s; cohesion p10 %.3f  median %.3f  p90 %.3f",
        themes.count,
        themes.sizes.tolist(),
        np.percentile(coh, 10),
        coh_ref,
        np.percentile(coh, 90),
    )
    previous = None
    for iteration in range(1, cfg.max_iter + 1):
        rounds = iteration
        assigned, cost = _assign(problem, themes, cfg, coh, coh_ref, price)
        if not assigned:
            raise errors.MtgCubeBuilderError(
                f"No card in the list fits any slot at max_cost="
                f"{cfg.max_cost}; raise --max-cost."
            )
        chosen = np.array(sorted(assigned.values()), dtype=int)
        counts = np.bincount(themes.labels[chosen], minlength=themes.count)
        quota = themes.share * len(chosen)
        balanced = themes.count == 1 or np.all(
            np.abs(counts - quota)
            <= np.maximum(_BALANCE_TOLERANCE * quota, _BALANCE_SLACK)
        )
        swapped = len(set(chosen) ^ previous) if previous is not None else None
        if (
            swapped is not None
            and swapped <= _CONVERGED_SWAP_FRACTION * len(chosen)
            and balanced
        ):
            break
        previous = set(chosen)
        price += cfg.price_step * (counts - quota) / np.maximum(quota, 1)
        # Damped update, so the assignment does not oscillate.
        coh = 0.5 * coh + 0.5 * cohesion(
            problem.item_emb, themes.labels, chosen, cfg.knn
        )
    _logger.info(
        "assignment %s %d iterations",
        "converged after" if rounds < cfg.max_iter else "did NOT converge in",
        rounds,
    )
    gaps = [i for i in range(len(problem.slots)) if i not in assigned]
    _logger.info("filled from list: %d; gaps: %d", len(assigned), len(gaps))
    return Fit(assigned, cost, coh, chosen, gaps)


def _eligible(entry: dict[str, Any], slot: cardinfo.Card, level: int) -> bool:
    """Returns whether a pool card may fill a slot at a matching level.

    Level 0 wants the same type, color and rarity and a mana value within one.
    Level 1 drops the rarity, level 2 allows a shared color and a mana value
    within two, and level 3 allows any color.
    """
    if entry["cat"] != slot.cat:
        return False
    if (
        level < 3
        and entry["colors"] != slot.colors
        and not (
            level >= 2 and cardinfo.color_cost(slot.colors, entry["colors"]) < 1
        )
    ):
        return False
    if abs(entry["cmc"] - slot.cmc) > (1 if level < 2 else 2):
        return False
    # Pool cards have no known rarity, so rarity cannot rule them out.
    return (
        level >= 1 or not entry["rarities"] or slot.rarity in entry["rarities"]
    )


def fill_gaps(
    problem: Problem,
    themes: Themes,
    fit_result: Fit,
    cfg: config_lib.BuildConfig,
) -> dict[int, Fill]:
    """Retrieves a card from the pool for each gap.

    Args:
      problem: The problem being solved.
      themes: The themes of the list.
      fit_result: The assignment of list cards to slots.
      cfg: The build settings.

    Returns:
      The card for each slot that could be filled.
    """
    used = {c.oracle_id for c in problem.items} | {
        s.oracle_id for s in problem.slots
    }  # Never copy or reuse.
    pool_ids = [
        o
        for o, e in problem.pool.items()
        if o in problem.row_of and o not in used
    ]
    entries = [problem.pool[o] for o in pool_ids]
    pool_emb = problem.all_emb[[problem.row_of[o] for o in pool_ids]]
    if themes.count > 1:
        pool_theme = np.argmax(pool_emb @ themes.centres.T, axis=1)
    else:
        pool_theme = np.zeros(len(pool_ids), dtype=int)

    chosen = fit_result.chosen
    chosen_emb = problem.item_emb[chosen]
    overall, _ = cluster.topk_mean(pool_emb @ chosen_emb.T, cfg.knn)
    context = 0.5 * overall  # For themes with fewer than two chosen cards.
    for t in range(themes.count):
        theme_chosen = chosen[themes.labels[chosen] == t]
        if len(theme_chosen) >= 2:
            mask = pool_theme == t
            context[mask], _ = cluster.topk_mean(
                pool_emb[mask] @ problem.item_emb[theme_chosen].T, cfg.knn
            )

    counts = np.bincount(themes.labels[chosen], minlength=themes.count).astype(
        float
    )
    target = themes.share * len(problem.slots)
    taken: set[str] = set()
    fills: dict[int, Fill] = {}
    for i in fit_result.gaps:
        slot = problem.slots[i]
        slot_sim = pool_emb @ problem.slot_emb[i]
        deficit = np.clip((target - counts) / np.maximum(target, 1), -1, 1)
        bonus = cfg.w_balance * deficit[pool_theme] if themes.count > 1 else 0.0
        for level in range(_RELAX_LEVELS):
            ok = np.array(
                [
                    pool_ids[k] not in taken
                    and _eligible(entries[k], slot, level)
                    for k in range(len(pool_ids))
                ]
            )
            if not ok.any():
                continue
            score = np.where(
                ok, context + bonus + cfg.w_fill_sim * slot_sim, -np.inf
            )
            order = np.argsort(-score)[: _ALTERNATIVES + 1]
            best = pool_ids[order[0]]
            taken.add(best)
            if themes.count > 1:
                counts[pool_theme[order[0]]] += 1
            fills[i] = Fill(
                best, [entries[k]["name"] for k in order[1:]], level
            )
            break
    return fills


def make_rows(
    problem: Problem, themes: Themes, fit_result: Fit, fills: dict[int, Fill]
) -> tuple[list[dict[str, Any]], list[str | None]]:
    """Describes what fills each slot.

    Args:
      problem: The problem being solved.
      themes: The themes of the list.
      fit_result: The assignment of list cards to slots.
      fills: The cards retrieved for gaps.

    Returns:
      One dict per slot, and the oracle id of the card in each slot (None if
      the slot is empty).
    """
    rows: list[dict[str, Any]] = []
    oracle_ids: list[str | None] = []
    for i, slot in enumerate(problem.slots):
        row: dict[str, Any] = {
            "slot_rarity": slot.rarity,
            "slot_colors": slot.colors or "C",
            "slot_type": slot.cat,
            "slot_cmc": slot.cmc,
            "template_card": slot.name,
        }
        if i in fit_result.assigned:
            j = fit_result.assigned[i]
            item = problem.items[j]
            oracle_id = item.oracle_id
            row.update(
                {
                    "card": item.name,
                    "source": "list",
                    "printed_rarity": item.rarity,
                    "card_colors": item.colors or "C",
                    "card_type": item.cat,
                    "card_cmc": item.cmc,
                    "fit": round(float(fit_result.cost[i, j]), 2),
                    "cohesion": round(float(fit_result.cohesion[j]), 3),
                    "alternatives": "",
                }
            )
        elif i in fills:
            oracle_id, alternatives, level = fills[i]
            entry = problem.pool[oracle_id]
            row.update(
                {
                    "card": entry["name"],
                    "source": f"retrieved(relax{level})",
                    "printed_rarity": "/".join(entry["rarities"]),
                    "card_colors": entry["colors"] or "C",
                    "card_type": entry["cat"],
                    "card_cmc": entry["cmc"],
                    "fit": "",
                    "cohesion": "",
                    "alternatives": "; ".join(alternatives),
                }
            )
        else:
            oracle_id = None
            row.update(
                {
                    "card": "",
                    "source": "EMPTY",
                    "printed_rarity": "",
                    "card_colors": "",
                    "card_type": "",
                    "card_cmc": "",
                    "fit": "",
                    "cohesion": "",
                    "alternatives": "",
                }
            )
        row["theme"] = ""
        if oracle_id:
            vector = problem.all_emb[problem.row_of[oracle_id]]
            row["theme"] = int(np.argmax(vector @ themes.centres.T))
        row["oracle_id"] = oracle_id or ""
        rows.append(row)
        oracle_ids.append(oracle_id)
    return rows, oracle_ids


def export_cards(
    paths: paths_lib.Paths,
    problem: Problem,
    themes: Themes,
    rows: list[dict[str, Any]],
    oracle_ids: list[str | None],
) -> list[dict[str, Any]]:
    """Returns the chosen cards as Scryfall objects.

    Each has the slot's rarity in `rarity` (the printed one moves to
    `printed_rarity`), plus its source, theme, template card and a 2D map
    position `x`, `y` (a t-SNE of the embeddings) for the report.

    Args:
      paths: The project paths.
      problem: The problem being solved.
      themes: The themes of the list.
      rows: The per-slot rows from `make_rows`.
      oracle_ids: The card in each slot, from `make_rows`.
    """
    unique = sorted({o for o in oracle_ids if o})
    objects = problem.item_objects
    embedded = problem.all_emb[[problem.row_of[o] for o in unique]]
    xy = manifold.TSNE(
        n_components=2,
        metric="cosine",
        init="pca",
        perplexity=min(_TSNE_PERPLEXITY, len(unique) - 1),
        random_state=0,
    ).fit_transform(embedded)
    xy = (xy - xy.mean(axis=0)) / np.abs(xy - xy.mean(axis=0)).max(axis=0)
    coords = {
        o: (float(xy[k, 0]), float(xy[k, 1])) for k, o in enumerate(unique)
    }
    cards = []
    for row, oracle_id in zip(rows, oracle_ids, strict=True):
        if not oracle_id:
            continue
        card = dict(objects.get(oracle_id) or problem.pool[oracle_id]["card"])
        vector = problem.all_emb[problem.row_of[oracle_id]]
        card.update(
            {
                "printed_rarity": card["rarity"],
                "rarity": row["slot_rarity"],
                "source": row["source"].split("(")[0],
                "theme": row["theme"],
                "template_card": row["template_card"],
                "x": round(coords[oracle_id][0], 4),
                "y": round(coords[oracle_id][1], 4),
                "theme_score": round(
                    float(vector @ themes.centres[row["theme"]]), 4
                ),
            }
        )
        cards.append(card)
    return cards


def _write_outputs(
    paths: paths_lib.Paths,
    template: str,
    rows: list[dict[str, Any]],
    cards: list[dict[str, Any]],
) -> None:
    """Writes the per-slot CSV and the card list."""
    paths.output.mkdir(parents=True, exist_ok=True)
    paths.built_json(template).write_text(
        json.dumps(cards, ensure_ascii=False, indent=1)
    )
    with open(
        paths.built_csv(template), "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_top_similarity(vectors: np.ndarray, knn: int) -> float:
    """Returns the mean over cards of their top-knn similarity to the others."""
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    return float(np.sort(sims, axis=1)[:, -knn:].mean())


def log_summary(
    paths: paths_lib.Paths,
    problem: Problem,
    themes: Themes,
    fit_result: Fit,
    fills: dict[int, Fill],
    rows: list[dict[str, Any]],
    cfg: config_lib.BuildConfig,
    n_cards: int,
) -> None:
    """Logs how well the built set clusters, theme by theme."""
    final_ids = [
        problem.items[j].oracle_id for j in fit_result.assigned.values()
    ] + [fill.oracle_id for fill in fills.values()]
    final = problem.all_emb[[problem.row_of[o] for o in final_ids]]
    listed = [r for r in rows if r["source"] == "list"]
    list_idx = np.array(sorted(fit_result.assigned.values()), dtype=int)
    _logger.info(
        "\nlist cards used: %d / %d; retrieved: %d; empty: %d",
        len(listed),
        len(problem.items),
        sum(r["source"].startswith("retrieved") for r in rows),
        sum(r["source"] == "EMPTY" for r in rows),
    )
    _logger.info(
        "SET COHESION (mean top-%d similarity): all %d chosen cards %.4f | "
        "list cards only %.4f",
        cfg.knn,
        len(final_ids),
        _mean_top_similarity(final, cfg.knn),
        _mean_top_similarity(problem.item_emb[list_idx], cfg.knn),
    )
    _logger.info(
        "fit cost (list cards): median %.2f  p90 %.2f",
        *np.percentile([r["fit"] for r in listed], [50, 90]),
    )
    _logger.info(
        "list cards whose printed rarity matches slot: %.0f%% | color match: "
        "%.0f%% | type match: %.0f%%",
        100
        * np.mean([r["printed_rarity"] == r["slot_rarity"] for r in listed]),
        100 * np.mean([r["card_colors"] == r["slot_colors"] for r in listed]),
        100 * np.mean([r["card_type"] == r["slot_type"] for r in listed]),
    )

    final_theme = np.argmax(final @ themes.centres.T, axis=1)
    counts = np.bincount(final_theme, minlength=themes.report_count)
    _logger.info(
        "\ntheme  list  chosen(list+retr)  within-theme cohesion   examples"
    )
    for t in range(themes.report_count):
        members = list_idx[themes.report_labels[list_idx] == t]
        n_list = int((themes.report_labels == t).sum())
        tight = float("nan")
        examples: list[str] = []
        if len(members):
            vectors = problem.item_emb[members]
            if len(members) >= 2:
                sims = vectors @ vectors.T
                np.fill_diagonal(sims, -np.inf)
                tight = float(cluster.topk_mean(sims, cfg.knn)[0].mean())
            nearest = members[np.argsort(-(vectors @ themes.centres[t]))][:3]
            examples = [problem.items[j].name for j in nearest]
        _logger.info(
            "  %-5d%5d  %4d + %-4d            %8.3f          %s",
            t,
            n_list,
            len(members),
            int(counts[t]) - len(members),
            tight,
            ", ".join(examples),
        )
    margins = np.sort(final @ themes.centres.T, axis=1)
    share = counts / counts.sum()
    entropy = float(
        -(share * np.log(share + 1e-12)).sum() / np.log(themes.report_count)
    )
    _logger.info(
        "theme balance (normalised entropy of chosen cards): %.3f | cluster "
        "separation (own minus next-best theme similarity): %.3f",
        entropy,
        float((margins[:, -1] - margins[:, -2]).mean()),
    )
    by_rarity: dict[str, collections.Counter] = {}
    for row in rows:
        by_rarity.setdefault(row["slot_rarity"], collections.Counter())[
            row["source"].split("(")[0]
        ] += 1
    _logger.info(
        "per slot rarity: %s",
        {k: dict(v) for k, v in sorted(by_rarity.items())},
    )
    _logger.info(
        "Wrote %s and %s (%d cards)",
        paths.rel(paths.built_csv(cfg.template)),
        paths.rel(paths.built_json(cfg.template)),
        n_cards,
    )


def build_set(
    paths: paths_lib.Paths,
    cfg: config_lib.BuildConfig,
    cards_file: pathlib.Path | str,
) -> BuildResult:
    """Builds a set from the card list and writes it to output/.

    Args:
      paths: The project paths.
      cfg: The build settings.
      cards_file: The text file of card names to build the set from, one per
        line.

    Returns:
      The built set.

    Raises:
      errors.MtgCubeBuilderError: If the template is not downloaded, a name in the
        card list is not in the pool, or no list card fits any slot.
    """
    problem = load_problem(paths, cfg, cards_file)
    themes = cluster_themes(problem.item_emb, cfg)
    fit_result = fit(problem, themes, cfg)
    fills = fill_gaps(problem, themes, fit_result, cfg)
    rows, oracle_ids = make_rows(problem, themes, fit_result, fills)
    cards = export_cards(paths, problem, themes, rows, oracle_ids)
    _write_outputs(paths, cfg.template, rows, cards)
    log_summary(
        paths, problem, themes, fit_result, fills, rows, cfg, len(cards)
    )
    return BuildResult(
        rows,
        cards,
        paths.built_csv(cfg.template),
        paths.built_json(cfg.template),
    )
