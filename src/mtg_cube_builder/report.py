"""Self-contained HTML report for a built set.

The page has charts (mana curve, colors, rarity, themes), a theme map and a
filterable card gallery. The map is drawn when the cards carry x/y positions
and themes, as built sets do. There is no build step and no dependency; card
images are loaded from Scryfall when the page is opened.
"""

import collections
import html
import json
import pathlib
from typing import Any

_TEMPLATE = pathlib.Path(__file__).parent / "templates" / "report.html"

Card = dict[str, Any]
Rows = list[tuple[Any, ...]]

_COLOR_ORDER = (
    "White",
    "Blue",
    "Black",
    "Red",
    "Green",
    "Multicolor",
    "Colorless",
    "Land",
)
_RARITY_ORDER = ("common", "uncommon", "rare", "mythic", "special")
_KEYWORD_RENAME = {"Equip": "Equipment"}
_KEYWORD_SKIP = frozenset({"Enchant"})  # Aura noise, not a theme.
_COLOR_NAMES = {
    "W": "White",
    "U": "Blue",
    "B": "Black",
    "R": "Red",
    "G": "Green",
}
_TOP_THEMES = 12
_MAX_CMC = 7
_THEME_COLORS = 8  # Themes beyond this share one grey.


def _colors_of(card: Card) -> list[str]:
    """Returns the color letters of a card, looking at faces if needed."""
    colors = card.get("colors")
    if colors is None:
        colors = {
            c
            for face in card.get("card_faces", [])
            for c in face.get("colors", [])
        }
    return list(colors)


def color_bucket(card: Card) -> str:
    """Returns the color group: a color, Multicolor, Colorless or Land."""
    if "Land" in card["type_line"]:
        return "Land"
    colors = _colors_of(card)
    if not colors:
        return "Colorless"
    return "Multicolor" if len(colors) > 1 else _COLOR_NAMES[colors[0]]


def _creature_types(card: Card) -> list[str]:
    """Returns the creature types of a card's front face."""
    front = card["type_line"].split(" // ")[0]
    if "Creature" not in front or "—" not in front:
        return []
    return front.split("—")[1].split()


def top_themes(cards: list[Card]) -> Rows:
    """Returns the most common creature types and keywords.

    Args:
      cards: Scryfall card objects.

    Returns:
      Up to twelve (name, number of cards, kind) rows, most common first.
    """
    tribes: collections.Counter = collections.Counter()
    mechanics: collections.Counter = collections.Counter()
    for card in cards:
        tribes.update(set(_creature_types(card)))
        mechanics.update(
            _KEYWORD_RENAME.get(k, k)
            for k in set(card.get("keywords", []))
            if k not in _KEYWORD_SKIP
        )
    rows = [(name, n, "Creature type") for name, n in tribes.items()]
    rows += [(name, n, "Keyword / mechanic") for name, n in mechanics.items()]
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:_TOP_THEMES]


def _bar_path(
    x: float, y: float, width: float, height: float, radius: float = 4
) -> str:
    """Returns an SVG path for a bar with rounded top corners."""
    r = min(radius, width / 2, height)
    return (
        f"M{x:.1f},{y + height:.1f}V{y + r:.1f}a{r},{r} 0 0 1 {r},{-r}"
        f"H{x + width - r:.1f}a{r},{r} 0 0 1 {r},{r}V{y + height:.1f}Z"
    )


def vbar_chart(rows: Rows, unit: str = "cards") -> str:
    """Draws a single-series vertical bar chart with values above the bars.

    Args:
      rows: (label, value) pairs.
      unit: Name of the counted thing, shown in the hover text.

    Returns:
      SVG markup.
    """
    width, height, left, right, top, bottom = 560, 260, 8, 8, 24, 36
    base = height - bottom
    vmax = max(value for _, value in rows) or 1
    slot = (width - left - right) / len(rows)
    bar_width = min(48, slot * 0.7)
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img">',
        f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" '
        f'y2="{base}"/>',
    ]
    for i, (label, value) in enumerate(rows):
        x = left + slot * i + (slot - bar_width) / 2
        bar_height = (base - top) * value / vmax
        y = base - bar_height
        tip = html.escape(f"{label}: {value} {unit}")
        if value:
            out.append(
                f'<path class="bar s1" '
                f'd="{_bar_path(x, y, bar_width, bar_height)}"/>'
            )
        out.append(
            f'<text class="val" x="{x + bar_width / 2:.1f}" '
            f'y="{y - 6:.1f}">{value}</text>'
        )
        out.append(
            f'<text class="lab" x="{x + bar_width / 2:.1f}" '
            f'y="{base + 18}">{html.escape(label)}</text>'
        )
        out.append(
            f'<rect class="hit" data-tip="{tip}" '
            f'x="{left + slot * i:.1f}" y="0" width="{slot:.1f}" '
            f'height="{base}"/>'
        )
    out.append("</svg>")
    return "".join(out)


def hbar_chart(rows: Rows) -> str:
    """Draws horizontal bars colored by group, with a legend.

    Args:
      rows: (label, value, group) triples. Groups take the first and second
        series colors in alphabetical order.

    Returns:
      HTML markup.
    """
    width, left, right, row_height = 560, 120, 36, 26
    height = row_height * len(rows) + 8
    vmax = max(value for _, value, _ in rows)
    groups = sorted({group for _, _, group in rows})
    series = {group: f"s{i + 1}" for i, group in enumerate(groups)}
    legend = "".join(
        f'<span><i class="sw {series[g]}"></i>{html.escape(g)}</span>'
        for g in groups
    )
    out = [
        f'<div class="legend">{legend}</div>',
        f'<svg viewBox="0 0 {width} {height}" role="img">',
    ]
    for i, (label, value, group) in enumerate(rows):
        y = 4 + row_height * i
        bar_width = (width - left - right) * value / vmax
        tip = html.escape(f"{label} ({group}): {value} cards")
        out.append(
            f'<text class="lab r" x="{left - 8}" y="{y + 14}">'
            f"{html.escape(label)}</text>"
        )
        out.append(
            f'<path class="bar {series[group]}" d="M{left},{y + 3}'
            f"H{left + bar_width - 4:.1f}a4,4 0 0 1 4,4v4a4,4 0 0 1 "
            f'-4,4H{left}Z"/>'
        )
        out.append(
            f'<text class="val l" x="{left + bar_width + 6:.1f}" '
            f'y="{y + 14}">{value}</text>'
        )
        out.append(
            f'<rect class="hit" data-tip="{tip}" x="0" y="{y}" '
            f'width="{width}" height="{row_height}"/>'
        )
    out.append("</svg>")
    return "".join(out)


def theme_panel(cards: list[Card]) -> str:
    """Draws a map of the cards colored by theme, with a clickable theme list.

    Args:
      cards: Built-set cards, each with `x`, `y`, `theme`, `theme_score`,
        `source` and `rarity`.

    Returns:
      HTML for a full-width panel.
    """
    width, height, margin = 760, 460, 30

    def px(v: float) -> float:
        return margin + (v + 1) / 2 * (width - 2 * margin)

    def py(v: float) -> float:
        return height - margin - (v + 1) / 2 * (height - 2 * margin)

    ids = sorted({c["theme"] for c in cards})
    slot = {t: min(i, _THEME_COLORS) for i, t in enumerate(ids)}
    by_theme = {t: [c for c in cards if c["theme"] == t] for t in ids}
    label = {
        t: ", ".join(
            c["name"].split(" // ")[0]
            for c in sorted(by_theme[t], key=lambda c: -c["theme_score"])[:2]
        )
        for t in ids
    }
    points, labels = [], []
    for card in cards:
        k = slot[card["theme"]]
        ring = card.get("source") == "retrieved"
        tip = html.escape(
            f"{card['name']} - theme {card['theme']}, "
            f"{card.get('source', '')}, {card['rarity']}"
        )
        style = f' style="stroke:var(--t{k})"' if ring else ""
        points.append(
            f'<circle class="pt t{k}{" ring" if ring else ""}" '
            f'data-theme="{card["theme"]}" data-tip="{tip}" '
            f'cx="{px(card["x"]):.1f}" cy="{py(card["y"]):.1f}" r="5"{style}/>'
        )
    for t in ids:
        xs = [c["x"] for c in by_theme[t]]
        ys = [c["y"] for c in by_theme[t]]
        labels.append(
            f'<text class="tlab" x="{px(sum(xs) / len(xs)):.1f}" '
            f'y="{py(sum(ys) / len(ys)):.1f}">{t}</text>'
        )
    svg = (
        f'<svg id="tsvg" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Cards plotted by embedding similarity, coloured by '
        'theme">' + "".join(points) + "".join(labels) + "</svg>"
    )

    def n_from(theme: int, source: str) -> int:
        return sum(c.get("source") == source for c in by_theme[theme])

    buttons = []
    for t in ids:
        n_list = n_from(t, "list")
        buttons.append(
            f'<button data-theme-btn="{t}"><i class="bg{slot[t]}"></i>'
            f"<span><b>{t}</b> {html.escape(label[t])}"
            f"<small>{n_list} from list, {len(by_theme[t]) - n_list} "
            f'retrieved</small></span><span class="n">{len(by_theme[t])}'
            "</span></button>"
        )
    legend = (
        '<p class="note">Filled dot = from your list, ring = retrieved. '
        "Position: t-SNE of the card embeddings, so nearby cards are similar "
        "but distances between clusters are not meaningful. Click a theme to "
        "filter the cards below.</p>"
    )
    data_table = table(
        ["Theme", "Cards", "List", "Retrieved", "Top cards"],
        [
            (
                t,
                len(by_theme[t]),
                n_from(t, "list"),
                n_from(t, "retrieved"),
                label[t],
            )
            for t in ids
        ],
    )
    return (
        '<section class="card wide"><h2>Theme map</h2>'
        f'<p class="note">{len(ids)} themes (clusters of the list by '
        "embedding similarity).</p>"
        f'<div class="tmap"><div>{svg}</div>'
        f'<div class="tl">{"".join(buttons)}</div></div>{legend}'
        f"{data_table}</section>"
    )


def card_records(cards: list[Card]) -> list[dict[str, Any]]:
    """Returns the fields of each card that the page's script needs."""
    theme_ids = sorted(
        {c["theme"] for c in cards if c.get("theme") is not None}
    )
    theme_slot = {t: min(i, _THEME_COLORS) for i, t in enumerate(theme_ids)}
    records = []
    for card in cards:
        front = (card.get("card_faces") or [card])[0]
        images = card.get("image_uris") or front.get("image_uris") or {}
        records.append(
            {
                "name": card["name"],
                "rarity": card["rarity"],
                "printed": card.get("printed_rarity"),
                "color": color_bucket(card),
                "cmc": int(card.get("cmc", 0)),
                "type": card.get("type_line", ""),
                "mana": front.get("mana_cost", ""),
                "img": images.get("normal"),
                "url": card.get("scryfall_uri"),
                "source": card.get("source"),
                "theme": card.get("theme"),
                "tslot": theme_slot.get(card.get("theme")),
            }
        )
    return records


def table(headers: list[str], rows: Rows) -> str:
    """Returns a collapsed HTML table of the data behind a chart."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
        + "</tr>"
        for row in rows
    )
    return (
        "<details><summary>Show data table</summary>"
        f"<table><tr>{head}</tr>{body}</table></details>"
    )


def panel(title: str, note: str, chart: str, data_table: str) -> str:
    """Returns a titled panel holding a chart and its data table."""
    return (
        f'<section class="card"><h2>{html.escape(title)}</h2>'
        f'<p class="note">{html.escape(note)}</p>{chart}{data_table}'
        "</section>"
    )


def _rarity_note(cards: list[Card]) -> str:
    if cards and "printed_rarity" in cards[0]:
        changed = sum(c["rarity"] != c["printed_rarity"] for c in cards)
        return (
            f"Slot rarity from the template; {changed} cards differ from "
            "their printed rarity."
        )
    return "Rarity of each card's default Scryfall printing."


def write_report(
    cards: list[Card],
    out_path: pathlib.Path | str,
    title: str = "Set report",
    source: str = "cards",
) -> pathlib.Path:
    """Renders cards to a self-contained HTML file.

    Args:
      cards: Scryfall card objects, optionally with the fields a build adds.
      out_path: Where to write the page.
      title: The page title.
      source: Name of the card file, shown under the title.

    Returns:
      The path written.
    """
    nonland = [c for c in cards if "Land" not in c["type_line"]]

    curve = collections.Counter(
        min(int(c.get("cmc", 0)), _MAX_CMC) for c in nonland
    )
    curve_rows = [
        (f"{i}+" if i == _MAX_CMC else str(i), curve[i])
        for i in range(_MAX_CMC + 1)
    ]
    colors = collections.Counter(color_bucket(c) for c in cards)
    color_rows = [(k, colors[k]) for k in _COLOR_ORDER]
    rarities = collections.Counter(c["rarity"] for c in cards)
    rarity_rows = [
        (r.capitalize(), rarities[r]) for r in _RARITY_ORDER if rarities[r]
    ]
    theme_rows = top_themes(cards)

    panels = [
        panel(
            "Mana curve",
            f"Mana value of the {len(nonland)} nonland cards (7+ pooled).",
            vbar_chart(curve_rows),
            table(["Mana value", "Cards"], curve_rows),
        ),
        panel(
            "Color distribution",
            "Card color; lands are counted separately.",
            vbar_chart(color_rows),
            table(["Color", "Cards"], color_rows),
        ),
        panel(
            "Rarity",
            _rarity_note(cards),
            vbar_chart(rarity_rows),
            table(["Rarity", "Cards"], rarity_rows),
        ),
        panel(
            "Themes",
            f"Top {_TOP_THEMES} creature types and keywords by number of "
            "cards.",
            hbar_chart(theme_rows),
            table(["Theme", "Cards", "Kind"], theme_rows),
        ),
    ]
    if cards and "x" in cards[0] and "theme" in cards[0]:
        panels.append(theme_panel(cards))

    data = json.dumps(card_records(cards), ensure_ascii=False)
    page = (
        _TEMPLATE.read_text(encoding="utf-8")
        .replace("__TITLE__", html.escape(title))
        .replace("__SUB__", f"{len(cards)} cards from {source}")
        .replace("__PANELS__", "".join(panels))
        .replace("__DATA__", data.replace("</", "<\\/"))
    )
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
