import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from mtg_cube_builder import embeddings
from mtg_cube_builder import paths as paths_lib

COLORS = ["W", "U", "B", "R", "G"]
TYPES = [
    "Creature — Elf",
    "Instant",
    "Sorcery",
    "Artifact",
    "Enchantment",
    "Land",
]
RARITIES = ["common", "uncommon", "rare", "mythic"]


def make_card(i, prefix, rng):
    color = [] if i % 7 == 0 else [COLORS[i % 5]]
    return {
        "object": "card",
        "id": f"{prefix}-id-{i}",
        "oracle_id": f"{prefix}-{i}",
        "name": f"{prefix.title()} Card {i}",
        "layout": "normal",
        "set": {"tpl": "tst", "pool": "oth", "mine": "min"}[prefix],
        "released_at": "2024-01-01",
        "type_line": TYPES[i % len(TYPES)],
        "cmc": float(i % 6),
        "mana_cost": "{1}",
        "colors": color,
        "rarity": RARITIES[i % 4],
        "keywords": [],
        "digital": False,
        "rarities": [RARITIES[i % 4]],
        "paper": True,
    }


@pytest.fixture
def project(tmp_path):
    """A tiny project with random embeddings.

    It has a 30-slot template (a set file), a 40-card list (cards.txt) and
    60 more cards in the pool (data/cards.jsonl).
    """
    rng = np.random.default_rng(0)
    root = tmp_path
    template = [make_card(i, "tpl", rng) for i in range(30)]
    pool = [make_card(i, "pool", rng) for i in range(60)]
    mine = [make_card(i, "mine", rng) for i in range(40)]

    sets = root / "data" / "sets"
    sets.mkdir(parents=True)
    (sets / "tst.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in template)
    )
    (root / "cards.txt").write_text("".join(c["name"] + "\n" for c in mine))

    paths = paths_lib.Paths(root)
    paths.pool.write_text(
        "".join(json.dumps(c) + "\n" for c in template + pool + mine)
    )
    rows = embeddings.collect(paths)
    centres = rng.normal(size=(4, 32))
    vecs = np.stack(
        [centres[i % 4] + 0.5 * rng.normal(size=32) for i in range(len(rows))]
    ).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    emb = paths.embeddings_dir
    emb.mkdir(parents=True)
    (emb / "cards.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    save_file({"embeddings": vecs}, str(emb / "embeddings.safetensors"))
    return root, template, pool, mine
