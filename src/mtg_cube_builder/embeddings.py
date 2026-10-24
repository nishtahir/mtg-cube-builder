"""Card embeddings from nishtahir/modern-BERT-MTG-Commander.

Each unique oracle card is serialized in Scryfall's text format and encoded
with mean pooling and L2 normalization, as in the model card. Torch is only
imported when embeddings are computed, so the other commands stay fast.
"""

import json
import logging
from typing import Any

import numpy as np
from safetensors import numpy as safetensors_numpy

from mtg_cube_builder import errors
from mtg_cube_builder import paths as paths_lib
from mtg_cube_builder import pool

_logger = logging.getLogger(__name__)
MODEL = "nishtahir/modern-BERT-MTG-Commander"
_CARDS_FILE = "cards.jsonl"
_TENSORS_FILE = "embeddings.safetensors"


def face_text(face: dict[str, Any]) -> str:
    """Returns one card face in Scryfall's text format.

    That is the name and cost, the type line, the rules text and then the
    power and toughness, loyalty or defense, one per line.
    """
    head = f"{face.get('name', '')} {face.get('mana_cost', '')}".strip()
    lines = [head, face.get("type_line", "")]
    if face.get("oracle_text"):
        lines.append(face["oracle_text"])
    if "power" in face and "toughness" in face:
        lines.append(f"{face['power']}/{face['toughness']}")
    if "loyalty" in face:
        lines.append(f"Loyalty: {face['loyalty']}")
    if "defense" in face:
        lines.append(f"Defense: {face['defense']}")
    return "\n".join(line for line in lines if line)


def card_text(card: dict[str, Any]) -> str:
    """Returns a card in Scryfall's text format, faces split by '----'."""
    faces = card.get("card_faces") or [card]
    return "\n----\n".join(face_text(face) for face in faces)


def collect(paths: paths_lib.Paths) -> list[dict[str, str]]:
    """Returns one row per pool card to embed.

    Args:
      paths: The project paths.

    Returns:
      Dicts with "oracle_id", "name" and "text".
    """
    return [
        {
            "oracle_id": card["oracle_id"],
            "name": card["name"],
            "text": card_text(card),
        }
        for card in pool.read_cards(paths)
    ]


def encode(
    rows: list[dict[str, str]],
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
):
    """Embeds the text of each row with the model.

    Args:
      rows: Dicts with a "text" key.
      batch_size: How many cards to encode at once.
      max_length: Cards longer than this many tokens are truncated.
      device: The torch device. Defaults to mps when available, else cpu.

    Returns:
      A torch tensor of shape (len(rows), dim) with unit-length rows, in the
      order of `rows`.
    """
    # Imported here because torch takes seconds to load and most commands
    # do not need it.
    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn.functional as functional  # pylint: disable=import-outside-toplevel
    from transformers import AutoModel  # pylint: disable=import-outside-toplevel
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval().to(device)
    # Sort by length so each batch needs little padding.
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    vectors = [None] * len(rows)
    truncated = 0
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            enc = tokenizer(
                [rows[i]["text"] for i in idx],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            truncated += int((enc["attention_mask"].sum(1) == max_length).sum())
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            batch_vectors = functional.normalize(pooled, p=2, dim=1).cpu()
            for i, vector in zip(idx, batch_vectors, strict=True):
                vectors[i] = vector
            if (start // batch_size) % 20 == 0:
                _logger.info(
                    "  %d/%d", min(start + batch_size, len(order)), len(order)
                )
    if truncated:
        _logger.warning(
            "%d cards hit max_length=%d and were truncated",
            truncated,
            max_length,
        )
    return torch.stack(vectors)


def embed_all(
    paths: paths_lib.Paths,
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
) -> None:
    """Embeds the card pool into data/embeddings/."""
    from safetensors.torch import save_file  # pylint: disable=import-outside-toplevel

    rows = collect(paths)
    _logger.info("%d unique oracle cards", len(rows))
    embeddings = encode(rows, batch_size, max_length, device)
    paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
    with open(paths.embeddings_dir / _CARDS_FILE, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    save_file(
        {"embeddings": embeddings.contiguous()},
        str(paths.embeddings_dir / _TENSORS_FILE),
    )
    _logger.info(
        "Wrote %s embeddings to %s",
        tuple(embeddings.shape),
        paths.rel(paths.embeddings_dir),
    )


def load_embeddings(paths: paths_lib.Paths) -> tuple[list[str], np.ndarray]:
    """Loads the saved embeddings.

    Args:
      paths: The project paths.

    Returns:
      The oracle ids and a float32 matrix of unit vectors, row i belonging to
      id i.
    """
    for name in (_CARDS_FILE, _TENSORS_FILE):
        paths.require(
            paths.embeddings_dir / name, "embeddings", "Run `mtg-cube-builder embed`."
        )
    with open(paths.embeddings_dir / _CARDS_FILE, encoding="utf-8") as f:
        ids = [json.loads(line)["oracle_id"] for line in f]
    matrix = safetensors_numpy.load_file(
        str(paths.embeddings_dir / _TENSORS_FILE)
    )["embeddings"]
    return ids, matrix


def nearest(
    paths: paths_lib.Paths, name: str, k: int = 10
) -> tuple[str, list[tuple[float, str]]]:
    """Finds the cards closest to a card by embedding.

    Args:
      paths: The project paths.
      name: A card name. An exact match wins; otherwise the first card whose
        name contains it.
      k: How many neighbors to return.

    Returns:
      The matched card's name and a list of (similarity, name) pairs, most
      similar first.

    Raises:
      errors.MtgCubeBuilderError: If no card matches the name.
    """
    with open(paths.embeddings_dir / _CARDS_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    _, matrix = load_embeddings(paths)
    lowered = name.lower()
    matches = [i for i, r in enumerate(rows) if r["name"].lower() == lowered]
    if not matches:
        matches = [
            i for i, r in enumerate(rows) if lowered in r["name"].lower()
        ]
    if not matches:
        raise errors.MtgCubeBuilderError(f"No card matching {name!r}")
    target = matches[0]
    sims = matrix @ matrix[target]
    best = [j for j in np.argsort(-sims)[: k + 1] if j != target][:k]
    return rows[target]["name"], [
        (float(sims[j]), rows[j]["name"]) for j in best
    ]
