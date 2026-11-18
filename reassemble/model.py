"""Core data model shared by every stage of the pipeline.

Why RGBA everywhere?  A strip coming out of a shredder is not a perfect rectangle once it
has been photographed: edges fray, the strip is slightly rotated, and torn paper has a
completely irregular outline.  Storing the paper mask in the alpha channel lets every
later stage ask "is this pixel paper?" without special-casing the shred type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class Piece:
    """One physical fragment of paper.

    Attributes:
        id: stable identifier (random-looking, never encodes the true position).
        image: HxWx4 uint8 RGBA array; alpha > 0 means "paper".
        back: optional HxWx4 RGBA scan of the reverse side, stored *as you would read it*
            (i.e. the strip was flipped over its vertical axis before scanning).
        meta: free-form metadata (source file, capture info, ...).
    """

    id: str
    image: np.ndarray
    back: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def rgb(self) -> np.ndarray:
        return self.image[..., :3]

    @property
    def mask(self) -> np.ndarray:
        """Boolean paper mask."""
        return self.image[..., 3] > 127


@dataclass
class PairScore:
    """Cost of placing ``right`` immediately after ``left`` (lower is better).

    ``side`` is ``"lr"`` for left->right adjacency (strips) or ``"tb"`` for top->bottom
    adjacency (cross-cut).  ``confidence`` is in [0, 1].
    """

    left: str
    right: str
    cost: float
    side: str = "lr"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"left": self.left, "right": self.right, "cost": float(self.cost),
                "side": self.side, "confidence": float(self.confidence)}


@dataclass
class Reconstruction:
    """Result of a solve.

    For strip shreds ``order`` is a list of piece ids (left to right) and ``None`` entries
    mark detected gaps (probable missing pieces).  For cross-cut shreds ``grid`` is a list
    of rows of ids (``None`` = empty cell).  ``joins`` holds one entry per adjacent pair.
    """

    order: list[str | None] = field(default_factory=list)
    grid: list[list[str | None]] | None = None
    joins: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    score: float = 0.0
    method: str = ""
    runtime_s: float = 0.0
    documents: list[dict] | None = None  # for mixed bins: one sub-reconstruction per doc
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def piece_order(self) -> list[str]:
        """Order without gap markers."""
        return [p for p in self.order if p is not None]

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "grid": self.grid,
            "joins": self.joins,
            "gaps": self.gaps,
            "score": float(self.score),
            "method": self.method,
            "runtime_s": float(self.runtime_s),
            "documents": self.documents,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reconstruction":
        return cls(order=d.get("order", []), grid=d.get("grid"), joins=d.get("joins", []),
                   gaps=d.get("gaps", []), score=d.get("score", 0.0),
                   method=d.get("method", ""), runtime_s=d.get("runtime_s", 0.0),
                   documents=d.get("documents"), extra=d.get("extra", {}))


# --------------------------------------------------------------------------- I/O helpers


def to_rgba(img: np.ndarray) -> np.ndarray:
    """Return an RGBA uint8 copy of an RGB / gray / RGBA array (opaque if no alpha)."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        return img.astype(np.uint8)
    alpha = np.full(img.shape[:2] + (1,), 255, np.uint8)
    return np.concatenate([img.astype(np.uint8), alpha], axis=-1)


def save_rgba(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_rgba(img), "RGBA").save(path, optimize=False, compress_level=3)


def load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"))


def save_case(folder: str | Path, pieces: list[Piece], truth: dict | None = None) -> Path:
    """Write a case folder: ``pieces/<id>.png``, ``back/<id>.png`` and ``truth.json``."""
    folder = Path(folder)
    (folder / "pieces").mkdir(parents=True, exist_ok=True)
    for p in pieces:
        save_rgba(folder / "pieces" / f"{p.id}.png", p.image)
        if p.back is not None:
            save_rgba(folder / "back" / f"{p.id}.png", p.back)
    if truth is not None:
        (folder / "truth.json").write_text(json.dumps(truth, indent=1))
    return folder


def load_case(folder: str | Path) -> tuple[list[Piece], dict | None]:
    """Load pieces (and ground truth if present) from a case folder.

    A bare folder of PNGs (no ``pieces/`` subfolder) is also accepted, which is what the
    capture stage produces when pointed at an arbitrary output directory.
    """
    folder = Path(folder)
    piece_dir = folder / "pieces" if (folder / "pieces").is_dir() else folder
    pieces = []
    for f in sorted(piece_dir.glob("*.png")):
        back_f = folder / "back" / f.name
        back = load_rgba(back_f) if back_f.exists() else None
        pieces.append(Piece(id=f.stem, image=load_rgba(f), back=back, meta={"file": str(f)}))
    truth_f = folder / "truth.json"
    truth = json.loads(truth_f.read_text()) if truth_f.exists() else None
    return pieces, truth
