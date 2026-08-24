"""
On-disk shards for a `cascadir.PseudoTubeSet` (CLAUDE.md §29 staged fit).

Why this exists
---------------
The full-90 Oesinghaus run splits one cascadir fit across a SLURM DAG: the pseudo-tubes
are materialised ONCE, then a chunked GPU array trains ten binary models per task and the
analysis jobs read the whole set back. Every chunk must see *byte-identical* tubes and the
*same* Stage-1 encoder — that is the §27.6 lesson, and rebuilding tubes per task would
break it (`cascadir.build_pseudotubes` advances a single RNG over the sorted
(condition, donor) pairs, so a condition subset yields different tubes for the same pair).

Shards are keyed by ``(donor, condition)`` so a chunk task can load only its own
conditions plus the control — ~7 GB instead of the full ~64 GB.

This module is **pure I/O**: it serialises and reconstructs cascadir's own dataclasses and
computes no statistics of any kind. All method math stays in `cascadir`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from cascadir.types import PseudoTube, PseudoTubeSet

META_NAME = "meta.json"


def _shard_name(donor: str, condition: str) -> str:
    """Filesystem-safe ``donor__condition`` stem (cytokine names carry '/' and spaces)."""
    safe = lambda s: "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))
    return f"{safe(donor)}__{safe(condition)}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ShardWriter:
    """Incremental writer — emits one ``.npy`` per (donor, condition) group as it goes.

    The full-90 tube set is ~64 GB, far more than the prepare job should ever hold at
    once, so tubes are materialised one (donor, condition) group at a time (~70 MB) and
    handed straight to :meth:`add_group`. Call :meth:`finalize` to write `meta.json`.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.shards: List[dict] = []
        self._seen: set = set()

    def add_group(self, donor: str, condition: str, tubes: Sequence[PseudoTube]) -> dict:
        """Write one (donor, condition) group's tubes as a single ``.npy`` shard."""
        if not tubes:
            raise ValueError(f"add_group: no tubes for ({donor!r}, {condition!r})")
        key = (str(donor), str(condition))
        if key in self._seen:
            raise ValueError(f"add_group: ({donor!r}, {condition!r}) written twice")
        self._seen.add(key)

        tubes = sorted(tubes, key=lambda t: t.tube_idx)
        stem = _shard_name(donor, condition)
        path = self.out / f"{stem}.npy"
        np.save(path, np.concatenate([t.X for t in tubes], axis=0).astype(np.float32))
        entry = {
            "donor": str(donor),
            "condition": str(condition),
            "file": path.name,
            "sha256": _sha256_file(path),
            "tubes": [
                {
                    "tube_idx": int(t.tube_idx),
                    "n_cells": int(t.X.shape[0]),
                    "cell_types": list(t.cell_types),
                    "cell_types_included": list(t.cell_types_included),
                }
                for t in tubes
            ],
        }
        self.shards.append(entry)
        return entry

    def finalize(self, gene_names: Sequence[str], control_label: str) -> dict:
        """Write `meta.json` describing every shard written so far."""
        if not self.shards:
            raise ValueError("finalize: no shards were written")
        conditions = sorted({s["condition"] for s in self.shards})
        if control_label not in conditions:
            raise ValueError(
                f"finalize: control {control_label!r} has no shard; cross_asym needs the "
                f"control baseline. Conditions written: {conditions}"
            )
        cell_types: set = set()
        n_tubes = 0
        for s in self.shards:
            n_tubes += len(s["tubes"])
            for t in s["tubes"]:
                cell_types.update(t["cell_types_included"])
        shards = sorted(self.shards, key=lambda s: (s["donor"], s["condition"]))
        meta = {
            "gene_names": [str(g) for g in gene_names],
            "control_label": control_label,
            "n_genes": len(gene_names),
            "n_tubes": n_tubes,
            "conditions": conditions,
            "donors": sorted({s["donor"] for s in shards}),
            "cell_types": sorted(cell_types),
            "shards": shards,
        }
        meta["shards_sha256"] = hashlib.sha256(
            json.dumps([s["sha256"] for s in shards], sort_keys=True).encode()
        ).hexdigest()
        with open(self.out / META_NAME, "w") as fh:
            json.dump(meta, fh)
        return meta


def save_tube_shards(tube_set: PseudoTubeSet, out_dir: str | Path) -> dict:
    """Write an in-memory `tube_set` as shards (convenience wrapper on :class:`ShardWriter`).

    Only suitable when the whole set fits in memory; the full-90 prepare job uses
    :class:`ShardWriter` directly.
    """
    groups: Dict[tuple, List[PseudoTube]] = {}
    for t in tube_set.tubes:
        groups.setdefault((t.donor, t.condition), []).append(t)
    writer = ShardWriter(out_dir)
    for (donor, condition), tubes in sorted(groups.items()):
        writer.add_group(donor, condition, tubes)
    return writer.finalize(tube_set.gene_names, tube_set.control_label)


def read_meta(shard_dir: str | Path) -> dict:
    with open(Path(shard_dir) / META_NAME) as fh:
        return json.load(fh)


def load_tube_set(
    shard_dir: str | Path,
    conditions: Optional[Sequence[str]] = None,
    donors: Optional[Sequence[str]] = None,
    include_control: bool = True,
    verify_sha: bool = False,
    tube_indices: Optional[Sequence[int]] = None,
) -> PseudoTubeSet:
    """Reconstruct a `PseudoTubeSet` from shards, optionally restricted.

    Args:
        shard_dir: Directory written by :func:`save_tube_shards`.
        conditions: Keep only these conditions (``None`` = all). The control is added
            back unless ``include_control=False``.
        donors: Keep only these donors (``None`` = all).
        include_control: Always load the control condition's tubes (cross_asym needs the
            PBS baseline).
        tube_indices: Keep only these ``tube_idx`` values within every (donor, condition)
            group (``None`` = all). The same indices are applied to every group, including
            the control, so the subset carries no per-condition choice. This is how the
            fixed main/reserve tube split is taken.
        verify_sha: Re-hash each loaded shard and check it against `meta.json`. Off by
            default (hashing 64 GB is slow); the DAG verifies `shards_sha256` instead.

    Returns:
        A `PseudoTubeSet` whose tubes are exactly those written by the producing job.
    """
    d = Path(shard_dir)
    meta = read_meta(d)
    control = meta["control_label"]

    keep_cond = None
    if conditions is not None:
        keep_cond = set(map(str, conditions))
        if include_control:
            keep_cond.add(control)
    keep_donor = set(map(str, donors)) if donors is not None else None
    keep_idx = set(map(int, tube_indices)) if tube_indices is not None else None

    tubes: List[PseudoTube] = []
    for shard in meta["shards"]:
        if keep_cond is not None and shard["condition"] not in keep_cond:
            continue
        if keep_donor is not None and shard["donor"] not in keep_donor:
            continue
        path = d / shard["file"]
        if verify_sha and _sha256_file(path) != shard["sha256"]:
            raise ValueError(f"shard {path} sha256 mismatch — tubes are not the ones written")
        X = np.load(path)
        start = 0
        for spec in shard["tubes"]:
            n = int(spec["n_cells"])
            if keep_idx is None or int(spec["tube_idx"]) in keep_idx:
                # Copy when subsetting: a view would pin the whole shard array in memory
                # for the sake of a few of its tubes.
                block = X[start : start + n]
                tubes.append(
                    PseudoTube(
                        X=block if keep_idx is None else np.ascontiguousarray(block),
                        condition=shard["condition"],
                        donor=shard["donor"],
                        cell_types=tuple(spec["cell_types"]),
                        cell_types_included=tuple(spec["cell_types_included"]),
                        tube_idx=int(spec["tube_idx"]),
                    )
                )
            start += n
        if start != X.shape[0]:
            raise ValueError(
                f"shard {path} has {X.shape[0]} rows but meta accounts for {start}"
            )

    if not tubes:
        raise ValueError(
            f"load_tube_set: no shards matched conditions={conditions} donors={donors} "
            f"tube_indices={tube_indices} under {d}"
        )
    return PseudoTubeSet(
        tubes=tubes,
        gene_names=tuple(meta["gene_names"]),
        control_label=control,
    )


# ---------------------------------------------------------------------------
# Frozen-encoder embedding cache (the encoded pseudo-tubes)
# ---------------------------------------------------------------------------

EMBED_META_NAME = "embeddings_meta.json"


def save_embedding_cache(cache: dict, out_dir: str | Path) -> dict:
    """Persist a `cascadir.build_frozen_embedding_cache` result.

    The cache is ``{(condition, donor, tube_idx): H}`` with ``H`` a ``(n_cells,
    embed_dim)`` tensor. Tubes are grouped into one ``.npy`` per (donor, condition), the
    same sharding the expression tubes use, so a chunk task can load only its own
    conditions. Pure I/O — no statistics.

    Returns the meta dict (also written to ``embeddings_meta.json``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups: Dict[tuple, List[tuple]] = {}
    for (condition, donor, tube_idx), h in cache.items():
        groups.setdefault((str(donor), str(condition)), []).append((int(tube_idx), h))

    shards: List[dict] = []
    embed_dim: Optional[int] = None
    for (donor, condition), items in sorted(groups.items()):
        items.sort(key=lambda kv: kv[0])
        arrs = [np.asarray(h.detach().cpu().numpy() if hasattr(h, "detach") else h,
                           dtype=np.float32) for _, h in items]
        if embed_dim is None:
            embed_dim = int(arrs[0].shape[1])
        block = np.concatenate(arrs, axis=0)
        fname = f"{_shard_name(donor, condition)}.npy"
        np.save(out / fname, block)
        shards.append({
            "file": fname,
            "donor": donor,
            "condition": condition,
            "sha256": _sha256_file(out / fname),
            "tubes": [
                {"tube_idx": idx, "n_cells": int(a.shape[0])}
                for (idx, _), a in zip(items, arrs)
            ],
        })

    meta = {
        "n_shards": len(shards),
        "n_tubes": sum(len(s["tubes"]) for s in shards),
        "embed_dim": embed_dim,
        "shards": shards,
        "shards_sha256": hashlib.sha256(
            "".join(s["sha256"] for s in sorted(shards, key=lambda s: s["file"])).encode()
        ).hexdigest(),
    }
    with open(out / EMBED_META_NAME, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def read_embedding_meta(embed_dir: str | Path) -> dict:
    with open(Path(embed_dir) / EMBED_META_NAME) as fh:
        return json.load(fh)


def load_embedding_cache(
    embed_dir: str | Path,
    conditions: Optional[Sequence[str]] = None,
    control_label: Optional[str] = None,
) -> dict:
    """Reload an embedding cache in the shape `train_all_binary(embedding_cache=...)` wants.

    Args:
        embed_dir: Directory written by :func:`save_embedding_cache`.
        conditions: Keep only these conditions (``None`` = all).
        control_label: Always kept when ``conditions`` is given (every binary model needs
            the control tubes).

    Returns:
        ``{(condition, donor, tube_idx): torch.Tensor}``.
    """
    import torch  # local: keeps this module importable without torch for pure I/O use

    d = Path(embed_dir)
    meta = read_embedding_meta(d)
    keep = None
    if conditions is not None:
        keep = set(map(str, conditions))
        if control_label is not None:
            keep.add(str(control_label))

    cache: dict = {}
    for shard in meta["shards"]:
        if keep is not None and shard["condition"] not in keep:
            continue
        block = np.load(d / shard["file"])
        start = 0
        for spec in shard["tubes"]:
            n = int(spec["n_cells"])
            key = (shard["condition"], shard["donor"], int(spec["tube_idx"]))
            cache[key] = torch.from_numpy(
                np.ascontiguousarray(block[start : start + n])
            )
            start += n
        if start != block.shape[0]:
            raise ValueError(
                f"embedding shard {shard['file']} has {block.shape[0]} rows but meta "
                f"accounts for {start}"
            )
    if not cache:
        raise ValueError(
            f"load_embedding_cache: no shards matched conditions={conditions} under {d}"
        )
    return cache
