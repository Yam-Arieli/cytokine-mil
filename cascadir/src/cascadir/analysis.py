"""Analysis / benchmark — score direction calls against known-direction labels.

This is the evaluation layer (the M8 analysis of the source study): given a list of
pairs whose true upstream is known, it computes how often cross_asym got the
direction right, compares against the symmetric ``directional_score`` control
(expected near chance), and reports the classification breakdown and null-pass count.

Use it to turn a `direction_table` into the headline numbers (e.g. "5/6 = 83%,
all non-ambiguous correct, all clearing the null").
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from cascadir.types import BenchmarkResult


def _sign(x: float) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    return int(np.sign(x))


def score_directions(
    direction_table: pd.DataFrame,
    labels: list[tuple[str, str]],
    *,
    null_alpha: float = 0.05,
) -> BenchmarkResult:
    """Score a direction table against known ``(upstream, downstream)`` labels.

    Args:
        direction_table: output of ``CascadeDirection.direction_table()`` /
            ``cross_asym.direction_table`` — must have columns ``condition_a``,
            ``condition_b``, ``cross_asym_median``, ``directional_score_median``,
            ``classification`` and (optionally) ``null_p``.
        labels: list of ``(upstream, downstream)`` pairs encoding the ground-truth
            direction (the first element is upstream). Order within the tuple is the
            biology; the pair is matched to the table by its canonical (sorted) form.
        null_alpha: threshold for counting a call as null-passing.

    Returns:
        A :class:`BenchmarkResult` with cross_asym accuracy (overall and among
        non-AMBIGUOUS), the symmetric directional_score control accuracy, the
        classification breakdown, the null-pass count, and a per-pair table.
    """
    required = {
        "condition_a",
        "condition_b",
        "cross_asym_median",
        "directional_score_median",
        "classification",
    }
    missing = required - set(direction_table.columns)
    if missing:
        raise ValueError(
            f"direction_table is missing columns {sorted(missing)}; pass the output "
            "of cascadir direction_table()."
        )
    has_null = "null_p" in direction_table.columns

    # index rows by canonical pair
    by_pair: dict[tuple[str, str], pd.Series] = {}
    for _, row in direction_table.iterrows():
        by_pair[(str(row["condition_a"]), str(row["condition_b"]))] = row

    rows: list[dict] = []
    for up, down in labels:
        a, b = sorted([up, down])
        expected_sign = +1 if up == a else -1
        row = by_pair.get((a, b))
        if row is None:
            rows.append(
                {
                    "condition_a": a,
                    "condition_b": b,
                    "expected_upstream": up,
                    "found": False,
                    "classification": None,
                    "cross_asym_median": np.nan,
                    "called_upstream": None,
                    "cross_correct": None,
                    "directional_score_median": np.nan,
                    "dirscore_correct": None,
                    "null_p": np.nan,
                    "null_pass": None,
                }
            )
            continue
        cross_med = float(row["cross_asym_median"])
        ds_med = float(row["directional_score_median"])
        classification = str(row["classification"])
        cross_sign = _sign(cross_med)
        ds_sign = _sign(ds_med)
        # directional_score may be absent (NaN) when only cross_asym is supplied;
        # in that case it is simply not scored (None), not counted as wrong.
        ds_correct = None if np.isnan(ds_med) else bool(ds_sign == expected_sign)
        called_upstream = a if cross_sign > 0 else (b if cross_sign < 0 else None)
        null_p = float(row["null_p"]) if has_null and pd.notna(row["null_p"]) else np.nan
        rows.append(
            {
                "condition_a": a,
                "condition_b": b,
                "expected_upstream": up,
                "found": True,
                "classification": classification,
                "cross_asym_median": cross_med,
                "called_upstream": called_upstream,
                "cross_correct": bool(cross_sign == expected_sign),
                "directional_score_median": ds_med,
                "dirscore_correct": ds_correct,
                "null_p": null_p,
                "null_pass": (bool(null_p < null_alpha) if not np.isnan(null_p) else None),
            }
        )

    table = pd.DataFrame(rows)
    found = table[table["found"]]
    n_found = int(len(found))
    scored = found[found["classification"] != "AMBIGUOUS"]
    n_scored = int(len(scored))

    cross_accuracy = float(scored["cross_correct"].mean()) if n_scored else float("nan")
    cross_accuracy_all = (
        float(found["cross_correct"].mean()) if n_found else float("nan")
    )
    # directional_score accuracy only over rows that actually carry a dirscore value
    scored_ds = scored[scored["dirscore_correct"].notna()] if n_scored else scored
    dirscore_accuracy = (
        float(scored_ds["dirscore_correct"].astype(bool).mean())
        if len(scored_ds)
        else float("nan")
    )
    n_null_pass = int(scored["null_pass"].fillna(False).sum()) if n_scored else 0
    classification_counts = dict(
        Counter(found["classification"].tolist())
    )

    return BenchmarkResult(
        n_labeled=len(labels),
        n_found=n_found,
        n_scored=n_scored,
        cross_accuracy=cross_accuracy,
        cross_accuracy_all=cross_accuracy_all,
        dirscore_accuracy=dirscore_accuracy,
        n_null_pass=n_null_pass,
        classification_counts=classification_counts,
        table=table,
    )
