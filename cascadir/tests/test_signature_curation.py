"""Promiscuous-gene curation: the null calibration and the removal semantics.

These lock two things down. (1) The cap is *calibrated*, not hand-picked — a fixed cap
means very different things at different (K, top_n, G), which is exactly the trap the
calibration exists to avoid. (2) Removal is global: a gene over the cap leaves EVERY
signature, including the one where it ranked first.
"""

from __future__ import annotations

import pytest

from cascadir.signatures import (
    curate_signatures,
    null_calibrated_max_occurrences,
    null_expected_removal,
    signature_gene_occurrences,
)
from cascadir.types import Signature


def _sig(condition: str, genes: list[str]) -> Signature:
    """A Signature with descending, deterministic ig_scores so alignment is checkable."""
    return Signature(
        condition=condition,
        genes=tuple(genes),
        ig_scores=tuple(float(len(genes) - i) for i in range(len(genes))),
        top_n=len(genes),
    )


# --- the null calibration --------------------------------------------------------


def test_null_expected_removal_is_a_probability_and_monotone_in_cap():
    vals = [null_expected_removal(90, 200, 4000, cap) for cap in range(1, 20)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert vals == sorted(vals, reverse=True)  # a larger cap removes no more


def test_null_expected_removal_matches_the_documented_reference_point():
    # ">3 of 24 at top-200" is the reference stringency the default target encodes.
    assert null_expected_removal(24, 200, 4000, cap=3) == pytest.approx(0.1052, abs=5e-4)


def test_calibration_reproduces_its_own_reference_and_scales_with_K():
    # Self-consistency: at the reference scale the calibrated cap IS 3.
    assert null_calibrated_max_occurrences(24, 200, 4000) == 3
    # At 90 conditions the same stringency needs a much larger cap — the whole point.
    assert null_calibrated_max_occurrences(90, 200, 4000) == 8
    assert null_calibrated_max_occurrences(90, 100, 4000) == 5


def test_calibrated_cap_never_exceeds_its_own_damage_budget():
    for k, n in [(24, 50), (45, 200), (90, 200), (90, 100)]:
        cap = null_calibrated_max_occurrences(k, n, 4000)
        assert null_expected_removal(k, n, 4000, cap) <= 0.1052
        # ...and it is the SMALLEST such cap.
        if cap > 1:
            assert null_expected_removal(k, n, 4000, cap - 1) > 0.1052


def test_calibration_rejects_a_nonsense_target():
    with pytest.raises(ValueError):
        null_calibrated_max_occurrences(90, 200, 4000, target_null_removal=1.5)


# --- curation semantics ----------------------------------------------------------


def test_gene_over_cap_is_removed_from_every_signature_including_its_top_ranked_one():
    # SHARED is rank 0 in cond_a and appears in 3 signatures; cap=2 => it must go
    # from ALL of them, cond_a included.
    sigs = {
        "a": _sig("a", ["SHARED", "A1", "A2"]),
        "b": _sig("b", ["B1", "SHARED", "B2"]),
        "c": _sig("c", ["C1", "C2", "SHARED"]),
        "d": _sig("d", ["D1", "D2", "D3"]),
    }
    assert signature_gene_occurrences(sigs)["SHARED"] == 3

    curated, report = curate_signatures(sigs, max_occurrences=2)

    assert set(curated) == {"a", "b", "c", "d"}
    for cond in ("a", "b", "c"):
        assert "SHARED" not in curated[cond].genes
        assert len(curated[cond].genes) == 2
    assert curated["d"].genes == ("D1", "D2", "D3")  # untouched
    assert report.set_index("condition").loc["a", "n_removed"] == 1
    assert report.set_index("condition").loc["d", "n_removed"] == 0


def test_gene_exactly_at_the_cap_is_kept():
    sigs = {
        "a": _sig("a", ["EDGE", "A1"]),
        "b": _sig("b", ["EDGE", "B1"]),
        "c": _sig("c", ["C1", "C2"]),
    }
    curated, _ = curate_signatures(sigs, max_occurrences=2)  # EDGE occurs exactly twice
    assert "EDGE" in curated["a"].genes
    assert "EDGE" in curated["b"].genes


def test_genes_and_ig_scores_stay_index_aligned():
    sigs = {
        "a": _sig("a", ["X", "KEEP_A", "Y"]),
        "b": _sig("b", ["X", "Y", "KEEP_B"]),
        "c": _sig("c", ["X", "Y", "KEEP_C"]),
    }
    before = dict(zip(sigs["a"].genes, sigs["a"].ig_scores))
    curated, _ = curate_signatures(sigs, max_occurrences=2)  # X and Y both go
    sig = curated["a"]
    assert sig.genes == ("KEEP_A",)
    assert len(sig.genes) == len(sig.ig_scores)
    assert sig.ig_scores[0] == before["KEEP_A"]  # the score followed its gene


def test_emptied_condition_is_dropped_and_reported_not_raised():
    # Signature.__post_init__ rejects an empty gene tuple, so a fully-curated-away
    # condition must be dropped rather than rebuilt.
    sigs = {
        "a": _sig("a", ["X", "Y"]),
        "b": _sig("b", ["X", "Y"]),
        "c": _sig("c", ["X", "Y"]),
        "d": _sig("d", ["D1", "D2"]),
    }
    curated, report = curate_signatures(sigs, max_occurrences=2)
    assert set(curated) == {"d"}
    dropped = set(report.loc[report["dropped"], "condition"])
    assert dropped == {"a", "b", "c"}
    assert len(report) == 4  # every INPUT condition is reported, dropped or not


def test_min_genes_drops_signatures_that_survive_but_are_too_small():
    sigs = {
        "a": _sig("a", ["X", "Y", "KEEP_A"]),
        "b": _sig("b", ["X", "Y", "B1", "B2", "B3"]),
        "c": _sig("c", ["X", "Y", "C1", "C2", "C3"]),
    }
    curated, report = curate_signatures(sigs, max_occurrences=2, min_genes=3)
    assert set(curated) == {"b", "c"}  # "a" keeps only 1 gene
    assert bool(report.set_index("condition").loc["a", "dropped"]) is True


def test_top_n_records_the_request_not_the_survivors():
    sigs = {
        "a": _sig("a", ["X", "A1", "A2"]),
        "b": _sig("b", ["X", "B1", "B2"]),
        "c": _sig("c", ["X", "C1", "C2"]),
    }
    curated, _ = curate_signatures(sigs, max_occurrences=2)
    assert curated["a"].top_n == 3
    assert len(curated["a"].genes) == 2


def test_a_cap_that_removes_nothing_is_an_identity():
    sigs = {"a": _sig("a", ["A1", "A2"]), "b": _sig("b", ["A1", "B2"])}
    curated, report = curate_signatures(sigs, max_occurrences=99)
    assert {c: s.genes for c, s in curated.items()} == {c: s.genes for c, s in sigs.items()}
    assert report["n_removed"].sum() == 0


@pytest.mark.parametrize("kwargs", [{"max_occurrences": 0}, {"max_occurrences": 2, "min_genes": 0}])
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        curate_signatures({"a": _sig("a", ["A1"])}, **kwargs)
