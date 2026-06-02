"""Tests for cascadir.coupling — Path A axis discovery + the axis-table collapse."""

from __future__ import annotations

from cascadir import build_pseudotubes, discover_axes, preprocess, train_encoder
from cascadir.coupling import build_axis_table


def test_build_axis_table_collapse():
    """A hand-made significance dict collapses to the expected unordered axis row."""
    sig = {
        "cascade_call": {("A", "B"): "A->B", ("B", "A"): "none"},
        "relay_T": {("A", "B"): "Mono", ("B", "A"): "NK"},
        "p_pair_fwd": {("A", "B"): 0.01, ("B", "A"): 0.5},
        "W_pair_fwd": {("A", "B"): 6.0, ("B", "A"): 1.0},
    }
    df = build_axis_table(sig, control_label="PBS")
    assert len(df) == 1
    r = df.iloc[0]
    assert r["axis_a"] == "A" and r["axis_b"] == "B"
    assert bool(r["coupled"]) is True
    assert r["axis_strength"] == 6.0
    assert r["coupling_call"] == "a_to_b"
    assert r["dominant_direction"] == "a_to_b"
    assert r["relay_T"] == "Mono"


def test_build_axis_table_excludes_control():
    sig = {
        "cascade_call": {("A", "PBS"): "A->B", ("A", "B"): "none"},
        "relay_T": {},
        "p_pair_fwd": {("A", "PBS"): 0.01, ("A", "B"): 0.5},
        "W_pair_fwd": {("A", "PBS"): 9.0, ("A", "B"): 2.0},
    }
    df = build_axis_table(sig, control_label="PBS")
    # the A/PBS pair must be dropped; only A/B remains
    assert "PBS" not in set(df["axis_a"]) | set(df["axis_b"])
    assert list(zip(df["axis_a"], df["axis_b"])) == [("A", "B")]


def test_discover_axes_runs_and_flags_underpower(synthetic_adata):
    proc = preprocess(synthetic_adata, assume="raw")
    ts = build_pseudotubes(
        proc,
        condition_col="cytokine",
        donor_col="donor",
        celltype_col="cell_type",
        control_label="PBS",
        n_per_cell_type=20,
        min_cells=8,
        n_tubes=4,
        seed=0,
    )
    enc = train_encoder(proc, celltype_col="cell_type", epochs=3, device="cpu", seed=0)
    res = discover_axes(ts, enc, device="cpu")

    assert res.n_donors == 3
    assert res.underpowered is True  # 3 donors < advisory floor of 8
    pairs = set(zip(res.axes["axis_a"], res.axes["axis_b"]))
    assert ("CytA", "CytB") in pairs
    assert "PBS" not in set(res.axes["axis_a"]) | set(res.axes["axis_b"])
    for col in (
        "axis_a",
        "axis_b",
        "coupled",
        "axis_strength",
        "coupling_call",
        "dominant_direction",
        "p_fwd",
        "p_rev",
        "relay_T",
    ):
        assert col in res.axes.columns
    assert "n_donors" in res.summary()
