"""
Tests for cascade triple synthesis pipeline (Steps 1 and 3 of the
"how to continue" plan).

Validates:
  - scripts.build_union_pair_list canonicalizes unordered pairs and tags source
    correctly when alignment and geo inputs overlap.
  - scripts.report_cascade_triples emits exactly the (A, B, T) triple where
    BOTH the geo refined readout (cascade_call="A->B" and p_fwd_bonf <= alpha)
    AND ablation (direction call A→B, T == argmax_T mean_relay, relay > 0)
    agree. Other (A,B,T) candidates that fail any clause are rejected.
"""

import importlib.util
import json
import pickle
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Import the two scripts directly by path (they live under scripts/, which is
# not on the package path).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(module_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / rel_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def union_mod():
    return _load("build_union_pair_list", "scripts/build_union_pair_list.py")


@pytest.fixture(scope="module")
def triples_mod():
    return _load("report_cascade_triples", "scripts/report_cascade_triples.py")


# ---------------------------------------------------------------------------
# build_union_pair_list
# ---------------------------------------------------------------------------

def test_build_union_canonicalizes_and_merges(tmp_path, union_mod):
    # Alignment provides (B, A) unordered (lexicographic order would put A first).
    align_path = tmp_path / "top_pairs_cosine_24D.json"
    json.dump(
        [
            {"A": "IL-12", "B": "IFN-gamma",
             "relay_cell_type": "NK", "score": 0.71, "rank": 1},
            {"A": "IL-9",  "B": "IL-10",
             "relay_cell_type": "Treg", "score": 0.62, "rank": 2},
        ],
        open(align_path, "w"),
    )

    # Geo result: (IL-12, IFN-gamma) is A->B; (IL-2, IL-15) is also significant.
    sig_result = {
        "cascade_call": {
            ("IL-12", "IFN-gamma"): "A->B",
            ("IFN-gamma", "IL-12"): "B->A",
            ("IL-2",  "IL-15"):     "A->B",
            ("IL-15", "IL-2"):      "B->A",
            ("IL-4",  "IL-13"):     "none",
        },
        "relay_T": {
            ("IL-12", "IFN-gamma"): "NK",
            ("IL-2",  "IL-15"):     "CD4_T",
        },
        "p_fwd_bonf": {
            ("IL-12", "IFN-gamma", "NK"):     0.001,
            ("IL-12", "IFN-gamma", "CD4_T"):  0.4,
            ("IL-2",  "IL-15", "CD4_T"):      0.01,
        },
    }
    geo_path = tmp_path / "latent_geo_results.pkl"
    pickle.dump({"sig_result": sig_result}, open(geo_path, "wb"))

    out = tmp_path / "top_pairs_union.json"
    import sys
    sys.argv = [
        "build_union_pair_list",
        "--alignment_pairs_file", str(align_path),
        "--geo_results",           str(geo_path),
        "--output",                str(out),
    ]
    union_mod.main()

    rows = json.load(open(out))
    assert isinstance(rows, list) and rows, "union output should be non-empty list"

    # Every entry must have the keys ablation expects (A, B) plus our metadata.
    for r in rows:
        assert {"A", "B", "source", "relay_cell_type"}.issubset(r)
        # Canonical: A <= B lexicographically
        assert r["A"] <= r["B"], f"non-canonical pair: {r}"

    # IL-12 and IFN-gamma both surface -> source=="both"
    il12_pair = next(r for r in rows if r["A"] == "IFN-gamma" and r["B"] == "IL-12")
    assert il12_pair["source"] == "both"
    # geo_cascade_call must reflect the *canonical* direction. Canonical is
    # (IFN-gamma, IL-12), and geo said IL-12 -> IFN-gamma was "A->B", so the
    # canonical form should be "B->A".
    assert il12_pair["geo_cascade_call"] == "B->A"

    # IL-9 / IL-10: alignment only
    il9_pair = next(r for r in rows if r["A"] == "IL-10" and r["B"] == "IL-9")
    assert il9_pair["source"] == "alignment"
    assert il9_pair["geo_cascade_call"] is None

    # IL-2 / IL-15: geo only
    il2_pair = next(r for r in rows if r["A"] == "IL-15" and r["B"] == "IL-2")
    assert il2_pair["source"] == "geo"
    assert il2_pair["alignment_score"] is None

    # IL-4/IL-13 had cascade_call="none" → must NOT appear.
    assert not any(
        (r["A"], r["B"]) == ("IL-13", "IL-4") for r in rows
    ), "geo cascade_call='none' pairs should be excluded"


# ---------------------------------------------------------------------------
# report_cascade_triples — conjunction logic
# ---------------------------------------------------------------------------

def _make_seed_dir(
    seed_dir: Path,
    *,
    cascade_call: dict,
    p_fwd_bonf: dict,
    pooled: dict,
):
    seed_dir.mkdir(parents=True, exist_ok=True)
    sig = {
        "cascade_call": cascade_call,
        "p_fwd_bonf":   p_fwd_bonf,
        "relay_T":      {},  # not consulted by the script
    }
    pickle.dump({"sig_result": sig},
                open(seed_dir / "latent_geo_results.pkl", "wb"))
    pickle.dump({"all_scores": {seed_dir.name: pooled},
                 "pooled":     pooled},
                open(seed_dir / "ablation_scores_shard_0.pkl", "wb"))


def test_conjunction_emits_only_agreement_triple(tmp_path, triples_mod):
    seed = tmp_path / "exp_0_seed42"

    # Geo says (IL-12, IFN-gamma) is A->B with NK significant (p=0.001),
    # CD4_T not significant (p=0.4).
    # (IL-2, IL-15) is A->B with CD4_T significant.
    # (IL-1-beta, IL-6) is "none" — should never appear.
    cascade_call = {
        ("IL-12",     "IFN-gamma"): "A->B",
        ("IL-2",      "IL-15"):     "A->B",
        ("IL-1-beta", "IL-6"):      "none",
    }
    p_fwd_bonf = {
        ("IL-12",     "IFN-gamma", "NK"):     0.001,
        ("IL-12",     "IFN-gamma", "CD4_T"):  0.4,    # geo-nonsig
        ("IL-2",      "IL-15",     "CD4_T"):  0.01,
        ("IL-2",      "IL-15",     "Mono"):   0.5,
    }

    # Ablation pooled:
    #   IL-12 → IFN-gamma: NK is argmax (+0.20) > CD4_T (+0.05); reverse (B→A)
    #     max is much smaller (+0.02). Conjunction with NK should fire.
    #   IL-2 → IL-15: ablation argmax is Mono (+0.30), but geo Mono p=0.5 fails
    #     significance. Conjunction must NOT fire (T mismatch).
    pooled = {
        ("IL-12",     "IFN-gamma", "NK"):     [0.20, 0.18, 0.22],
        ("IL-12",     "IFN-gamma", "CD4_T"):  [0.05, 0.06, 0.04],
        ("IFN-gamma", "IL-12",     "NK"):     [0.02, 0.01, 0.03],
        ("IL-2",      "IL-15",     "CD4_T"):  [0.10, 0.09, 0.11],
        ("IL-2",      "IL-15",     "Mono"):   [0.30, 0.28, 0.32],  # geo-nonsig
        ("IL-15",     "IL-2",      "Mono"):   [0.01, 0.00, 0.02],
    }

    _make_seed_dir(seed,
                   cascade_call=cascade_call,
                   p_fwd_bonf=p_fwd_bonf,
                   pooled=pooled)

    rows = triples_mod._seed_triples(seed, alpha=0.05)
    keys = {(r["A"], r["B"], r["T"]) for r in rows}

    # Exactly one triple should survive
    assert keys == {("IL-12", "IFN-gamma", "NK")}, f"unexpected keys: {keys}"
    triple = rows[0]
    assert triple["p_fwd_bonf"] == pytest.approx(0.001)
    assert triple["ablation_relay"] == pytest.approx(0.20, abs=0.01)
    assert triple["geo_cascade_call"] == "A->B"
    assert triple["ablation_direction_call"] == "A→B"


def test_conjunction_rejects_when_ablation_calls_reverse(tmp_path, triples_mod):
    seed = tmp_path / "exp_1_seed123"
    # Geo says A→B but ablation says B→A (reverse stronger). Must not fire.
    cascade_call = {("A_cyto", "B_cyto"): "A->B"}
    p_fwd_bonf   = {("A_cyto", "B_cyto", "T_relay"): 0.001}
    pooled       = {
        ("A_cyto", "B_cyto", "T_relay"): [0.05, 0.04, 0.06],
        ("B_cyto", "A_cyto", "T_relay"): [0.50, 0.48, 0.52],
    }
    _make_seed_dir(seed, cascade_call=cascade_call,
                   p_fwd_bonf=p_fwd_bonf, pooled=pooled)
    assert triples_mod._seed_triples(seed, alpha=0.05) == []


def test_conjunction_rejects_when_relay_negative(tmp_path, triples_mod):
    seed = tmp_path / "exp_2_seed7"
    cascade_call = {("A_cyto", "B_cyto"): "A->B"}
    p_fwd_bonf   = {("A_cyto", "B_cyto", "T_relay"): 0.001}
    # Ablation forward is the argmax, but it is negative — removing T_relay
    # INCREASES P(B). That is the opposite of a relay.
    pooled = {
        ("A_cyto", "B_cyto", "T_relay"): [-0.05, -0.04, -0.06],
        ("B_cyto", "A_cyto", "T_relay"): [-0.20, -0.21, -0.22],
    }
    _make_seed_dir(seed, cascade_call=cascade_call,
                   p_fwd_bonf=p_fwd_bonf, pooled=pooled)
    # Forward best (-0.04) still > reverse best (-0.20), so direction = "A→B",
    # but mean relay <= 0 → reject.
    assert triples_mod._seed_triples(seed, alpha=0.05) == []


def test_build_union_alignment_top_k_cap(tmp_path, union_mod):
    """--alignment_top_k caps the alignment list to top-K entries by rank."""
    align_path = tmp_path / "top_pairs_inner_product_128D.json"
    json.dump(
        [
            {"A": "A1", "B": "B1", "relay_cell_type": "T", "score": 0.9, "rank": 1},
            {"A": "A2", "B": "B2", "relay_cell_type": "T", "score": 0.8, "rank": 2},
            {"A": "A3", "B": "B3", "relay_cell_type": "T", "score": 0.7, "rank": 3},
            {"A": "A4", "B": "B4", "relay_cell_type": "T", "score": 0.6, "rank": 4},
        ],
        open(align_path, "w"),
    )
    out = tmp_path / "union.json"
    import sys
    sys.argv = [
        "build_union_pair_list",
        "--alignment_pairs_file", str(align_path),
        "--alignment_top_k", "2",
        "--output",               str(out),
    ]
    union_mod.main()
    rows = json.load(open(out))
    assert len(rows) == 2, f"top_k=2 should keep 2 entries, got {len(rows)}"
    pairs = {(r["A"], r["B"]) for r in rows}
    # Canonical of (A1, B1) is (A1, B1) since A1 < B1; same for (A2, B2)
    assert pairs == {("A1", "B1"), ("A2", "B2")}


def test_aggregation_requires_min_seeds(tmp_path, triples_mod):
    """A triple seen in 1/3 seeds must not survive min_seeds=2."""
    triple_a = {"A": "IL-12", "B": "IFN-gamma", "T": "NK",
                "p_fwd_bonf": 0.001, "ablation_relay": 0.2,
                "geo_cascade_call": "A->B", "ablation_direction_call": "A→B",
                "seed": "exp_0_seed42"}
    triple_b = {"A": "IL-2", "B": "IL-15", "T": "CD4_T",
                "p_fwd_bonf": 0.01, "ablation_relay": 0.15,
                "geo_cascade_call": "A->B", "ablation_direction_call": "A→B",
                "seed": "exp_1_seed123"}
    # IL-12→IFN-gamma appears in 2 seeds; IL-2→IL-15 in 1.
    per_seed = [triple_a, {**triple_a, "seed": "exp_2_seed7"}, triple_b]
    df = triples_mod._aggregate(per_seed, min_seeds=2)
    assert len(df) == 1
    row = df.iloc[0]
    assert (row["A"], row["B"], row["T"]) == ("IL-12", "IFN-gamma", "NK")
    assert row["n_seeds"] == 2
