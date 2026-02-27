"""
Reusable experiment setup helpers for different experimental scenarios.

Extracts the boilerplate that used to live inline in experiment.ipynb so that
variant notebooks (subset experiment, binary experiment, etc.) can import and
reuse it without copy-pasting.

Exported functions
------------------
build_stage1_manifest    — select one tube per cytokine (rotating donors)
filter_manifest          — keep only a subset of cytokines
make_binary_manifest     — one-vs-control 2-class manifest + BinaryLabel
split_manifest_by_donor  — donor-level train/val split
build_encoder            — construct InstanceEncoder from dimension params
build_mil_model          — wrap an encoder in a full CytokineABMIL pipeline
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from cytokine_mil.data.label_encoder import BinaryLabel
from cytokine_mil.models.attention import AttentionModule
from cytokine_mil.models.bag_classifier import BagClassifier
from cytokine_mil.models.cytokine_abmil import CytokineABMIL
from cytokine_mil.models.instance_encoder import InstanceEncoder


def build_stage1_manifest(
    manifest: List[dict],
    save_path: Optional[str] = None,
) -> List[dict]:
    """
    Select one tube per cytokine from the full manifest, rotating donors.

    For each cytokine (sorted alphabetically), picks the entry at donor index
    ``i % n_donors`` where ``i`` is the cytokine's sorted rank. This distributes
    Stage 1 training across donors without repetition.

    Only tube_idx == 0 entries are considered so the selection is deterministic
    regardless of how many pseudo-tubes exist per (donor, cytokine) pair.

    Args:
        manifest: Full manifest list loaded from manifest.json.
        save_path: If provided, writes the result to this path as JSON.
    Returns:
        List of manifest entries — one per cytokine (~91 for the full dataset).
    """
    cyt_to_entries: Dict[str, List[dict]] = defaultdict(list)
    for entry in manifest:
        if entry["tube_idx"] == 0:
            cyt_to_entries[entry["cytokine"]].append(entry)

    stage1_manifest = []
    for i, cyt in enumerate(sorted(cyt_to_entries)):
        entries = sorted(cyt_to_entries[cyt], key=lambda e: e["donor"])
        stage1_manifest.append(entries[i % len(entries)])

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(stage1_manifest, f)

    return stage1_manifest


def filter_manifest(
    manifest: List[dict],
    cytokines: List[str],
    include_pbs: bool = True,
) -> List[dict]:
    """
    Filter a manifest to a specific subset of cytokines.

    Args:
        manifest: Full manifest list.
        cytokines: Cytokine names to keep (do not include PBS here).
        include_pbs: If True, PBS entries are always included.
    Returns:
        Filtered manifest list.
    """
    keep = set(cytokines)
    if include_pbs:
        keep.add("PBS")
    return [e for e in manifest if e["cytokine"] in keep]


def make_binary_manifest(
    manifest: List[dict],
    target_cytokine: str,
    control: str = "PBS",
) -> Tuple[List[dict], BinaryLabel]:
    """
    Build a 2-class manifest for one-vs-control classification.

    Filters the manifest to tubes of ``target_cytokine`` and ``control`` only.
    Returns a BinaryLabel encoder so that target_cytokine → 0, control → 1.

    The returned label encoder has ``n_classes() == 2``, so pass
    ``n_classes=2`` to ``build_mil_model`` when using this manifest.

    Args:
        manifest: Full manifest list.
        target_cytokine: The cytokine to classify against the control.
        control: Control condition name (default: "PBS").
    Returns:
        (filtered_manifest, BinaryLabel)
    """
    filtered = [
        e for e in manifest if e["cytokine"] in {target_cytokine, control}
    ]
    label_encoder = BinaryLabel(positive=target_cytokine, negative=control)
    return filtered, label_encoder


def split_manifest_by_donor(
    manifest: List[dict],
    val_donors: List[str],
) -> Tuple[List[dict], List[dict]]:
    """
    Split a manifest into train and val sets at the donor level.

    Pseudo-tubes from the same donor are highly correlated (effective N = 12).
    Holding out at the donor level is the only valid generalization test.
    See CLAUDE.md Section 16 for scientific rationale and donor selection.

    Args:
        manifest: Full manifest list.
        val_donors: Donor names to hold out (e.g., ["Donor2", "Donor3"]).
    Returns:
        (train_manifest, val_manifest) where val_manifest contains all
        entries whose donor is in val_donors, and train_manifest contains
        the rest. Both retain the full set of cytokines.
    """
    val_set = set(val_donors)
    train_manifest = [e for e in manifest if e["donor"] not in val_set]
    val_manifest = [e for e in manifest if e["donor"] in val_set]
    return train_manifest, val_manifest


def build_encoder(
    n_input_genes: int,
    n_cell_types: int,
    embed_dim: int = 128,
) -> InstanceEncoder:
    """
    Construct an InstanceEncoder.

    Args:
        n_input_genes: Number of HVG input features.
        n_cell_types: Number of cell types for the Stage 1 classification head.
        embed_dim: Embedding dimension (default 128).
    Returns:
        Untrained InstanceEncoder.
    """
    return InstanceEncoder(
        input_dim=n_input_genes,
        embed_dim=embed_dim,
        n_cell_types=n_cell_types,
    )


def build_mil_model(
    encoder: InstanceEncoder,
    embed_dim: int = 128,
    attention_hidden_dim: int = 64,
    n_classes: int = 91,
    encoder_frozen: bool = True,
) -> CytokineABMIL:
    """
    Wrap an (optionally pre-trained) encoder in a full CytokineABMIL pipeline.

    Args:
        encoder: InstanceEncoder (typically returned by build_encoder and
            trained via train_encoder for Stage 2 setup).
        embed_dim: Must match encoder.embed_dim.
        attention_hidden_dim: Attention hidden dimension (default 64).
        n_classes: Number of output classes.
            91 for the full multi-class experiment (90 cytokines + PBS).
            2 for binary experiments (use with BinaryLabel).
        encoder_frozen: If True, encoder weights are frozen during Stage 2.
    Returns:
        CytokineABMIL ready for train_mil.
    """
    attention = AttentionModule(
        embed_dim=embed_dim,
        attention_hidden_dim=attention_hidden_dim,
    )
    classifier = BagClassifier(embed_dim=embed_dim, n_classes=n_classes)
    return CytokineABMIL(encoder, attention, classifier, encoder_frozen=encoder_frozen)
