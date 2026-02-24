"""
CytokineLabel: consistent cytokine -> integer mapping with PBS fixed at index 90.
"""

import json
from typing import Dict, List

PBS_INDEX = 90


class CytokineLabel:
    """
    Builds a consistent cytokine -> integer mapping from a manifest.

    PBS is always mapped to index 90.
    All other cytokines are sorted alphabetically and assigned indices 0..N-1.
    The mapping is saved/loaded from JSON for reproducibility across runs.
    """

    def __init__(self) -> None:
        self._label_to_idx: Dict[str, int] = {}
        self._idx_to_label: Dict[int, str] = {}

    def fit(self, manifest: List[dict]) -> "CytokineLabel":
        """Build mapping from a list of manifest entries."""
        non_pbs = sorted(
            {entry["cytokine"] for entry in manifest if entry["cytokine"] != "PBS"}
        )
        self._label_to_idx = {cyt: idx for idx, cyt in enumerate(non_pbs)}
        self._label_to_idx["PBS"] = PBS_INDEX
        self._idx_to_label = {v: k for k, v in self._label_to_idx.items()}
        return self

    def encode(self, cytokine: str) -> int:
        return self._label_to_idx[cytokine]

    def decode(self, idx: int) -> str:
        return self._idx_to_label[idx]

    def n_classes(self) -> int:
        """
        Number of output units required by the classifier.

        Always equals PBS_INDEX + 1 (= 91) regardless of how many cytokines
        are actually present in the dataset. This keeps the output dimension
        consistent between demo (10 cytokines) and real (90 cytokines) data.
        """
        return PBS_INDEX + 1

    @property
    def cytokines(self) -> List[str]:
        """All cytokine names, sorted by index."""
        return [self._idx_to_label[i] for i in sorted(self._idx_to_label)]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._label_to_idx, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CytokineLabel":
        encoder = cls()
        with open(path) as f:
            encoder._label_to_idx = json.load(f)
        encoder._idx_to_label = {v: k for k, v in encoder._label_to_idx.items()}
        return encoder
