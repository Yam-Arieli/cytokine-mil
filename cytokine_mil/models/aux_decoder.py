"""
AuxDecoder: post-hoc MLP that injects cytokine geometry into cell-level embeddings.

Trained on frozen encoder/MIL output, supervised by bag-level softmax (ŷ).
Used in Experiment 3 (latent geometry analysis) when Experiment 0 alignment
gate fails.

Architecture:
    h_i ∈ R^input_dim  →  Linear(input_dim, hidden_dim) → ReLU
                        →  Linear(hidden_dim, n_classes) → cytokine_logits_i

The hidden layer output (hidden_dim intermediate) is exposed via embed() and
used as the cytokine-aware cell embedding g_i ∈ R^hidden_dim for Experiments
1 and 2 in analysis/latent_geometry.py.

Scientific claim: decoder-injected cytokine geometry, not emergent encoder
geometry. After training, g_i replaces h_i as the embedding space for
directional bias analysis.
"""

import torch
import torch.nn as nn


class AuxDecoder(nn.Module):
    """
    Post-hoc MLP: h_i ∈ R^input_dim → g_i ∈ R^hidden_dim → logits ∈ R^n_classes

    Encoder and MIL model are frozen during training.
    g_i (hidden_dim intermediate) is the cytokine-aware cell embedding used in
    Experiments 1 and 2 of the latent geometry analysis.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        n_classes: int = 91,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, input_dim) cell encoder embeddings (frozen encoder output).
        Returns:
            logits: (N, n_classes) per-cell cytokine logits.
        """
        return self.head(self.project(h))

    def embed(self, h: torch.Tensor) -> torch.Tensor:
        """
        Return g_i — the cytokine-aware cell embedding (decoder intermediate).

        Used as the embedding space for latent geometry Experiments 1 and 2
        after training. Replaces h_i ∈ R^128 with g_i ∈ R^hidden_dim.

        Args:
            h: (N, input_dim) frozen encoder output.
        Returns:
            g: (N, hidden_dim) cytokine-aware cell embeddings.
        """
        return self.project(h)
