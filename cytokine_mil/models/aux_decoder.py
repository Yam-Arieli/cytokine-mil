"""
AuxDecoder: post-hoc MLP that injects cytokine geometry into cell-level embeddings.

Trained on frozen encoder/MIL output. Supervised by sharpened bag-level softmax
(temperature τ). Used in Experiment 3 (latent geometry analysis) when Experiment 0
alignment gate fails.

Loss (uniform KL — attention-weighted was empirically disqualified, see CLAUDE.md §20.8):
    L = (1/N) sum_i KL( softmax(y_hat/τ) || softmax(decoder(h_i)) )

After training, g_i = decoder.embed(h_i) ∈ R^64 replaces h_i ∈ R^128 as the
cell-level embedding for Experiments 1 and 2 in analysis/latent_geometry.py.
"""

import torch
import torch.nn as nn


class AuxDecoder(nn.Module):
    """
    Post-hoc MLP: h_i ∈ R^input_dim → g_i ∈ R^embed_dim → cytokine_logits ∈ R^n_classes

    Encoder and MIL model are frozen during training.
    g_i (embed_dim intermediate) is the cytokine-aware cell embedding used in
    Experiments 1 and 2 of the latent geometry analysis.

    Scientific claim: decoder-injected cytokine geometry, not emergent encoder geometry.
    """

    def __init__(
        self,
        input_dim: int = 128,
        embed_dim: int = 64,
        n_classes: int = 91,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_classes = n_classes

        self.project = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(embed_dim, n_classes)
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
        g = self.project(h)
        return self.head(g)

    def embed(self, h: torch.Tensor) -> torch.Tensor:
        """
        Return g_i — the cytokine-aware cell embedding (decoder intermediate).

        Used as the embedding space for latent geometry Experiments 1 and 2
        after training. Replaces h_i ∈ R^128 with g_i ∈ R^embed_dim.

        Args:
            h: (N, input_dim) frozen encoder output.
        Returns:
            g: (N, embed_dim) cytokine-aware cell embeddings.
        """
        return self.project(h)
