"""
InstanceEncoder: maps single-cell expression vectors to dense embeddings.

Pre-trained with cell-type supervision (Stage 1). After pre-training, only
the backbone is used — the cell_type_head is never seen by the MIL model.
"""

from typing import Optional

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Same-dimension residual block: LayerNorm -> Linear -> GELU -> Linear + skip."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + h


class _DownBlock(nn.Module):
    """Downsampling residual block with a linear skip projection."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return self.skip(x) + h


class InstanceEncoder(nn.Module):
    """
    MLP encoder: maps single-cell expression -> dense embedding.

    Pre-trained with cell-type supervision before MIL training.
    Architecture: input projection + two ResBlocks with downsampling.

    Input:  x_i in R^G  (G = number of HVGs)
    Output: h_i in R^embed_dim  (default 128)

    The cell_type_head attribute (if n_cell_types is given) is used only
    during Stage 1 pre-training and is never passed to the MIL model.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        n_cell_types: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self._build_layers(input_dim, embed_dim)
        if n_cell_types is not None:
            self.cell_type_head = nn.Linear(embed_dim, n_cell_types)
        else:
            self.cell_type_head = None
        self._init_weights()

    def _build_layers(self, input_dim: int, embed_dim: int) -> None:
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        self.res1 = _ResBlock(512)
        self.down1 = _DownBlock(512, 256)
        self.res2 = _ResBlock(256)
        self.down2 = _DownBlock(256, embed_dim)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the last linear in each residual block so every block
        # starts as an identity mapping — prevents variance from compounding
        # across skip connections and keeps the initial loss near ln(n_classes).
        for m in [self.res1, self.res2, self.down1, self.down2]:
            nn.init.zeros_(m.fc2.weight)
            nn.init.zeros_(m.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, G) cell expression matrix.
        Returns:
            h: (N, embed_dim) cell embeddings.
        """
        h = self.input_proj(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        h = self.down2(h)
        return h
