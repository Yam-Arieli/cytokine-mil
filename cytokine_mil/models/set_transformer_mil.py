"""
CytokineSelfAttnMIL: AB-MIL with a self-attention block over cells (CLAUDE.md §34).

Unlike CytokineABMIL (whose AttentionModule is a per-cell *pooling* op — cell i's
weight depends only on h_i, cells couple only through the softmax denominator), a
Set-Attention Block (SAB) lets every cell attend to every other cell:

    H' = SAB(H)      -> H'_i depends on ALL cells, and exposes an N x N matrix
                        A[i, j] = how much cell i attends to cell j.

The bag decision then uses the EXISTING AB-MIL pooling on the interacted cells:

    H  = encoder(X)              # (N, D) frozen — returned unchanged as 3rd output
    H' = SAB(H)                  # (N, D) cells interact; per-head A[i, j]
    a  = AttentionModule(H')     # (N,)  pooling weights over interacted cells
    z  = sum_i a_i H'_i ; y_hat = classifier(z)

forward(X) returns (y_hat, a, H) — the SAME contract as CytokineABMIL, so
train_mil runs unchanged (it dispatches on isinstance(model, CytokineABMIL_V2),
which is False here). H is the ORIGINAL frozen encoder embedding (not H') so the
centroid / PBS-RC logging in train_mil keeps its §33 semantics.

Extra methods for the frozen-encoder reconstruction extractor
(scripts/extract_selfattn_trajectory.py): pool_from_H, interaction_from_H,
readout_from_H, forward_with_interaction.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from cytokine_mil.models.attention import AttentionModule
from cytokine_mil.models.bag_classifier import BagClassifier
from cytokine_mil.models.instance_encoder import InstanceEncoder


class SelfAttentionBlock(nn.Module):
    """
    Set-Transformer Set-Attention Block (Lee et al. 2019), pre-LayerNorm variant.

        h  = LN1(X);  A = MultiheadAttention(h, h, h)
        u  = X + A(h)                    # residual after self-attention
        H' = u + FFN(LN2(u))             # residual after position-wise FFN

    Self-attention (Q=K=V) makes each cell's output a function of all cells. No
    dropout — attention-weight stability is required for the §34 trajectory readout.

    Input:  X in R^(N x embed_dim)     (one pseudo-tube's cell embeddings)
    Output: H' in R^(N x embed_dim), and optionally the per-head attention
            weights A in R^(n_heads x N x N) with rows summing to 1 over keys.
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, ffn_mult: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True, dropout=0.0,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_mult * embed_dim),
            nn.GELU(),
            nn.Linear(ffn_mult * embed_dim, embed_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.ffn:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the FFN output so the block starts near identity (matches the
        # encoder's identity-init convention; keeps initial loss near ln(K)).
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(
        self, X: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            X: (N, embed_dim) cell embeddings for one tube.
            return_attn: if True, also return per-head attention (n_heads, N, N).
        Returns:
            H_prime: (N, embed_dim) interacted cell embeddings.
            attn: (n_heads, N, N) per-head attention weights, or None.
        """
        h = self.ln1(X).unsqueeze(0)                     # (1, N, D)
        attn_out, attn_w = self.mha(
            h, h, h, need_weights=return_attn,
            average_attn_weights=False,
        )                                                # attn_out (1,N,D); attn_w (1,n_heads,N,N)|None
        u = X + attn_out.squeeze(0)                      # (N, D)
        H_prime = u + self.ffn(self.ln2(u))              # (N, D)
        attn = attn_w.squeeze(0) if (return_attn and attn_w is not None) else None
        return H_prime, attn


class CytokineSelfAttnMIL(nn.Module):
    """
    Full pipeline: InstanceEncoder -> SelfAttentionBlock(s) -> AttentionModule
    (AB-MIL pooling) -> BagClassifier.

    Same public interface as CytokineABMIL (forward -> (y_hat, a, H); .encoder;
    .encoder_frozen; unfreeze_encoder) so train_mil needs no changes.

    Input:  X in R^(N x G)
    Output: y_hat in R^K, a in R^N (pooling weights, sum to 1),
            H in R^(N x embed_dim) (ORIGINAL frozen encoder embeddings).
    """

    def __init__(
        self,
        encoder: InstanceEncoder,
        attention: AttentionModule,
        classifier: BagClassifier,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        encoder_frozen: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.sab_layers = nn.ModuleList(
            [SelfAttentionBlock(embed_dim=embed_dim, n_heads=n_heads)
             for _ in range(max(1, n_layers))]
        )
        self.attention = attention   # AB-MIL pooling over interacted cells
        self.classifier = classifier
        self.encoder_frozen: bool = encoder_frozen  # tracked for _build_optimizer
        if encoder_frozen:
            self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder_frozen = True

    def unfreeze_encoder(self) -> None:
        """Enable encoder weight updates (Stage 3 fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder_frozen = False

    # -- internal: run the SAB stack on cell embeddings H ---------------------
    def _run_sab(
        self, H: torch.Tensor, return_attn: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the SAB stack to H. Returns (H', first-layer per-head attention).

        The interaction matrix is taken from the FIRST SAB layer (operating
        directly on the encoder embeddings), which is the interpretable
        "which cell attends to which"; deeper layers only refine H'.
        """
        first_attn = None
        H_prime = H
        for i, sab in enumerate(self.sab_layers):
            want = return_attn and (i == 0)
            H_prime, attn = sab(H_prime, return_attn=want)
            if want:
                first_attn = attn
        return H_prime, first_attn

    def forward(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            X: (N, G) pseudo-tube expression matrix (one tube at a time).
        Returns:
            y_hat: (K,) class logits.
            a:     (N,) AB-MIL pooling weights over interacted cells.
            H:     (N, embed_dim) ORIGINAL frozen encoder embeddings.
        """
        H = self.encoder(X)                              # (N, D) frozen
        H_prime, _ = self._run_sab(H, return_attn=False)  # (N, D) interacted
        a = self.attention(H_prime)                       # (N,)
        z_tube = (a.unsqueeze(1) * H_prime).sum(dim=0)    # (D,)
        y_hat = self.classifier(z_tube)                   # (K,)
        return y_hat, a, H

    def forward_with_interaction(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extended forward returning the head-averaged cell x cell interaction.

        Returns (y_hat, a, H, A) where A: (N, N) is the first-SAB-layer
        head-averaged attention (rows sum to 1 over keys).
        """
        H = self.encoder(X)
        H_prime, attn = self._run_sab(H, return_attn=True)
        a = self.attention(H_prime)
        z_tube = (a.unsqueeze(1) * H_prime).sum(dim=0)
        y_hat = self.classifier(z_tube)
        A = attn.mean(dim=0) if attn is not None else None   # (N, N)
        return y_hat, a, H, A

    # -- extractor helpers: operate on cached (frozen) embeddings H -----------
    def readout_from_H(
        self, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """From cached frozen H, run the SAB+pool once and return both readouts.

        Returns (a, A): a (N,) pooling weights; A (N, N) head-averaged
        cell x cell attention. Used by the frozen-encoder trajectory extractor.
        """
        H_prime, attn = self._run_sab(H, return_attn=True)
        a = self.attention(H_prime)
        A = attn.mean(dim=0) if attn is not None else None
        return a, A

    def pool_from_H(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """From cached frozen H, return (a, H') — pooling weights and interacted H."""
        H_prime, _ = self._run_sab(H, return_attn=False)
        return self.attention(H_prime), H_prime

    def interaction_from_H(self, H: torch.Tensor) -> torch.Tensor:
        """From cached frozen H, return the head-averaged cell x cell attention (N, N)."""
        _, attn = self._run_sab(H, return_attn=True)
        return attn.mean(dim=0)
