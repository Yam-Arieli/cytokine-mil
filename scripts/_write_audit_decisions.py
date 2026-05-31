"""
Author the audit_decisions.yaml file programmatically so every quote is
guaranteed to be a verbatim substring of the source evidence_summary
(or, for R5 pre-registered entries, the literal pre-reg note).

Usage:
    python scripts/_write_audit_decisions.py

This is the *hand-audit-as-data* artifact. Every per-direction decision
is recorded with the rule_id, the verbatim supporting quote, and the
citation count. The validator (validate_audit.py) will re-check that
each quote is in fact a substring of the corresponding source summary.

Decisions made by Yam after reading the full evidence digest. See
reports/cascade_pairs/audit_digest.csv for the source material.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DIGEST_PATH = REPO / "reports/cascade_pairs/audit_digest.csv"
OUT_PATH = REPO / "reports/cascade_pairs/audit_decisions.yaml"


# ----------------------------------------------------------------------- #
# Per-direction decision template helpers
# ----------------------------------------------------------------------- #

NOT_MENTIONED = "(direction not mentioned in summary)"
PRE_REG_QUOTE = "Pre-registered KNOWN_CASCADE"


def positive_strong(quote: str, n_primary: int, n_secondary: int = 0, immune: bool = True):
    return dict(
        label="POSITIVE_STRONG", rule_id="R1", quote=quote,
        n_citations_primary=n_primary, n_citations_secondary=n_secondary,
        lit_T_immune=immune,
    )


def positive_weak(quote: str, n_primary: int = 0, n_secondary: int = 0, immune=True):
    return dict(
        label="POSITIVE_WEAK", rule_id="R2", quote=quote,
        n_citations_primary=n_primary, n_citations_secondary=n_secondary,
        lit_T_immune=immune,
    )


def inhibitory(quote: str, n_primary: int = 0, n_secondary: int = 0):
    return dict(
        label="INHIBITORY", rule_id="R3", quote=quote,
        n_citations_primary=n_primary, n_citations_secondary=n_secondary,
        lit_T_immune=True,
    )


def unknown(quote: str = NOT_MENTIONED, n_primary: int = 0, n_secondary: int = 0):
    return dict(
        label="UNKNOWN", rule_id="R4", quote=quote,
        n_citations_primary=n_primary, n_citations_secondary=n_secondary,
        lit_T_immune=None,
    )


def pre_registered():
    return dict(
        label="POSITIVE_STRONG", rule_id="R5", quote=PRE_REG_QUOTE,
        n_citations_primary=0, n_citations_secondary=0,
        lit_T_immune=True,
    )


# ----------------------------------------------------------------------- #
# Per-axis decisions
# (axis_a, axis_b) → {"a_to_b": {...}, "b_to_a": {...}}
# Keys must match canonical (axis_a < axis_b alphabetically) from digest.
# ----------------------------------------------------------------------- #

DECISIONS = {
    # [00] CD30L / IL-17A — KNOWN_DIRECTIONAL a_to_b
    ("CD30L", "IL-17A"): {
        "a_to_b": positive_strong(
            "CD30 stimulation directly induces IL-17 production in human cells",
            n_primary=2, n_secondary=0,
        ),
        "b_to_a": unknown(),
    },
    # [01] CD30L / IL-27 — NOVEL
    ("CD30L", "IL-27"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "no direct evidence found that IL-27 induces CD30L/TNFSF8 expression on monocytes"
        ),
    },
    # [02] CD30L / IL-9 — NOVEL
    ("CD30L", "IL-9"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "no published evidence IL-9 induces CD30L expression on HSPCs"
        ),
    },
    # [03] CD30L / VEGF — NOVEL
    ("CD30L", "VEGF"): {
        "a_to_b": unknown(
            "no clear literature was found establishing CD30L→VEGF induction in CD14+ monocytes"
        ),
        "b_to_a": unknown(),
    },
    # [04] Decorin / IL-16 — NOVEL
    ("Decorin", "IL-16"): {
        "a_to_b": unknown(
            "no published evidence found linking decorin to IL-16 induction"
        ),
        "b_to_a": unknown(),
    },
    # [05] Decorin / IL-27 — NOVEL
    ("Decorin", "IL-27"): {
        "a_to_b": unknown(
            "no direct evidence was found linking decorin to IL-27 induction"
        ),
        "b_to_a": unknown(),
    },
    # [06] Decorin / IL-6 — KNOWN_DIRECTIONAL a_to_b
    # Evidence: "IL-6 directly induces decorin mRNA in human endothelial cells"
    # Canonical: Decorin=a, IL-6=b. Summary describes IL-6 → Decorin = b_to_a.
    # Cell type: endothelial (non-immune). → POSITIVE_WEAK
    ("Decorin", "IL-6"): {
        "a_to_b": unknown(),
        "b_to_a": positive_weak(
            "IL-6 directly induces decorin mRNA in human endothelial cells in a dose-dependent manner",
            n_primary=1, n_secondary=0, immune=False,
        ),
    },
    # [07] Decorin / IL-9 — NOVEL
    ("Decorin", "IL-9"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "No literature found linking IL-9 to decorin expression in immune cells"
        ),
    },
    # [08] Decorin / VEGF — KNOWN_COREGULATED b_to_a
    # Summary: "decorin induces VEGF and modulates VEGFR2; VEGF participates in
    # decorin down-regulation (autocrine feedback) rather than induction"
    # Canonical: Decorin=a, VEGF=b. Decorin → VEGF = a_to_b, but described
    # "Not described in CD8 T cells" → POSITIVE_WEAK (non-immune).
    # VEGF → Decorin = b_to_a, described as "down-regulation" → INHIBITORY.
    ("Decorin", "VEGF"): {
        "a_to_b": positive_weak(
            "decorin induces VEGF and modulates VEGFR2",
            n_primary=1, n_secondary=1, immune=False,
        ),
        "b_to_a": inhibitory(
            "VEGF participates in decorin down-regulation (autocrine feedback) rather than induction",
            n_primary=1, n_secondary=1,
        ),
    },
    # [09] GM-CSF / IL-17A — KNOWN_COREGULATED bidir
    # "GM-CSF promotes Th17 cell survival and IL-17 production"
    # "canonical direction is Th17→GM-CSF; relationship is bidirectional"
    # "promotes" is not a strict R1 induction verb → POSITIVE_WEAK both directions.
    ("GM-CSF", "IL-17A"): {
        "a_to_b": positive_weak(
            "GM-CSF promotes Th17 cell survival and IL-17 production in autoimmunity",
            n_primary=1, n_secondary=1, immune=True,
        ),
        "b_to_a": positive_weak(
            "the canonical direction is Th17→GM-CSF; the relationship is bidirectional in autoimmune feedback loops",
            n_primary=1, n_secondary=1, immune=True,
        ),
    },
    # [10] GM-CSF / IL-27 — KNOWN_COREGULATED b_to_a
    # b_to_a (IL-27 → GM-CSF) is SUPPRESSION → INHIBITORY
    ("GM-CSF", "IL-27"): {
        "a_to_b": unknown(),
        "b_to_a": inhibitory(
            "IL-27 is a robust SUPPRESSOR of GM-CSF in T cells (STAT1-dependent), well documented in EAE and CNS inflammation",
            n_primary=2, n_secondary=0,
        ),
    },
    # [11] GM-CSF / TL1A — KNOWN_DIRECTIONAL a_to_b
    ("GM-CSF", "TL1A"): {
        "a_to_b": positive_strong(
            "GM-CSF induces TL1A (Tnfsf15) expression in macrophages, demonstrated in intestinal inflammation models",
            n_primary=1, n_secondary=1,
        ),
        "b_to_a": unknown(),
    },
    # [12] GM-CSF / VEGF — NOVEL
    ("GM-CSF", "VEGF"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "No literature describes VEGF inducing GM-CSF in CD4 naive T cells"
        ),
    },
    # [13] IFN-beta / IFN-gamma — KNOWN_COREGULATED coregulated_other
    # "in CD4+ T cells it can suppress IFN-gamma" → INHIBITORY a_to_b
    ("IFN-beta", "IFN-gamma"): {
        "a_to_b": inhibitory(
            "in CD4+ T cells it can suppress IFN-gamma",
            n_primary=1, n_secondary=1,
        ),
        "b_to_a": unknown(),
    },
    # [14] IFN-beta / IFN-omega — KNOWN_COREGULATED bidir
    # "IFN-beta is not documented as a direct upstream inducer of IFN-omega"
    # Both directions UNKNOWN — they are co-induced by shared upstream IRF3/IRF7
    ("IFN-beta", "IFN-omega"): {
        "a_to_b": unknown(
            "IFN-beta is not documented as a direct upstream inducer of IFN-omega in CD8 T cells"
        ),
        "b_to_a": unknown(),
    },
    # [15] IFN-beta / IL-2 — KNOWN_DIRECTIONAL a_to_b
    # 0 primary citations; specific evidence is for IFN-α not IFN-β. POSITIVE_WEAK.
    ("IFN-beta", "IL-2"): {
        "a_to_b": positive_weak(
            "Type-I IFNs (IFN-α/β) act as 'signal 3' on CD8 T cells; IFN-α subtypes induce significantly higher frequencies of IL-2-expressing CD8+ T cells",
            n_primary=0, n_secondary=2, immune=True,
        ),
        "b_to_a": unknown(),
    },
    # [16] IFN-gamma / IFN-omega — KNOWN_DIRECTIONAL a_to_b (TAG WRONG)
    # Canonical: IFN-gamma=a, IFN-omega=b. Summary: "IFN-omega specifically
    # enhances cytolytic CD8+ T cell IFN-gamma responses" = b_to_a positive.
    ("IFN-gamma", "IFN-omega"): {
        "a_to_b": unknown(),
        "b_to_a": positive_strong(
            "IFN-omega specifically enhances cytolytic CD8+ T cell IFN-gamma responses",
            n_primary=2, n_secondary=1,
        ),
    },
    # [17] IFN-gamma / IL-2 — KNOWN_DIRECTIONAL a_to_b (TAG WRONG)
    # Canonical: IFN-gamma=a, IL-2=b. Summary: "IL-2 ... induce IFN-gamma" = b_to_a positive.
    ("IFN-gamma", "IL-2"): {
        "a_to_b": unknown(),
        "b_to_a": positive_strong(
            "IL-2 is well-established to induce IFN-gamma production in CD8+ T cells via STAT5/T-bet signaling",
            n_primary=1, n_secondary=1,
        ),
    },
    # [18] IFN-lambda1 / IL-27 — NOVEL
    ("IFN-lambda1", "IL-27"): {
        "a_to_b": unknown(
            "No direct evidence found that IFN-lambda1 (IL-29) induces IL-27 in MAIT or other immune cells"
        ),
        "b_to_a": unknown(),
    },
    # [19] IFN-lambda1 / IL-6 — KNOWN_DIRECTIONAL a_to_b
    ("IFN-lambda1", "IL-6"): {
        "a_to_b": positive_strong(
            "IL-29/IFN-λ1 up-regulates IL-6 production in PBMCs, macrophages, and synovial fibroblasts",
            n_primary=1, n_secondary=1,
        ),
        "b_to_a": unknown(),
    },
    # [20] IFN-lambda1 / IL-9 — NOVEL
    ("IFN-lambda1", "IL-9"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "No published evidence IL-9 induces IFN-λ1 in CD8 T cells"
        ),
    },
    # [21] IFN-lambda1 / VEGF — NOVEL
    ("IFN-lambda1", "VEGF"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "no published evidence found for VEGF induction of IFN-lambda1 (IL-29) in CD8 or any immune cell type"
        ),
    },
    # [22] IFN-omega / IL-15 — KNOWN_DIRECTIONAL a_to_b
    ("IFN-omega", "IL-15"): {
        "a_to_b": positive_strong(
            "Type I interferons (including IFN-omega, signaling through the same IFNAR receptor as IFN-alpha/beta) upregulate IL-15 expression in vitro and in vivo",
            n_primary=3, n_secondary=0,
        ),
        "b_to_a": unknown(),
    },
    # [23] IL-12 / IL-27 — KNOWN_COREGULATED bidir
    ("IL-12", "IL-27"): {
        "a_to_b": unknown(
            "IL-12 does not directly induce IL-27 expression"
        ),
        "b_to_a": unknown(),
    },
    # [24] IL-12 / IL-6 — NOVEL
    ("IL-12", "IL-6"): {
        "a_to_b": unknown(
            "no published evidence that IL-12 induces IL-6 production in NKT cells"
        ),
        "b_to_a": unknown(),
    },
    # [25] IL-12 / IL-9 — PARTIAL partial_lit
    # "IL-12 is listed among cytokines that can increase IL-9 production from
    # CD4+ T cells under certain conditions" — hedged → POSITIVE_WEAK
    ("IL-12", "IL-9"): {
        "a_to_b": positive_weak(
            "IL-12 is listed among cytokines that can increase IL-9 production from CD4+ T cells under certain conditions",
            n_primary=1, n_secondary=0, immune=True,
        ),
        "b_to_a": unknown(),
    },
    # [26] IL-12 / TL1A — KNOWN_COREGULATED coregulated_other
    ("IL-12", "TL1A"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "TL1A does not directly induce IL-12 expression"
        ),
    },
    # [27] IL-12 / VEGF — KNOWN_COREGULATED b_to_a (TAG WRONG — direction is A inhibits B)
    # "documented cascade is IL-12 inhibiting (not inducing) VEGF" → INHIBITORY a_to_b
    ("IL-12", "VEGF"): {
        "a_to_b": inhibitory(
            "the documented cascade is IL-12 inhibiting (not inducing) VEGF",
            n_primary=1, n_secondary=1,
        ),
        "b_to_a": unknown(),
    },
    # [28] IL-13 / IL-27 — KNOWN_COREGULATED b_to_a
    # "IL-27 is documented to suppress Th2 responses (including IL-13 production)"
    # → INHIBITORY for b_to_a (IL-27 → IL-13)
    ("IL-13", "IL-27"): {
        "a_to_b": unknown(),
        "b_to_a": inhibitory(
            "IL-27 is documented to suppress Th2 responses (including IL-13 production)",
            n_primary=1, n_secondary=1,
        ),
    },
    # [29] IL-13 / IL-6 — PARTIAL partial_lit
    # "No direct mechanistic study of IL-6 inducing IL-13 transcription" → UNKNOWN
    ("IL-13", "IL-6"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "No direct mechanistic study of IL-6 inducing IL-13 transcription in naive CD4 T cells"
        ),
    },
    # [30] IL-13 / TL1A — KNOWN_COREGULATED coregulated_other
    # b_to_a: TL1A → IL-13 documented strongly in ILC2/Th2 → POSITIVE_STRONG
    ("IL-13", "TL1A"): {
        "a_to_b": unknown(),
        "b_to_a": positive_strong(
            "TL1A/TNFSF15 drives IL-13 production primarily from ILC2s and Th2/Th9 cells",
            n_primary=1, n_secondary=1,
        ),
    },
    # [31] IL-13 / VEGF — KNOWN_DIRECTIONAL a_to_b (TAG WRONG)
    # Canonical: IL-13=a, VEGF=b. Summary: "VEGF induces IL-13-dependent asthma-like
    # phenotype in transgenic mice" → b_to_a (VEGF → IL-13) POSITIVE_STRONG.
    ("IL-13", "VEGF"): {
        "a_to_b": unknown(),
        "b_to_a": positive_strong(
            "VEGF induces IL-13-dependent asthma-like phenotype in transgenic mice",
            n_primary=3, n_secondary=0,
        ),
    },
    # [32] IL-15 / IL-2 — PRE_REGISTERED b_to_a
    ("IL-15", "IL-2"): {
        "a_to_b": unknown(),
        "b_to_a": pre_registered(),
    },
    # [33] IL-15 / IL-9 — NOVEL
    ("IL-15", "IL-9"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "no evidence IL-9 induces IL-15 production"
        ),
    },
    # [34] IL-15 / VEGF — KNOWN_COREGULATED b_to_a (TAG WRONG)
    # Canonical: IL-15=a, VEGF=b. Summary: "IL-15 signaling via JAK-STAT5 drives
    # VEGF-A expression in NK cells" → a_to_b POSITIVE_STRONG.
    ("IL-15", "VEGF"): {
        "a_to_b": positive_strong(
            "IL-15 signaling via JAK-STAT5 drives VEGF-A expression in NK cells",
            n_primary=1, n_secondary=1,
        ),
        "b_to_a": unknown(
            "No published evidence VEGF induces IL-15"
        ),
    },
    # [35] IL-16 / IL-6 — KNOWN_DIRECTIONAL a_to_b
    ("IL-16", "IL-6"): {
        "a_to_b": positive_strong(
            "IL-16 stimulation of CD14+ CD4+ monocytes/maturing macrophages induces IL-1b, IL-6, IL-15, and TNF-alpha production",
            n_primary=1, n_secondary=0,
        ),
        "b_to_a": unknown(),
    },
    # [36] IL-16 / IL-9 — KNOWN_DIRECTIONAL a_to_b (TAG WRONG; evidence is b_to_a, weak/non-immune)
    # Canonical: IL-16=a, IL-9=b. Summary: "IL-9 acts on bronchial epithelial cells (BEAS-2B)
    # ... to induce ... IL-16" → b_to_a POSITIVE_WEAK (non-immune cell type).
    ("IL-16", "IL-9"): {
        "a_to_b": unknown(),
        "b_to_a": positive_weak(
            "IL-9 acts on bronchial epithelial cells (BEAS-2B) in vitro to induce the T-cell chemoattractants IL-16 and RANTES",
            n_primary=1, n_secondary=1, immune=False,
        ),
    },
    # [37] IL-16 / VEGF — NOVEL
    ("IL-16", "VEGF"): {
        "a_to_b": unknown(
            "No clear evidence of IL-16 inducing VEGF expression in immune cells"
        ),
        "b_to_a": unknown(),
    },
    # [38] IL-17A / IL-36-alpha — KNOWN_DIRECTIONAL a_to_b (TAG WRONG)
    # Canonical: IL-17A=a, IL-36-alpha=b. Summary: "IL-36 (including IL-36alpha)
    # activates Th17 cells and induces IL-17A expression" → b_to_a POSITIVE_STRONG.
    ("IL-17A", "IL-36-alpha"): {
        "a_to_b": unknown(),
        "b_to_a": positive_strong(
            "IL-36 (including IL-36alpha) activates Th17 cells and induces IL-17A expression",
            n_primary=2, n_secondary=1,
        ),
    },
    # [39] IL-27 / IL-35 — KNOWN_COREGULATED coregulated_other
    ("IL-27", "IL-35"): {
        "a_to_b": unknown(),
        "b_to_a": unknown(
            "A→B direction (IL-35 inducing IL-27) is not explicitly documented"
        ),
    },
    # [40] IL-27 / IL-36-alpha — NOVEL
    ("IL-27", "IL-36-alpha"): {
        "a_to_b": unknown(
            "no published evidence shows IL-27 induces IL-36α expression"
        ),
        "b_to_a": unknown(),
    },
    # [41] IL-27 / IL-6 — KNOWN_DIRECTIONAL a_to_b
    # "IL-27 has been shown to augment TNF-alpha-induced IL-6 production in epithelial cells"
    # (non-immune, indirect via TNF). bidir mentioned but no specific b_to_a evidence.
    # → both POSITIVE_WEAK → LOW_CONFIDENCE
    ("IL-27", "IL-6"): {
        "a_to_b": positive_weak(
            "IL-27 has been shown to augment TNF-alpha-induced IL-6 production in epithelial cells",
            n_primary=1, n_secondary=1, immune=False,
        ),
        "b_to_a": positive_weak(
            "the IL-6/IL-27 axis is well-documented bidirectional within the gp130 cytokine family, with both directions reported",
            n_primary=1, n_secondary=1, immune=True,
        ),
    },
    # [42] IL-27 / IL-9 — KNOWN_COREGULATED b_to_a
    # "IL-27 is a well-documented SUPPRESSOR of Th9 differentiation and IL-9 production"
    # → INHIBITORY a_to_b
    ("IL-27", "IL-9"): {
        "a_to_b": inhibitory(
            "IL-27 is a well-documented SUPPRESSOR of Th9 differentiation and IL-9 production in naive CD4+ T cells",
            n_primary=0, n_secondary=2,
        ),
        "b_to_a": unknown(),
    },
    # [43] IL-27 / VEGF — KNOWN_COREGULATED b_to_a (TAG WRONG — direction is A inhibits B)
    # "IL-27 INHIBITS VEGF expression via STAT1 signaling" → INHIBITORY a_to_b
    ("IL-27", "VEGF"): {
        "a_to_b": inhibitory(
            "IL-27 INHIBITS VEGF expression via STAT1 signaling and has documented anti-angiogenic properties",
            n_primary=0, n_secondary=1,
        ),
        "b_to_a": unknown(),
    },
    # [44] IL-35 / VEGF — KNOWN_DIRECTIONAL a_to_b
    # "Tumor-derived IL-35 promotes VEGF secretion and angiogenesis via STAT3 activation in myeloid cells"
    # "promotes" not strict R1 verb + context-dependent → POSITIVE_WEAK
    ("IL-35", "VEGF"): {
        "a_to_b": positive_weak(
            "Tumor-derived IL-35 promotes VEGF secretion and angiogenesis via STAT3 activation in myeloid cells",
            n_primary=3, n_secondary=0, immune=True,
        ),
        "b_to_a": unknown(),
    },
    # [45] IL-36-alpha / IL-6 — KNOWN_COREGULATED b_to_a
    # Canonical: IL-36-alpha=a, IL-6=b. Summary's "reverse" = IL-36α → IL-6.
    # "IL-36-alpha induces IL-6 (in fibroblasts and synovial cells)" — non-immune, 0 primary.
    # → POSITIVE_WEAK for a_to_b.
    ("IL-36-alpha", "IL-6"): {
        "a_to_b": positive_weak(
            "IL-36-alpha induces IL-6 (in fibroblasts and synovial cells, via MyD88/NF-kB)",
            n_primary=0, n_secondary=2, immune=False,
        ),
        "b_to_a": unknown(
            "IL-6 inducing IL-36-alpha is not the documented cascade"
        ),
    },
    # [46] IL-36-alpha / IL-9 — KNOWN_COREGULATED b_to_a (TAG WRONG)
    # Canonical: IL-36-alpha=a, IL-9=b. Summary: "IL-36 signaling ... drives Th9
    # differentiation and IL-9 production in CD4+ T cells" → a_to_b POSITIVE_STRONG.
    ("IL-36-alpha", "IL-9"): {
        "a_to_b": positive_strong(
            "IL-36 signaling (via IL-36R/MyD88/NF-kB) drives Th9 differentiation and IL-9 production in CD4+ T cells",
            n_primary=2, n_secondary=0,
        ),
        "b_to_a": unknown(),
    },
    # [47] IL-36-alpha / TL1A — NOVEL
    ("IL-36-alpha", "TL1A"): {
        "a_to_b": unknown(
            "no published evidence IL-36α induces TL1A in NK CD56bright cells"
        ),
        "b_to_a": unknown(),
    },
    # [48] IL-36-alpha / VEGF — KNOWN_DIRECTIONAL a_to_b
    ("IL-36-alpha", "VEGF"): {
        "a_to_b": positive_strong(
            "IL-36 family upregulates VEGFs and VEGFR2/3 in monocyte-rich inflammatory milieus (psoriasis)",
            n_primary=2, n_secondary=1,
        ),
        "b_to_a": unknown(),
    },
    # [49] IL-6 / TNF-alpha — PRE_REGISTERED b_to_a
    ("IL-6", "TNF-alpha"): {
        "a_to_b": unknown(),
        "b_to_a": pre_registered(),
    },
    # [50] IL-6 / VEGF — KNOWN_DIRECTIONAL a_to_b
    ("IL-6", "VEGF"): {
        "a_to_b": positive_strong(
            "IL-6 is a well-established inducer of VEGF expression via trans-signaling and STAT3 activation",
            n_primary=1, n_secondary=2,
        ),
        "b_to_a": unknown(),
    },
    # [51] IL-9 / TNF-alpha — KNOWN_DIRECTIONAL a_to_b (TAG WRONG)
    # Canonical: IL-9=a, TNF-alpha=b. Summary: "TNF-alpha potently promotes Th9 cell
    # differentiation and IL-9 production" → b_to_a POSITIVE_WEAK (no primary citations).
    ("IL-9", "TNF-alpha"): {
        "a_to_b": unknown(),
        "b_to_a": positive_weak(
            "TNF-alpha potently promotes Th9 cell differentiation and IL-9 production through TNFR2-dependent pathways",
            n_primary=0, n_secondary=2, immune=True,
        ),
    },
    # [52] IL-9 / VEGF — KNOWN_COREGULATED b_to_a (TAG WRONG)
    # Canonical: IL-9=a, VEGF=b. Summary: "IL-9 induces VEGF secretion from mast cells"
    # → a_to_b POSITIVE_STRONG (mast cells treated as immune per plan note).
    ("IL-9", "VEGF"): {
        "a_to_b": positive_strong(
            "IL-9 induces VEGF secretion from mast cells",
            n_primary=1, n_secondary=0,
        ),
        "b_to_a": unknown(
            "No evidence VEGF induces IL-9 in monocytes"
        ),
    },
}


# ----------------------------------------------------------------------- #
# Emit YAML
# ----------------------------------------------------------------------- #

def _yaml_str(s: str) -> str:
    """Escape a string for YAML double-quoted form."""
    if s is None:
        return '""'
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _yaml_val(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return _yaml_str(str(v))


def _emit_direction(dir_dict: dict, indent: str) -> list[str]:
    lines = []
    for k in ("label", "rule_id", "quote",
              "n_citations_primary", "n_citations_secondary", "lit_T_immune"):
        lines.append(f"{indent}{k}: {_yaml_val(dir_dict[k])}")
    return lines


def main():
    if not DIGEST_PATH.exists():
        print(f"FATAL: digest missing: {DIGEST_PATH}", file=sys.stderr)
        sys.exit(2)

    # Load digest as a dict to cross-check axis coverage
    digest_axes = set()
    with DIGEST_PATH.open() as f:
        for row in csv.DictReader(f):
            digest_axes.add((row["axis_a"], row["axis_b"]))

    decided_axes = set(DECISIONS.keys())

    missing = digest_axes - decided_axes
    extra = decided_axes - digest_axes
    if missing:
        print(f"FATAL: {len(missing)} digest axes without decisions:", file=sys.stderr)
        for ax in sorted(missing):
            print(f"  {ax}", file=sys.stderr)
        sys.exit(2)
    if extra:
        print(f"FATAL: {len(extra)} decisions not in digest:", file=sys.stderr)
        for ax in sorted(extra):
            print(f"  {ax}", file=sys.stderr)
        sys.exit(2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Audit decisions for the 53 evaluable Oesinghaus axes.")
    lines.append("# One block per axis; both a_to_b and b_to_a directions scored.")
    lines.append("# Source evidence: reports/cascade_pairs/audit_digest.csv")
    lines.append("# Decision rules: see /Users/yam/.claude/plans/no-job-is-running-glittery-eagle.md")
    lines.append("")
    lines.append("audits:")
    for ax in sorted(decided_axes):
        a, b = ax
        d = DECISIONS[ax]
        lines.append(f"  - axis: [{_yaml_str(a)}, {_yaml_str(b)}]")
        lines.append("    a_to_b:")
        lines.extend(_emit_direction(d["a_to_b"], "      "))
        lines.append("    b_to_a:")
        lines.extend(_emit_direction(d["b_to_a"], "      "))
        lines.append("    reviewer_action: agree")
    lines.append("")
    OUT_PATH.write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH} with {len(decided_axes)} axis decisions", flush=True)


if __name__ == "__main__":
    main()
