# Strict literature audit log — 53 evaluable Oesinghaus axes

Per-direction labels: `POSITIVE_STRONG` (R1), `POSITIVE_WEAK` (R2), `INHIBITORY` (R3), `UNKNOWN` (R4), `POSITIVE_STRONG` (R5 = pre-registered).
Per-pair status derived from the two directional labels; see plan §Per-pair status.

## Headline numbers

- Total axes audited: **53**
- Axes counted in benchmark accuracy: **17**
- Of those, tag flipped vs original `cytokine_axes.csv`: **7**

## Per-pair-status counts

| Status | Count |
|---|---:|
| DIRECTIONAL_a_to_b | 10 |
| DIRECTIONAL_b_to_a | 7 |
| PARTIAL_INHIBITORY | 7 |
| WEAK_a_to_b | 4 |
| WEAK_b_to_a | 3 |
| LOW_CONFIDENCE | 2 |
| UNKNOWN | 20 |

## DIRECTIONAL_a_to_b

| axis | original_tag | tag_changed | a_to_b | b_to_a | a_to_b quote | b_to_a quote | primary cits |
|---|---|:-:|---|---|---|---|---|
| CD30L / IL-17A | a_to_b |  | POSITIVE_STRONG (R1) | UNKNOWN (R4) | CD30 stimulation directly induces IL-17 production in human cells | (direction not mentioned in summary) | 2 |
| GM-CSF / TL1A | a_to_b |  | POSITIVE_STRONG (R1) | UNKNOWN (R4) | GM-CSF induces TL1A (Tnfsf15) expression in macrophages, demonstrated in intesti... | (direction not mentioned in summary) | 1 |
| IFN-lambda1 / IL-6 | a_to_b |  | POSITIVE_STRONG (R1) | UNKNOWN (R4) | IL-29/IFN-λ1 up-regulates IL-6 production in PBMCs, macrophages, and synovial fi... | (direction not mentioned in summary) | 1 |
| IFN-omega / IL-15 | a_to_b |  | POSITIVE_STRONG (R1) | UNKNOWN (R4) | Type I interferons (including IFN-omega, signaling through the same IFNAR recept... | (direction not mentioned in summary) | 3 |
| IL-15 / VEGF | b_to_a | ✓ | POSITIVE_STRONG (R1) | UNKNOWN (R4) | IL-15 signaling via JAK-STAT5 drives VEGF-A expression in NK cells | No published evidence VEGF induces IL-15 | 1 |
| IL-16 / IL-6 | a_to_b |  | POSITIVE_STRONG (R1) | UNKNOWN (R4) | IL-16 stimulation of CD14+ CD4+ monocytes/maturing macrophages induces IL-1b, IL... | (direction not mentioned in summary) | 1 |
| IL-36-alpha / IL-9 | b_to_a | ✓ | POSITIVE_STRONG (R1) | UNKNOWN (R4) | IL-36 signaling (via IL-36R/MyD88/NF-kB) drives Th9 differentiation and IL-9 pro... | (direction not mentioned in summary) | 2 |
| IL-36-alpha / VEGF | a_to_b |  | POSITIVE_STRONG (R1) | UNKNOWN (R4) | IL-36 family upregulates VEGFs and VEGFR2/3 in monocyte-rich inflammatory milieu... | (direction not mentioned in summary) | 2 |
| IL-6 / VEGF | a_to_b |  | POSITIVE_STRONG (R1) | UNKNOWN (R4) | IL-6 is a well-established inducer of VEGF expression via trans-signaling and ST... | (direction not mentioned in summary) | 1 |
| IL-9 / VEGF | b_to_a | ✓ | POSITIVE_STRONG (R1) | UNKNOWN (R4) | IL-9 induces VEGF secretion from mast cells | No evidence VEGF induces IL-9 in monocytes | 1 |

## DIRECTIONAL_b_to_a

| axis | original_tag | tag_changed | a_to_b | b_to_a | a_to_b quote | b_to_a quote | primary cits |
|---|---|:-:|---|---|---|---|---|
| IFN-gamma / IFN-omega | a_to_b | ✓ | UNKNOWN (R4) | POSITIVE_STRONG (R1) | (direction not mentioned in summary) | IFN-omega specifically enhances cytolytic CD8+ T cell IFN-gamma responses | 2 |
| IFN-gamma / IL-2 | a_to_b | ✓ | UNKNOWN (R4) | POSITIVE_STRONG (R1) | (direction not mentioned in summary) | IL-2 is well-established to induce IFN-gamma production in CD8+ T cells via STAT... | 1 |
| IL-13 / TL1A | coregulated_other |  | UNKNOWN (R4) | POSITIVE_STRONG (R1) | (direction not mentioned in summary) | TL1A/TNFSF15 drives IL-13 production primarily from ILC2s and Th2/Th9 cells | 1 |
| IL-13 / VEGF | a_to_b | ✓ | UNKNOWN (R4) | POSITIVE_STRONG (R1) | (direction not mentioned in summary) | VEGF induces IL-13-dependent asthma-like phenotype in transgenic mice | 3 |
| IL-15 / IL-2 | b_to_a |  | UNKNOWN (R4) | POSITIVE_STRONG (R5) | (direction not mentioned in summary) | Pre-registered KNOWN_CASCADE | 0 |
| IL-17A / IL-36-alpha | a_to_b | ✓ | UNKNOWN (R4) | POSITIVE_STRONG (R1) | (direction not mentioned in summary) | IL-36 (including IL-36alpha) activates Th17 cells and induces IL-17A expression | 2 |
| IL-6 / TNF-alpha | b_to_a |  | UNKNOWN (R4) | POSITIVE_STRONG (R5) | (direction not mentioned in summary) | Pre-registered KNOWN_CASCADE | 0 |

## PARTIAL_INHIBITORY

| axis | original_tag | tag_changed | a_to_b | b_to_a | a_to_b quote | b_to_a quote | primary cits |
|---|---|:-:|---|---|---|---|---|
| Decorin / VEGF | b_to_a |  | POSITIVE_WEAK (R2) | INHIBITORY (R3) | decorin induces VEGF and modulates VEGFR2 | VEGF participates in decorin down-regulation (autocrine feedback) rather than in... | 2 |
| GM-CSF / IL-27 | b_to_a |  | UNKNOWN (R4) | INHIBITORY (R3) | (direction not mentioned in summary) | IL-27 is a robust SUPPRESSOR of GM-CSF in T cells (STAT1-dependent), well docume... | 2 |
| IFN-beta / IFN-gamma | coregulated_other |  | INHIBITORY (R3) | UNKNOWN (R4) | in CD4+ T cells it can suppress IFN-gamma | (direction not mentioned in summary) | 1 |
| IL-12 / VEGF | b_to_a |  | INHIBITORY (R3) | UNKNOWN (R4) | the documented cascade is IL-12 inhibiting (not inducing) VEGF | (direction not mentioned in summary) | 1 |
| IL-13 / IL-27 | b_to_a |  | UNKNOWN (R4) | INHIBITORY (R3) | (direction not mentioned in summary) | IL-27 is documented to suppress Th2 responses (including IL-13 production) | 1 |
| IL-27 / IL-9 | b_to_a |  | INHIBITORY (R3) | UNKNOWN (R4) | IL-27 is a well-documented SUPPRESSOR of Th9 differentiation and IL-9 production... | (direction not mentioned in summary) | 0 |
| IL-27 / VEGF | b_to_a |  | INHIBITORY (R3) | UNKNOWN (R4) | IL-27 INHIBITS VEGF expression via STAT1 signaling and has documented anti-angio... | (direction not mentioned in summary) | 0 |

## WEAK_a_to_b

| axis | original_tag | tag_changed | a_to_b | b_to_a | a_to_b quote | b_to_a quote | primary cits |
|---|---|:-:|---|---|---|---|---|
| IFN-beta / IL-2 | a_to_b |  | POSITIVE_WEAK (R2) | UNKNOWN (R4) | Type-I IFNs (IFN-α/β) act as 'signal 3' on CD8 T cells; IFN-α subtypes induce si... | (direction not mentioned in summary) | 0 |
| IL-12 / IL-9 | partial_lit |  | POSITIVE_WEAK (R2) | UNKNOWN (R4) | IL-12 is listed among cytokines that can increase IL-9 production from CD4+ T ce... | (direction not mentioned in summary) | 1 |
| IL-35 / VEGF | a_to_b |  | POSITIVE_WEAK (R2) | UNKNOWN (R4) | Tumor-derived IL-35 promotes VEGF secretion and angiogenesis via STAT3 activatio... | (direction not mentioned in summary) | 3 |
| IL-36-alpha / IL-6 | b_to_a |  | POSITIVE_WEAK (R2) | UNKNOWN (R4) | IL-36-alpha induces IL-6 (in fibroblasts and synovial cells, via MyD88/NF-kB) | IL-6 inducing IL-36-alpha is not the documented cascade | 0 |

## WEAK_b_to_a

| axis | original_tag | tag_changed | a_to_b | b_to_a | a_to_b quote | b_to_a quote | primary cits |
|---|---|:-:|---|---|---|---|---|
| Decorin / IL-6 | a_to_b |  | UNKNOWN (R4) | POSITIVE_WEAK (R2) | (direction not mentioned in summary) | IL-6 directly induces decorin mRNA in human endothelial cells in a dose-dependen... | 1 |
| IL-16 / IL-9 | a_to_b |  | UNKNOWN (R4) | POSITIVE_WEAK (R2) | (direction not mentioned in summary) | IL-9 acts on bronchial epithelial cells (BEAS-2B) in vitro to induce the T-cell ... | 1 |
| IL-9 / TNF-alpha | a_to_b |  | UNKNOWN (R4) | POSITIVE_WEAK (R2) | (direction not mentioned in summary) | TNF-alpha potently promotes Th9 cell differentiation and IL-9 production through... | 0 |

## LOW_CONFIDENCE

| axis | original_tag | tag_changed | a_to_b | b_to_a | a_to_b quote | b_to_a quote | primary cits |
|---|---|:-:|---|---|---|---|---|
| GM-CSF / IL-17A | bidir |  | POSITIVE_WEAK (R2) | POSITIVE_WEAK (R2) | GM-CSF promotes Th17 cell survival and IL-17 production in autoimmunity | the canonical direction is Th17→GM-CSF; the relationship is bidirectional in aut... | 2 |
| IL-27 / IL-6 | a_to_b |  | POSITIVE_WEAK (R2) | POSITIVE_WEAK (R2) | IL-27 has been shown to augment TNF-alpha-induced IL-6 production in epithelial ... | the IL-6/IL-27 axis is well-documented bidirectional within the gp130 cytokine f... | 2 |

## UNKNOWN

| axis | original_tag | tag_changed | a_to_b | b_to_a | a_to_b quote | b_to_a quote | primary cits |
|---|---|:-:|---|---|---|---|---|
| CD30L / IL-27 | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | no direct evidence found that IL-27 induces CD30L/TNFSF8 expression on monocytes | 0 |
| CD30L / IL-9 | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | no published evidence IL-9 induces CD30L expression on HSPCs | 0 |
| CD30L / VEGF | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | no clear literature was found establishing CD30L→VEGF induction in CD14+ monocyt... | (direction not mentioned in summary) | 0 |
| Decorin / IL-16 | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | no published evidence found linking decorin to IL-16 induction | (direction not mentioned in summary) | 0 |
| Decorin / IL-27 | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | no direct evidence was found linking decorin to IL-27 induction | (direction not mentioned in summary) | 0 |
| Decorin / IL-9 | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | No literature found linking IL-9 to decorin expression in immune cells | 0 |
| GM-CSF / VEGF | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | No literature describes VEGF inducing GM-CSF in CD4 naive T cells | 0 |
| IFN-beta / IFN-omega | bidir |  | UNKNOWN (R4) | UNKNOWN (R4) | IFN-beta is not documented as a direct upstream inducer of IFN-omega in CD8 T ce... | (direction not mentioned in summary) | 0 |
| IFN-lambda1 / IL-27 | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | No direct evidence found that IFN-lambda1 (IL-29) induces IL-27 in MAIT or other... | (direction not mentioned in summary) | 0 |
| IFN-lambda1 / IL-9 | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | No published evidence IL-9 induces IFN-λ1 in CD8 T cells | 0 |
| IFN-lambda1 / VEGF | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | no published evidence found for VEGF induction of IFN-lambda1 (IL-29) in CD8 or ... | 0 |
| IL-12 / IL-27 | bidir |  | UNKNOWN (R4) | UNKNOWN (R4) | IL-12 does not directly induce IL-27 expression | (direction not mentioned in summary) | 0 |
| IL-12 / IL-6 | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | no published evidence that IL-12 induces IL-6 production in NKT cells | (direction not mentioned in summary) | 0 |
| IL-12 / TL1A | coregulated_other |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | TL1A does not directly induce IL-12 expression | 0 |
| IL-13 / IL-6 | partial_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | No direct mechanistic study of IL-6 inducing IL-13 transcription in naive CD4 T ... | 0 |
| IL-15 / IL-9 | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | no evidence IL-9 induces IL-15 production | 0 |
| IL-16 / VEGF | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | No clear evidence of IL-16 inducing VEGF expression in immune cells | (direction not mentioned in summary) | 0 |
| IL-27 / IL-35 | coregulated_other |  | UNKNOWN (R4) | UNKNOWN (R4) | (direction not mentioned in summary) | A→B direction (IL-35 inducing IL-27) is not explicitly documented | 0 |
| IL-27 / IL-36-alpha | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | no published evidence shows IL-27 induces IL-36α expression | (direction not mentioned in summary) | 0 |
| IL-36-alpha / TL1A | no_lit |  | UNKNOWN (R4) | UNKNOWN (R4) | no published evidence IL-36α induces TL1A in NK CD56bright cells | (direction not mentioned in summary) | 0 |
