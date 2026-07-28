# Pipeline A → Bridge → Path B — directional verdict

**Primary metric = cross_asym** (antisymmetric cross-engagement, `s(a,S_b) − s(b,S_a)`, PBS-normalised). Its sign encodes direction: positive ⇒ axis_a upstream (a_to_b). `directional_score` (the §24 scalar) is reported as a SECONDARY reference — it is symmetric in (a,b) and does NOT encode direction for self-signatures.

**axes_csv:**            `reports/immune_dictionary/id_axes_all66.csv`
**binary_ig_parquet:**   `results/id_cascade/binary_ig/binary_ig.parquet`
**top_n:**               50
**evaluable axes:**      66
**include_donors:**      None
**exclude_donors:**      None
**n_null_perms:**        100

## Headline

- Classification breakdown (cross_asym): **28 STRONG, 23 WEAK, 15 AMBIGUOUS** (out of 66)
- **cross_asym** ground-truth sign accuracy (non-AMBIGUOUS only): **5 / 5**
- directional_score (secondary) sign accuracy (all graded axes): **2 / 6**
- cross_asym non-AMBIGUOUS axes also passing the GENE-SET null control (p_emp < 0.05): **38**
- axes passing the DIRECTION-permutation null (§27.2; dir_p_emp < 0.05): **0** / 66  (Group-U FDR aggregation is done downstream by run_group_u_fdr.py)

## Per-axis summary (cross_asym primary, directional_score secondary)

| axis_a | axis_b | literature_direction | expected_sign | cross_median | cross_consensus | cross_n_pos | cross_n_neg | classification | cross_p_emp_two_sided | cross_sign_correct | dirscore_median | dirscore_sign_correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IFNb | IL15 | no_lit | NaN | +0.0136 | +0.6154 | 8 | 5 | WEAK | +0.3100 | False | +0.1204 | False |
| IFNb | IFNg | a_to_b | +1.0000 | +0.0770 | +0.8462 | 11 | 2 | STRONG | +0.0000 | True | +0.2061 | True |
| IL4 | IL6 | no_lit | NaN | +0.0623 | +0.8333 | 10 | 2 | STRONG | +0.0000 | False | -0.0341 | False |
| IFNb | IL18 | no_lit | NaN | +0.0232 | +0.5385 | 7 | 6 | WEAK | +0.1000 | False | +0.2281 | False |
| IFNg | IL18 | b_to_a | -1.0000 | -0.0606 | +0.7692 | 3 | 10 | STRONG | +0.0000 | True | +0.0384 | False |
| IL13 | TNFa | no_lit | NaN | -0.0376 | +0.9231 | 1 | 12 | STRONG | +0.0000 | False | -0.0240 | False |
| IL13 | IL1b | no_lit | NaN | -0.0178 | +0.6923 | 4 | 9 | WEAK | +0.0300 | False | -0.0146 | False |
| IL12 | IL4 | no_lit | NaN | +0.0056 | +0.5833 | 7 | 5 | AMBIGUOUS | +0.4000 | False | +0.0221 | False |
| IL10 | IL1b | no_lit | NaN | +0.0195 | +0.6923 | 9 | 4 | WEAK | +0.0000 | False | +0.0106 | False |
| IL13 | IL6 | no_lit | NaN | +0.0224 | +0.9231 | 12 | 1 | STRONG | +0.0000 | False | -0.0113 | False |
| IL2 | IL4 | no_lit | NaN | +0.0483 | +0.8333 | 10 | 2 | STRONG | +0.0000 | False | -0.0026 | False |
| IL10 | IL13 | no_lit | NaN | +0.0055 | +0.5385 | 7 | 6 | AMBIGUOUS | +0.2200 | False | +0.0254 | False |
| IL13 | IL4 | no_lit | NaN | +0.0233 | +0.7500 | 9 | 3 | STRONG | +0.0000 | False | +0.0157 | False |
| IFNb | IL1b | no_lit | NaN | +0.0538 | +0.9231 | 12 | 1 | STRONG | +0.0000 | False | +0.2081 | False |
| IL10 | IL6 | no_lit | NaN | +0.0253 | +0.9231 | 12 | 1 | STRONG | +0.0000 | False | +0.0073 | False |
| IL12 | IL2 | no_lit | NaN | -0.0174 | +0.5385 | 6 | 7 | WEAK | +0.0300 | False | -0.0132 | False |
| IL10 | TNFa | no_lit | NaN | +0.0158 | +0.6154 | 8 | 5 | WEAK | +0.0400 | False | +0.0145 | False |
| IFNb | IL2 | no_lit | NaN | +0.0211 | +0.6923 | 9 | 4 | WEAK | +0.1200 | False | +0.2183 | False |
| IL18 | IL2 | no_lit | NaN | -0.0189 | +0.6154 | 5 | 8 | WEAK | +0.0300 | False | +0.0232 | False |
| IFNg | IL15 | b_to_a | -1.0000 | -0.0630 | +1.0000 | 0 | 13 | STRONG | +0.0000 | True | +0.0587 | False |
| IL12 | IL13 | no_lit | NaN | +0.0031 | +0.6154 | 8 | 5 | AMBIGUOUS | +0.5600 | False | +0.0209 | False |
| IL10 | IL12 | no_lit | NaN | -0.0042 | +0.6923 | 4 | 9 | AMBIGUOUS | +0.6200 | False | +0.0135 | False |
| IL15 | IL1b | no_lit | NaN | +0.0108 | +0.6154 | 8 | 5 | WEAK | +0.2300 | False | +0.0059 | False |
| IFNg | IL12 | bidir | NaN | -0.0247 | +0.6923 | 4 | 9 | WEAK | +0.0000 | False | +0.0945 | False |
| IL15 | IL18 | no_lit | NaN | +0.0333 | +0.7692 | 10 | 3 | STRONG | +0.0000 | False | +0.0316 | False |
| IL6 | TNFa | b_to_a | -1.0000 | -0.0227 | +0.8462 | 2 | 11 | STRONG | +0.0000 | True | -0.0163 | True |
| IL1b | IL6 | a_to_b | +1.0000 | -0.0057 | +0.6154 | 5 | 8 | AMBIGUOUS | +0.3500 | False | -0.0129 | False |
| IL2 | IL6 | no_lit | NaN | +0.0414 | +0.9231 | 12 | 1 | STRONG | +0.0000 | False | -0.0207 | False |
| IL10 | IL4 | no_lit | NaN | +0.0392 | +0.8333 | 10 | 2 | STRONG | +0.0000 | False | +0.0497 | False |
| IL12 | IL15 | no_lit | NaN | -0.0158 | +0.6154 | 5 | 8 | WEAK | +0.0100 | False | +0.0230 | False |
| IFNg | IL2 | b_to_a | -1.0000 | -0.0629 | +0.7692 | 3 | 10 | STRONG | +0.0000 | True | +0.0732 | False |
| IL10 | IL2 | no_lit | NaN | +0.0111 | +0.6923 | 9 | 4 | WEAK | +0.0200 | False | -0.0014 | False |
| IL18 | IL4 | no_lit | NaN | +0.0243 | +0.5833 | 7 | 5 | WEAK | +0.0000 | False | +0.0802 | False |
| IL12 | IL6 | no_lit | NaN | +0.0345 | +0.8462 | 11 | 2 | STRONG | +0.0000 | False | +0.0078 | False |
| IL12 | IL1b | no_lit | NaN | +0.0360 | +0.6154 | 8 | 5 | WEAK | +0.0000 | False | +0.0052 | False |
| IL1b | TNFa | no_lit | NaN | +0.0185 | +0.7692 | 10 | 3 | STRONG | +0.0300 | False | +0.0016 | False |
| IL18 | TNFa | no_lit | NaN | -0.0064 | +0.5385 | 6 | 7 | AMBIGUOUS | +0.3700 | False | +0.0325 | False |
| IFNg | TNFa | no_lit | NaN | -0.0060 | +0.5385 | 6 | 7 | AMBIGUOUS | +0.3400 | False | +0.0716 | False |
| IL15 | IL4 | no_lit | NaN | +0.0721 | +0.9167 | 11 | 1 | STRONG | +0.0000 | False | +0.0614 | False |
| IL18 | IL6 | no_lit | NaN | +0.0552 | +1.0000 | 13 | 0 | STRONG | +0.0000 | False | +0.0055 | False |
| IL2 | TNFa | no_lit | NaN | +0.0145 | +0.5385 | 7 | 6 | WEAK | +0.0800 | False | +0.0006 | False |
| IL12 | TNFa | no_lit | NaN | -0.0086 | +0.5385 | 6 | 7 | AMBIGUOUS | +0.1400 | False | +0.0122 | False |
| IFNb | TNFa | no_lit | NaN | +0.0125 | +0.5385 | 7 | 6 | WEAK | +0.2600 | False | +0.2486 | False |
| IFNg | IL6 | no_lit | NaN | +0.0468 | +1.0000 | 13 | 0 | STRONG | +0.0000 | False | +0.0919 | False |
| IL15 | TNFa | no_lit | NaN | +0.0284 | +0.6923 | 9 | 4 | WEAK | +0.0000 | False | +0.0203 | False |
| IL13 | IL18 | no_lit | NaN | -0.0111 | +0.8462 | 2 | 11 | STRONG | +0.1400 | False | +0.0542 | False |
| IL13 | IL2 | no_lit | NaN | -0.0100 | +0.7692 | 3 | 10 | STRONG | +0.1100 | False | +0.0145 | False |
| IL15 | IL2 | no_lit | NaN | +0.0147 | +0.6923 | 9 | 4 | WEAK | +0.0500 | False | +0.0137 | False |
| IL1b | IL4 | no_lit | NaN | +0.0229 | +0.7500 | 9 | 3 | STRONG | +0.0100 | False | +0.0638 | False |
| IL12 | IL18 | no_lit | NaN | +0.0035 | +0.5385 | 7 | 6 | AMBIGUOUS | +0.5400 | False | +0.0334 | False |
| IL4 | TNFa | no_lit | NaN | -0.0170 | +0.8333 | 2 | 10 | STRONG | +0.0200 | False | +0.0688 | False |
| IL1b | IL2 | no_lit | NaN | -0.0059 | +0.5385 | 6 | 7 | AMBIGUOUS | +0.4500 | False | +0.0156 | False |
| IL10 | IL15 | no_lit | NaN | -0.0134 | +0.8462 | 2 | 11 | STRONG | +0.0900 | False | +0.0376 | False |
| IL10 | IL18 | no_lit | NaN | +0.0009 | +0.6154 | 8 | 5 | AMBIGUOUS | +0.9400 | False | +0.0547 | False |
| IFNb | IL12 | no_lit | NaN | -0.0042 | +0.5385 | 6 | 7 | AMBIGUOUS | +0.8000 | False | +0.2951 | False |
| IFNg | IL13 | no_lit | NaN | +0.0090 | +0.5385 | 7 | 6 | AMBIGUOUS | +0.0700 | False | +0.1095 | False |
| IFNg | IL1b | no_lit | NaN | +0.0167 | +0.6923 | 9 | 4 | WEAK | +0.0300 | False | +0.0988 | False |
| IL15 | IL6 | no_lit | NaN | +0.0351 | +1.0000 | 13 | 0 | STRONG | +0.0000 | False | +0.0203 | False |
| IFNb | IL10 | no_lit | NaN | +0.0208 | +0.8462 | 11 | 2 | STRONG | +0.1300 | False | +0.2897 | False |
| IFNg | IL4 | no_lit | NaN | +0.0155 | +0.6667 | 8 | 4 | WEAK | +0.0200 | False | +0.1672 | False |
| IFNg | IL10 | no_lit | NaN | -0.0112 | +0.6923 | 4 | 9 | WEAK | +0.0500 | False | +0.1276 | False |
| IL18 | IL1b | no_lit | NaN | +0.0170 | +0.6154 | 8 | 5 | WEAK | +0.1100 | False | +0.0770 | False |
| IL13 | IL15 | no_lit | NaN | -0.0170 | +0.6154 | 5 | 8 | WEAK | +0.0400 | False | +0.0847 | False |
| IFNb | IL13 | no_lit | NaN | +0.0477 | +0.9231 | 12 | 1 | STRONG | +0.0100 | False | +0.3359 | False |
| IFNb | IL4 | no_lit | NaN | +0.0032 | +0.5000 | 6 | 6 | AMBIGUOUS | +0.7600 | False | +0.3711 | False |
| IFNb | IL6 | no_lit | NaN | -0.0080 | +0.7692 | 3 | 10 | AMBIGUOUS | +0.3500 | False | +0.3078 | False |

## Definitions

- `cross_median`: median across cell types of `cross_asym = s(axis_a,S_axis_b) − s(axis_b,S_axis_a)` (PBS-normalised). Positive ⇒ axis_a engages axis_b's discovered signature more than the reverse ⇒ axis_a upstream ⇒ a_to_b. This is the direction-bearing call.
- `cross_consensus`: fraction of cell types whose `cross_asym` matches the sign of `cross_median`.
- `dirscore_median`: median of the §24 `directional_score` (= asym_PA − asym_PB). Symmetric in (a,b); kept only as a coupling-strength reference. On the Oesinghaus audited benchmark it scored 8/17 (chance) vs cross_asym 15/17 — it cannot infer direction from self-signatures.
- `classification` (on cross_asym):
    - **STRONG**: `|cross_median| ≥ 0.01` AND `consensus ≥ 0.75`
    - **WEAK**: `|cross_median| ≥ 0.01` AND `0.5 ≤ consensus < 0.75`
    - **AMBIGUOUS**: otherwise — not scored against literature
- Null control: cross_asym `p_emp` from 100 random (S_A, S_B) gene-set pairs of the same sizes, drawn from HVGs disjoint from any observed S_X. Tests whether the DISCOVERED S_X carries cytokine-specific information vs random activation-responsive HVGs (the trap §23 Audit 2 caught on Sheu).
- `expected_sign`: +1 if literature says a_to_b, −1 if b_to_a, blank for NOVEL / no_lit axes.
- §24 min_cells = 10.  S_X size = top_50 genes per cytokine from `binary_ig.parquet`.
