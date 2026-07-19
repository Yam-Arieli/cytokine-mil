# source_potency validated against the exact cascade_forge ground truth

Seeds: 3 · labels scored (P_max >= 0.1): 20 of 20

Ground truth (from LARGE_CASCADES, no proxy/pool/audit): out_degree = direct downstream targets; cascade_size = transitive-closure reach; is_source = out_degree>0 (A,B,C,E,F,H,L,M,O,P); isolated = Q,R,S,T (zero cascade edges, true negative controls); leaf = in-cascade but out_degree=0 (D,G,I,J,K,N).

## Validation
- **out_degree** (P1-equivalent): Spearman rho = **-0.290** (n=20)
- **cascade_size** (transitive reach): Spearman rho = **-0.411** (n=20)
- **source (out_degree>0) > leaf-or-isolated (out_degree=0)** (P3-equivalent): mean Delta = -0.416, one-sided p = **0.6845** (source n=10, leaf/isolated n=10)
- **non-isolated (any cascade role) > isolated negatives**: mean Delta = -0.217, one-sided p = **0.6075** (non-isolated n=16, isolated n=4)

## Full ranked table

| rank | label | source_potency | P_max | norm_auc | plateau_ep | late_gain | out_deg | cascade_size | role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | I | +4.30 | 1.000 | 0.998 | 20 | 0.000 | 0 | 0 | leaf |
| 2 | P | +3.85 | 1.000 | 0.998 | 20 | 0.000 | 1 | 1 | source |
| 3 | M | +1.71 | 1.000 | 0.999 | 14 | 0.000 | 1 | 1 | source |
| 4 | T | +0.96 | 1.000 | 0.999 | 14 | 0.000 | 0 | 0 | isolated |
| 5 | S | +0.94 | 1.000 | 0.999 | 13 | 0.000 | 0 | 0 | isolated |
| 6 | D | +0.64 | 1.000 | 0.999 | 11 | 0.000 | 0 | 0 | leaf |
| 7 | O | +0.58 | 1.000 | 0.999 | 14 | 0.000 | 1 | 1 | source |
| 8 | Q | +0.38 | 1.000 | 0.999 | 10 | 0.000 | 0 | 0 | isolated |
| 9 | C | +0.37 | 1.000 | 0.999 | 12 | 0.000 | 1 | 1 | source |
| 10 | N | -0.38 | 1.000 | 0.999 | 12 | 0.000 | 0 | 0 | leaf |
| 11 | G | -0.66 | 1.000 | 0.999 | 10 | 0.000 | 0 | 0 | leaf |
| 12 | K | -0.67 | 1.000 | 0.999 | 14 | 0.000 | 0 | 0 | leaf |
| 13 | L | -0.74 | 1.000 | 0.999 | 13 | 0.000 | 1 | 1 | source |
| 14 | F | -0.95 | 1.000 | 0.999 | 10 | 0.000 | 1 | 1 | source |
| 15 | B | -1.02 | 1.000 | 0.999 | 11 | 0.000 | 1 | 2 | source |
| 16 | E | -1.57 | 1.000 | 0.999 | 10 | 0.000 | 1 | 2 | source |
| 17 | R | -1.58 | 1.000 | 0.999 | 11 | 0.000 | 0 | 0 | isolated |
| 18 | J | -1.86 | 1.000 | 0.999 | 9 | 0.000 | 0 | 0 | leaf |
| 19 | A | -1.86 | 1.000 | 0.999 | 11 | 0.000 | 1 | 3 | source |
| 20 | H | -2.45 | 1.000 | 0.999 | 7 | 0.000 | 3 | 3 | source |

(interpretation added by hand after inspecting the numbers above)
