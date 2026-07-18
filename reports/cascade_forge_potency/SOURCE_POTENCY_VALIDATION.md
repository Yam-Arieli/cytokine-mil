# source_potency validated against the exact cascade_forge ground truth

Seeds: 3 · labels scored (P_max >= 0.1): 20 of 20

Ground truth (from LARGE_CASCADES, no proxy/pool/audit): out_degree = direct downstream targets; cascade_size = transitive-closure reach; is_source = out_degree>0 (A,B,C,E,F,H,L,M,O,P); isolated = Q,R,S,T (zero cascade edges, true negative controls); leaf = in-cascade but out_degree=0 (D,G,I,J,K,N).

## Validation
- **out_degree** (P1-equivalent): Spearman rho = **-0.225** (n=20)
- **cascade_size** (transitive reach): Spearman rho = **-0.274** (n=20)
- **source (out_degree>0) > leaf-or-isolated (out_degree=0)** (P3-equivalent): mean Delta = -0.241, one-sided p = **0.5976** (source n=10, leaf/isolated n=10)
- **non-isolated (any cascade role) > isolated negatives**: mean Delta = 1.364, one-sided p = **0.1125** (non-isolated n=16, isolated n=4)

## Full ranked table

| rank | label | source_potency | P_max | norm_auc | plateau_ep | late_gain | out_deg | cascade_size | role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | I | +3.51 | 0.995 | 0.956 | 33 | 0.003 | 0 | 0 | leaf |
| 2 | L | +3.37 | 0.995 | 0.957 | 33 | 0.003 | 1 | 1 | source |
| 3 | N | +2.76 | 0.996 | 0.956 | 37 | 0.003 | 0 | 0 | leaf |
| 4 | P | +2.66 | 0.996 | 0.952 | 36 | 0.003 | 1 | 1 | source |
| 5 | C | +2.10 | 0.995 | 0.957 | 34 | 0.003 | 1 | 1 | source |
| 6 | D | +1.27 | 0.996 | 0.960 | 32 | 0.003 | 0 | 0 | leaf |
| 7 | J | +1.24 | 0.995 | 0.960 | 31 | 0.003 | 0 | 0 | leaf |
| 8 | G | -0.42 | 0.996 | 0.964 | 26 | 0.002 | 0 | 0 | leaf |
| 9 | R | -0.57 | 0.996 | 0.964 | 26 | 0.002 | 0 | 0 | isolated |
| 10 | Q | -0.60 | 0.996 | 0.964 | 27 | 0.002 | 0 | 0 | isolated |
| 11 | A | -0.89 | 0.997 | 0.965 | 30 | 0.002 | 1 | 3 | source |
| 12 | B | -0.89 | 0.997 | 0.963 | 31 | 0.002 | 1 | 2 | source |
| 13 | O | -0.95 | 0.996 | 0.964 | 27 | 0.002 | 1 | 1 | source |
| 14 | F | -1.18 | 0.997 | 0.965 | 28 | 0.002 | 1 | 1 | source |
| 15 | S | -1.54 | 0.996 | 0.966 | 24 | 0.002 | 0 | 0 | isolated |
| 16 | T | -1.65 | 0.996 | 0.967 | 26 | 0.002 | 0 | 0 | isolated |
| 17 | M | -1.67 | 0.997 | 0.966 | 28 | 0.002 | 1 | 1 | source |
| 18 | H | -1.71 | 0.996 | 0.968 | 23 | 0.002 | 3 | 3 | source |
| 19 | E | -2.04 | 0.997 | 0.967 | 27 | 0.002 | 1 | 2 | source |
| 20 | K | -2.79 | 0.997 | 0.970 | 23 | 0.002 | 0 | 0 | leaf |

(interpretation added by hand after inspecting the numbers above)
