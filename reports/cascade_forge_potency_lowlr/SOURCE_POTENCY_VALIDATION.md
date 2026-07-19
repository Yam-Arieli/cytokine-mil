# source_potency validated against the exact cascade_forge ground truth

Seeds: 3 · labels scored (P_max >= 0.1): 20 of 20

Ground truth (from LARGE_CASCADES, no proxy/pool/audit): out_degree = direct downstream targets; cascade_size = transitive-closure reach; is_source = out_degree>0 (A,B,C,E,F,H,L,M,O,P); isolated = Q,R,S,T (zero cascade edges, true negative controls); leaf = in-cascade but out_degree=0 (D,G,I,J,K,N).

## Validation
- **out_degree** (P1-equivalent): Spearman rho = **-0.081** (n=20)
- **cascade_size** (transitive reach): Spearman rho = **-0.173** (n=20)
- **source (out_degree>0) > leaf-or-isolated (out_degree=0)** (P3-equivalent): mean Delta = -0.243, one-sided p = **0.5971** (source n=10, leaf/isolated n=10)
- **non-isolated (any cascade role) > isolated negatives**: mean Delta = 0.970, one-sided p = **0.1969** (non-isolated n=16, isolated n=4)

## Full ranked table

| rank | label | source_potency | P_max | norm_auc | plateau_ep | late_gain | out_deg | cascade_size | role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | N | +4.06 | 0.994 | 0.953 | 397 | 0.004 | 0 | 0 | leaf |
| 2 | L | +2.70 | 0.995 | 0.956 | 363 | 0.004 | 1 | 1 | source |
| 3 | P | +2.65 | 0.995 | 0.952 | 388 | 0.003 | 1 | 1 | source |
| 4 | D | +2.29 | 0.995 | 0.955 | 356 | 0.003 | 0 | 0 | leaf |
| 5 | I | +1.83 | 0.995 | 0.957 | 329 | 0.003 | 0 | 0 | leaf |
| 6 | C | +1.25 | 0.996 | 0.958 | 328 | 0.003 | 1 | 1 | source |
| 7 | J | +0.72 | 0.995 | 0.959 | 309 | 0.003 | 0 | 0 | leaf |
| 8 | O | +0.11 | 0.995 | 0.961 | 295 | 0.003 | 1 | 1 | source |
| 9 | F | +0.00 | 0.996 | 0.961 | 311 | 0.003 | 1 | 1 | source |
| 10 | Q | -0.11 | 0.995 | 0.961 | 299 | 0.003 | 0 | 0 | isolated |
| 11 | T | -0.60 | 0.995 | 0.963 | 289 | 0.003 | 0 | 0 | isolated |
| 12 | A | -0.79 | 0.996 | 0.963 | 309 | 0.003 | 1 | 3 | source |
| 13 | B | -0.99 | 0.997 | 0.961 | 329 | 0.002 | 1 | 2 | source |
| 14 | M | -1.06 | 0.996 | 0.964 | 282 | 0.002 | 1 | 1 | source |
| 15 | R | -1.06 | 0.996 | 0.964 | 249 | 0.003 | 0 | 0 | isolated |
| 16 | G | -1.15 | 0.996 | 0.964 | 275 | 0.002 | 0 | 0 | leaf |
| 17 | S | -1.34 | 0.996 | 0.966 | 238 | 0.003 | 0 | 0 | isolated |
| 18 | E | -1.43 | 0.996 | 0.965 | 286 | 0.002 | 1 | 2 | source |
| 19 | K | -3.43 | 0.997 | 0.969 | 236 | 0.002 | 0 | 0 | leaf |
| 20 | H | -3.66 | 0.997 | 0.970 | 223 | 0.002 | 3 | 3 | source |

(interpretation added by hand after inspecting the numbers above)
