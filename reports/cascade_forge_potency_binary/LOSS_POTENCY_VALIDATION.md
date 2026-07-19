# loss_potency (loss-based shape metric) vs the exact cascade_forge ground truth

Seeds: 3 · labels scored (L_min <= 1.0): 20 of 20

loss = -log(p_correct) per tube per epoch (exact cross-entropy on the true class -- no retraining, recomputed from the existing p_correct_trajectory). normalized_loss_persistence: HIGH = loss stays elevated longer relative to its own start/floor = drops LATE = deep prediction. late_phase_drop: >0 = loss kept falling in the final third = still learning late. loss_potency = z(normalized_loss_persistence) + z(late_phase_drop).

## Validation
- **out_degree**: Spearman rho = **-0.145** (n=20)
- **cascade_size**: Spearman rho = **-0.209** (n=20)
- **source (out_degree>0) > leaf-or-isolated**: mean Delta = -0.595, one-sided p = **0.7387** (source n=10, leaf/isolated n=10)
- **non-isolated > isolated negatives**: mean Delta = -0.705, one-sided p = **0.7532** (non-isolated n=16, isolated n=4)

## Full ranked table

| rank | label | loss_potency | L_min | L_start | norm_loss_persist | late_drop | out_deg | cascade_size | role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | I | +5.00 | 0.00002 | 0.5930 | 0.003 | 0.00001 | 0 | 0 | leaf |
| 2 | M | +2.63 | 0.00002 | 0.5829 | 0.003 | 0.00001 | 1 | 1 | source |
| 3 | P | +1.80 | 0.00002 | 1.5209 | 0.003 | 0.00001 | 1 | 1 | source |
| 4 | T | +1.57 | 0.00002 | 0.5749 | 0.002 | 0.00001 | 0 | 0 | isolated |
| 5 | S | +1.38 | 0.00002 | 0.6043 | 0.002 | 0.00001 | 0 | 0 | isolated |
| 6 | B | +0.98 | 0.00002 | 0.6396 | 0.003 | 0.00001 | 1 | 2 | source |
| 7 | Q | +0.77 | 0.00002 | 0.6665 | 0.002 | 0.00001 | 0 | 0 | isolated |
| 8 | C | +0.50 | 0.00002 | 0.6306 | 0.002 | 0.00001 | 1 | 1 | source |
| 9 | D | +0.15 | 0.00002 | 0.8526 | 0.002 | 0.00001 | 0 | 0 | leaf |
| 10 | O | -0.41 | 0.00002 | 0.8912 | 0.002 | 0.00001 | 1 | 1 | source |
| 11 | N | -0.42 | 0.00002 | 0.7057 | 0.002 | 0.00001 | 0 | 0 | leaf |
| 12 | G | -0.47 | 0.00002 | 0.6047 | 0.002 | 0.00001 | 0 | 0 | leaf |
| 13 | F | -0.82 | 0.00002 | 0.6028 | 0.002 | 0.00001 | 1 | 1 | source |
| 14 | L | -1.15 | 0.00002 | 0.7781 | 0.002 | 0.00001 | 1 | 1 | source |
| 15 | A | -1.42 | 0.00002 | 0.6326 | 0.002 | 0.00001 | 1 | 3 | source |
| 16 | K | -1.43 | 0.00002 | 0.8663 | 0.002 | 0.00001 | 0 | 0 | leaf |
| 17 | R | -1.46 | 0.00002 | 0.6468 | 0.002 | 0.00001 | 0 | 0 | isolated |
| 18 | E | -1.95 | 0.00002 | 0.7182 | 0.001 | 0.00001 | 1 | 2 | source |
| 19 | J | -2.12 | 0.00002 | 0.8528 | 0.001 | 0.00001 | 0 | 0 | leaf |
| 20 | H | -3.14 | 0.00002 | 1.0648 | 0.001 | 0.00001 | 3 | 3 | source |

(interpretation added by hand after inspecting the numbers above)
