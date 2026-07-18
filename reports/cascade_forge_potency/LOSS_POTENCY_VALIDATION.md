# loss_potency (loss-based shape metric) vs the exact cascade_forge ground truth

Seeds: 3 · labels scored (L_min <= 1.0): 20 of 20

loss = -log(p_correct) per tube per epoch (exact cross-entropy on the true class -- no retraining, recomputed from the existing p_correct_trajectory). normalized_loss_persistence: HIGH = loss stays elevated longer relative to its own start/floor = drops LATE = deep prediction. late_phase_drop: >0 = loss kept falling in the final third = still learning late. loss_potency = z(normalized_loss_persistence) + z(late_phase_drop).

## Validation
- **out_degree**: Spearman rho = **-0.044** (n=20)
- **cascade_size**: Spearman rho = **-0.105** (n=20)
- **source (out_degree>0) > leaf-or-isolated**: mean Delta = -0.189, one-sided p = **0.5825** (source n=10, leaf/isolated n=10)
- **non-isolated > isolated negatives**: mean Delta = 1.324, one-sided p = **0.1021** (non-isolated n=16, isolated n=4)

## Full ranked table

| rank | label | loss_potency | L_min | L_start | norm_loss_persist | late_drop | out_deg | cascade_size | role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | L | +3.34 | 0.00505 | 3.6950 | 0.021 | 0.00333 | 1 | 1 | source |
| 2 | P | +2.73 | 0.00424 | 3.4556 | 0.024 | 0.00259 | 1 | 1 | source |
| 3 | I | +2.44 | 0.00507 | 4.2056 | 0.018 | 0.00330 | 0 | 0 | leaf |
| 4 | N | +2.16 | 0.00427 | 3.7516 | 0.019 | 0.00302 | 0 | 0 | leaf |
| 5 | D | +2.13 | 0.00435 | 3.3452 | 0.021 | 0.00269 | 0 | 0 | leaf |
| 6 | C | +2.06 | 0.00468 | 3.4449 | 0.021 | 0.00275 | 1 | 1 | source |
| 7 | J | +0.71 | 0.00456 | 4.2801 | 0.017 | 0.00267 | 0 | 0 | leaf |
| 8 | R | +0.65 | 0.00375 | 3.0731 | 0.020 | 0.00230 | 0 | 0 | isolated |
| 9 | G | +0.06 | 0.00398 | 3.4932 | 0.018 | 0.00239 | 0 | 0 | leaf |
| 10 | H | -0.71 | 0.00368 | 3.0784 | 0.017 | 0.00221 | 3 | 3 | source |
| 11 | A | -0.74 | 0.00337 | 3.2777 | 0.016 | 0.00222 | 1 | 3 | source |
| 12 | B | -0.92 | 0.00294 | 3.4728 | 0.017 | 0.00206 | 1 | 2 | source |
| 13 | M | -1.03 | 0.00349 | 3.3265 | 0.017 | 0.00203 | 1 | 1 | source |
| 14 | T | -1.52 | 0.00356 | 3.9188 | 0.015 | 0.00209 | 0 | 0 | isolated |
| 15 | S | -1.63 | 0.00384 | 4.1207 | 0.015 | 0.00212 | 0 | 0 | isolated |
| 16 | O | -1.68 | 0.00358 | 4.3590 | 0.014 | 0.00212 | 1 | 1 | source |
| 17 | Q | -1.74 | 0.00406 | 5.0204 | 0.013 | 0.00230 | 0 | 0 | isolated |
| 18 | F | -1.75 | 0.00343 | 4.0220 | 0.014 | 0.00210 | 1 | 1 | source |
| 19 | E | -2.26 | 0.00308 | 4.2290 | 0.014 | 0.00194 | 1 | 2 | source |
| 20 | K | -2.32 | 0.00326 | 3.8814 | 0.014 | 0.00188 | 0 | 0 | leaf |

(interpretation added by hand after inspecting the numbers above)
