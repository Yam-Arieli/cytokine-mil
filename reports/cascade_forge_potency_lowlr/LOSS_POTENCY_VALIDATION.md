# loss_potency (loss-based shape metric) vs the exact cascade_forge ground truth

Seeds: 3 · labels scored (L_min <= 1.0): 20 of 20

loss = -log(p_correct) per tube per epoch (exact cross-entropy on the true class -- no retraining, recomputed from the existing p_correct_trajectory). normalized_loss_persistence: HIGH = loss stays elevated longer relative to its own start/floor = drops LATE = deep prediction. late_phase_drop: >0 = loss kept falling in the final third = still learning late. loss_potency = z(normalized_loss_persistence) + z(late_phase_drop).

## Validation
- **out_degree**: Spearman rho = **-0.072** (n=20)
- **cascade_size**: Spearman rho = **-0.173** (n=20)
- **source (out_degree>0) > leaf-or-isolated**: mean Delta = -0.100, one-sided p = **0.5379** (source n=10, leaf/isolated n=10)
- **non-isolated > isolated negatives**: mean Delta = 0.995, one-sided p = **0.1843** (non-isolated n=16, isolated n=4)

## Full ranked table

| rank | label | loss_potency | L_min | L_start | norm_loss_persist | late_drop | out_deg | cascade_size | role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | L | +3.41 | 0.00547 | 4.2090 | 0.017 | 0.00363 | 1 | 1 | source |
| 2 | N | +3.27 | 0.00558 | 5.0696 | 0.015 | 0.00395 | 0 | 0 | leaf |
| 3 | P | +2.57 | 0.00486 | 4.6698 | 0.017 | 0.00311 | 1 | 1 | source |
| 4 | D | +2.51 | 0.00544 | 4.5850 | 0.016 | 0.00330 | 0 | 0 | leaf |
| 5 | I | +1.58 | 0.00509 | 5.1265 | 0.014 | 0.00323 | 0 | 0 | leaf |
| 6 | C | +1.43 | 0.00432 | 4.7173 | 0.015 | 0.00303 | 1 | 1 | source |
| 7 | J | +0.82 | 0.00473 | 5.0591 | 0.014 | 0.00287 | 0 | 0 | leaf |
| 8 | M | -0.06 | 0.00423 | 4.1240 | 0.014 | 0.00251 | 1 | 1 | source |
| 9 | F | -0.44 | 0.00429 | 5.1318 | 0.012 | 0.00279 | 1 | 1 | source |
| 10 | S | -0.45 | 0.00425 | 4.7556 | 0.013 | 0.00254 | 0 | 0 | isolated |
| 11 | A | -0.50 | 0.00386 | 4.3348 | 0.013 | 0.00255 | 1 | 3 | source |
| 12 | T | -0.52 | 0.00456 | 4.9408 | 0.012 | 0.00268 | 0 | 0 | isolated |
| 13 | R | -0.59 | 0.00438 | 4.6159 | 0.013 | 0.00255 | 0 | 0 | isolated |
| 14 | O | -0.85 | 0.00451 | 5.8272 | 0.011 | 0.00278 | 1 | 1 | source |
| 15 | G | -1.09 | 0.00412 | 4.9522 | 0.012 | 0.00248 | 0 | 0 | leaf |
| 16 | E | -1.42 | 0.00388 | 5.0584 | 0.011 | 0.00240 | 1 | 2 | source |
| 17 | Q | -1.63 | 0.00460 | 6.8156 | 0.010 | 0.00270 | 0 | 0 | isolated |
| 18 | B | -1.69 | 0.00348 | 5.4252 | 0.012 | 0.00225 | 1 | 2 | source |
| 19 | H | -2.96 | 0.00306 | 4.7404 | 0.011 | 0.00184 | 3 | 3 | source |
| 20 | K | -3.41 | 0.00318 | 5.8041 | 0.010 | 0.00179 | 0 | 0 | leaf |

(interpretation added by hand after inspecting the numbers above)
