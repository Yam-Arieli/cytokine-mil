# Early-rise feature: alone vs combined with source_potency

Seeds: 3 · early_window: first 50 epochs · labels scored: 20 of 20

early_rise_mean = mean(p_correct) over the first `early_window` epochs of the SAME binary dynamics.pkl already used for source_potency -- no new training. combined = z(source_potency) + z(early_rise_mean), z-scored over included labels.

## (0) source_potency alone (reference, already reported)
- out_degree rho = **-0.290** (n=20)
- cascade_size rho = **-0.411** (n=20)

## (a) early_rise_mean ALONE
- out_degree rho = **0.193** (n=20)
- cascade_size rho = **0.320** (n=20)
- source > leaf-or-isolated: Delta=0.0021, p=**0.4043** (n_a=10, n_b=10)
- non-isolated > isolated: Delta=0.0021, p=**0.3922** (n_a=16, n_b=4)

## (b) combined = z(source_potency) + z(early_rise_mean)
- out_degree rho = **-0.108** (n=20)
- cascade_size rho = **-0.105** (n=20)
- source > leaf-or-isolated: Delta=-0.1218, p=**0.7075** (n_a=10, n_b=10)
- non-isolated > isolated: Delta=-0.0079, p=**0.5055** (n_a=16, n_b=4)

## Full per-label table

| label | early_rise_mean | source_potency | combined | out_deg | cascade_size | role |
|---|---:|---:|---:|---:|---:|---|
| D | 0.9268 | +0.64 | +0.75 | 0 | 0 | leaf |
| M | 0.9131 | +1.71 | +0.62 | 1 | 1 | source |
| I | 0.8841 | +4.30 | +0.49 | 0 | 0 | leaf |
| Q | 0.9238 | +0.38 | +0.44 | 0 | 0 | isolated |
| B | 0.9382 | -1.02 | +0.43 | 1 | 2 | source |
| G | 0.9325 | -0.66 | +0.33 | 0 | 0 | leaf |
| C | 0.9212 | +0.37 | +0.29 | 1 | 1 | source |
| H | 0.9495 | -2.45 | +0.24 | 3 | 3 | source |
| T | 0.9131 | +0.96 | +0.18 | 0 | 0 | isolated |
| S | 0.9124 | +0.94 | +0.13 | 0 | 0 | isolated |
| F | 0.9316 | -0.95 | +0.11 | 1 | 1 | source |
| J | 0.9383 | -1.86 | -0.04 | 0 | 0 | leaf |
| E | 0.9348 | -1.57 | -0.07 | 1 | 2 | source |
| N | 0.9212 | -0.38 | -0.14 | 0 | 0 | leaf |
| A | 0.9317 | -1.86 | -0.41 | 1 | 3 | source |
| O | 0.9047 | +0.58 | -0.50 | 1 | 1 | source |
| P | 0.8703 | +3.85 | -0.53 | 1 | 1 | source |
| R | 0.9233 | -1.58 | -0.72 | 0 | 0 | isolated |
| L | 0.9137 | -0.74 | -0.77 | 1 | 1 | source |
| K | 0.9122 | -0.67 | -0.80 | 0 | 0 | leaf |

(interpretation added by hand after inspecting the numbers above)
