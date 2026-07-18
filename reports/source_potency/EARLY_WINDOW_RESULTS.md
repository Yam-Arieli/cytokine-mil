# Source-potency, recomputed on the pre-overfitting window

Truncation epoch (data-driven: where the aggregate val D2/D3 curve peaks, 3-seed mean): **60** of 250. Same dynamics.pkl as the published run (`results/attention_dynamics/seed_*`), same TRAIN records -- only the trajectory length changes; no re-training.

## Headline comparison: does removing the overfit tail rescue P1?

| window | n scored | P1: rho(potency, out-degree) | P2: rho(potency, coupling-degree) | P3: DEEP>SHALLOW p |
|---|---:|---:|---:|---:|
| FULL window (0-250, published) | 90/90 | -0.067 (n=13) | 0.216 (n=45) | 0.1236 (Delta=1.299) |
| EARLY window (0-60, pre-overfitting) | 26/90 | -0.866 (n=5) | -0.290 (n=18) | 0.7056 (Delta=-0.805) |

## Master-regulator ranks, both windows

| master regulator | FULL rank | EARLY rank |
|---|---:|---:|
| TNF-alpha | 49/90 | n/a/26 |
| IL-1-beta | 81/90 | 5/26 |
| IFN-beta | 86/90 | 22/26 |
| IFN-gamma | 85/90 | 9/26 |
| IL-12 | 71/90 | 18/26 |
| GM-CSF | 76/90 | 16/26 |

## Full ranked table (EARLY window)

| rank | cytokine | source_potency | P_max | norm_auc | plateau_ep | late_gain | out_deg | coup_deg | pool |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | CD40L | +3.46 | 0.517 | 0.438 | 57 | 0.263 | 0 | 2 |  |
| 2 | IL-10 | +3.46 | 0.413 | 0.405 | 57 | 0.245 | 0 | 0 | SHALLOW |
| 3 | IL-21 | +3.07 | 0.259 | 0.373 | 57 | 0.204 | 0 | 3 |  |
| 4 | M-CSF | +2.05 | 0.356 | 0.430 | 60 | 0.175 | 0 | 0 | SHALLOW |
| 5 | IL-1-beta | +1.89 | 0.459 | 0.496 | 57 | 0.202 | 0 | 0 | SHALLOW |
| 6 | IFN-epsilon | +1.01 | 0.103 | 0.354 | 60 | 0.072 | 0 | 0 |  |
| 7 | CT-1 | +0.93 | 0.195 | 0.443 | 57 | 0.116 | 0 | 2 |  |
| 8 | C5a | +0.62 | 0.155 | 0.421 | 60 | 0.086 | 0 | 2 |  |
| 9 | IFN-gamma | +0.49 | 0.500 | 0.542 | 53 | 0.144 | 0 | 3 |  |
| 10 | IFN-lambda3 | +0.29 | 0.137 | 0.438 | 60 | 0.075 | 0 | 0 |  |
| 11 | IL-19 | +0.23 | 0.136 | 0.501 | 60 | 0.106 | 0 | 4 |  |
| 12 | TSLP | +0.01 | 0.716 | 0.576 | 53 | 0.134 | 0 | 1 |  |
| 13 | IL-24 | -0.05 | 0.119 | 0.507 | 60 | 0.093 | 0 | 1 |  |
| 14 | IL-13 | -0.20 | 0.397 | 0.541 | 60 | 0.102 | 0 | 5 |  |
| 15 | IL-6 | -0.46 | 0.100 | 0.490 | 60 | 0.059 | 1 | 12 | DEEP |
| 16 | GM-CSF | -0.65 | 0.333 | 0.556 | 60 | 0.084 | 1 | 6 |  |
| 17 | IL-32-beta | -0.78 | 0.761 | 0.659 | 47 | 0.133 | 0 | 0 | DEEP |
| 18 | IL-12 | -1.16 | 0.149 | 0.548 | 53 | 0.050 | 0 | 6 | DEEP |
| 19 | IL-15 | -1.18 | 0.491 | 0.625 | 57 | 0.091 | 1 | 5 |  |
| 20 | IL-3 | -1.18 | 0.206 | 0.566 | 57 | 0.058 | 0 | 0 |  |
| 21 | IL-2 | -1.27 | 0.402 | 0.624 | 60 | 0.085 | 2 | 3 | SHALLOW |
| 22 | IFN-beta | -1.72 | 0.563 | 0.704 | 47 | 0.102 | 0 | 3 | SHALLOW |
| 23 | IL-7 | -1.77 | 0.622 | 0.664 | 50 | 0.077 | 0 | 0 | SHALLOW |
| 24 | IL-8 | -1.95 | 0.109 | 0.588 | 53 | 0.025 | 0 | 2 |  |
| 25 | IFN-omega | -2.54 | 0.471 | 0.719 | 37 | 0.061 | 2 | 3 |  |
| 26 | IL-4 | -2.61 | 0.759 | 0.774 | 40 | 0.087 | 0 | 3 | SHALLOW |

(interpretation added by hand after inspecting the numbers above)
