# Overfit diagnosis for the source-potency RED verdict

Seeds: 3 · cytokines: 90 (included above ceiling_floor=0.1: 90)

## Q1 -- Dataset
Oesinghaus 24h PBMC, 91-class multiclass (90 cytokines + PBS), Stage-2 AB-MIL with a FROZEN Stage-1 encoder. Confirmed from `results/attention_dynamics/seed_42/train.log`: 10920 tubes total (9100 train / 1820 val), val donors = Donor2/Donor3. **Stage 2 here is the short 'Stage 2a warmup' from the confusion-dynamics (§33) plan -- 250 epochs at lr=0.001, explicitly a small LR chosen 'so the attention/FCN settle without disturbing the encoder' / 'smooth centroid trajectories', NOT a from-scratch convergence run.** `source_potency` reuses this dynamics.pkl (`records`, i.e. TRAIN donors only) without re-training -- it was never trained/tuned for this specific question.

## Q2 -- Overfitting
**Severe, seed-consistent.** run_summary.json, all 3 seeds:

| seed | train_final | val_final |
|---|---:|---:|
| 42 | 0.4765 | 0.0491 |
| 123 | 0.4545 | 0.0517 |
| 7 | 0.3585 | 0.0565 |

Val final accuracy (5-6%) is barely above 91-class chance (1.1%) and far below train (36-48%). Aggregate (mean-over-cytokines) curve: val peaks at epoch **60** of 250 then declines (val_peak=0.0447, val_final=0.0417); train keeps climbing throughout (train_final=0.4285). See `overfit_aggregate_curve.png`.

## Q3 -- Underfitting (per class, on TRAIN)
1 of 90 cytokines never exceed train P_max=0.2 (never learned even on train donors): IFN-lambda1

## Q2/Q3 tie-in -- does source_potency just measure overfitting?
**Spearman(source_potency, generalization_gap) = -0.184** (n=90). Spearman(source_potency, val_P_max) = -0.393 (n=90).

(interpretation added by hand after inspecting the numbers above)

## Q4 -- family/similar-class confusion vs PBS confusion
Per cytokine, at the FINAL logged epoch of TRAIN records, which OTHER class gets the most softmax mass (`compute_confusion_trajectory`, reused unchanged from `analysis/confusion_dynamics.py`). `top1_same_stem` is a naive string heuristic (strip trailing greek-letter/number/single-letter suffixes, e.g. IL-17A/IL-17F -> 'IL-17') -- NOT a curated immunology family list; a flag to inspect, not a claim.

Of 90 scored cytokines: **0** have PBS as their #1 confusor (majority vote across seeds), **90** have another cytokine. Of those 90, **0** share a name-stem with their #1 confusor.

**Spearman(source_potency, top1_is_pbs_frac) = 0.017** (n=90). **Spearman(source_potency, top1_same_stem_frac) = 0.165** (n=90).

(interpretation added by hand after inspecting the numbers above)

## Full per-cytokine table
| cytokine | source_potency | train_Pmax | val_Pmax | gen_gap | val_final | top1_confusor | top1_mass | same_stem | pool |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| TSLP | -4.96 | 0.892 | 0.135 | 0.757 | 0.085 | IL-1-beta | 0.025 | 0.00 |  |
| IL-21 | -0.27 | 0.785 | 0.043 | 0.742 | 0.004 | CD40L | 0.071 | 0.00 |  |
| IFN-gamma | -3.52 | 0.820 | 0.097 | 0.723 | 0.053 | IL-12 | 0.046 | 0.00 |  |
| M-CSF | -1.00 | 0.721 | 0.057 | 0.664 | 0.011 | Decorin | 0.023 | 0.00 | SHALLOW |
| IL-12 | -0.82 | 0.634 | 0.034 | 0.600 | 0.013 | IFN-epsilon | 0.033 | 0.00 | DEEP |
| IL-7 | -4.44 | 0.792 | 0.216 | 0.576 | 0.165 | IL-2 | 0.017 | 0.00 | SHALLOW |
| GM-CSF | -1.26 | 0.638 | 0.074 | 0.564 | 0.027 | IL-3 | 0.164 | 0.00 |  |
| IL-13 | -3.36 | 0.763 | 0.199 | 0.563 | 0.153 | BAFF | 0.033 | 0.00 |  |
| LIF | -0.76 | 0.544 | 0.012 | 0.532 | 0.000 | IL-36-alpha | 0.041 | 0.00 |  |
| IL-6 | +1.80 | 0.553 | 0.024 | 0.528 | 0.004 | IL-21 | 0.021 | 0.00 | DEEP |
| IL-1Ra | +0.33 | 0.546 | 0.022 | 0.524 | 0.004 | TPO | 0.033 | 0.00 |  |
| CT-1 | -0.33 | 0.525 | 0.010 | 0.515 | 0.000 | IL-19 | 0.044 | 0.00 |  |
| IL-3 | +1.55 | 0.569 | 0.070 | 0.499 | 0.022 | GM-CSF | 0.163 | 0.00 |  |
| Leptin | +0.55 | 0.513 | 0.020 | 0.493 | 0.010 | TNF-alpha | 0.040 | 0.00 |  |
| LIGHT | +1.16 | 0.505 | 0.014 | 0.491 | 0.000 | FGF-beta | 0.034 | 0.00 |  |
| CD40L | -0.55 | 0.821 | 0.343 | 0.478 | 0.259 | IL-21 | 0.063 | 0.00 |  |
| IL-24 | -1.11 | 0.484 | 0.011 | 0.474 | 0.000 | M-CSF | 0.028 | 0.00 |  |
| IL-2 | -1.51 | 0.728 | 0.257 | 0.472 | 0.153 | IL-15 | 0.094 | 0.00 | SHALLOW |
| TL1A | +0.26 | 0.520 | 0.054 | 0.466 | 0.034 | CD40L | 0.044 | 0.00 |  |
| IFN-lambda3 | -0.26 | 0.484 | 0.023 | 0.461 | 0.010 | C3a | 0.063 | 0.00 |  |
| CD27L | +0.24 | 0.558 | 0.100 | 0.459 | 0.090 | RANKL | 0.033 | 0.00 |  |
| IL-36Ra | +0.62 | 0.469 | 0.011 | 0.457 | 0.000 | TL1A | 0.044 | 0.00 |  |
| FGF-beta | +0.86 | 0.458 | 0.026 | 0.432 | 0.012 | GM-CSF | 0.059 | 0.00 |  |
| IFN-epsilon | -0.01 | 0.469 | 0.042 | 0.427 | 0.006 | IL-1-alpha | 0.070 | 0.00 |  |
| IL-10 | -0.63 | 0.703 | 0.276 | 0.427 | 0.194 | TRAIL | 0.050 | 0.00 | SHALLOW |
| IL-15 | -2.95 | 0.742 | 0.322 | 0.421 | 0.169 | IL-2 | 0.102 | 0.00 |  |
| IL-17B | +1.71 | 0.428 | 0.008 | 0.420 | 0.002 | VEGF | 0.040 | 0.00 |  |
| IFN-beta | -4.32 | 0.790 | 0.374 | 0.416 | 0.150 | IFN-omega | 0.144 | 0.00 | SHALLOW |
| TPO | -0.24 | 0.420 | 0.012 | 0.408 | 0.001 | IL-1Ra | 0.053 | 0.00 |  |
| IL-8 | -0.69 | 0.414 | 0.012 | 0.402 | 0.000 | IL-31 | 0.084 | 0.00 |  |
| EGF | +2.05 | 0.410 | 0.011 | 0.399 | 0.001 | RANKL | 0.045 | 0.00 |  |
| GITRL | +3.49 | 0.403 | 0.012 | 0.391 | 0.002 | BAFF | 0.044 | 0.00 |  |
| IFN-alpha1 | +0.70 | 0.430 | 0.043 | 0.388 | 0.022 | SCF | 0.049 | 0.00 |  |
| OX40L | +1.26 | 0.395 | 0.014 | 0.381 | 0.005 | HGF | 0.063 | 0.00 |  |
| PRL | -0.16 | 0.391 | 0.011 | 0.379 | 0.002 | GITRL | 0.034 | 0.00 |  |
| 4-1BBL | +0.76 | 0.390 | 0.011 | 0.379 | 0.001 | Decorin | 0.053 | 0.00 |  |
| IL-1-alpha | +0.75 | 0.421 | 0.043 | 0.378 | 0.018 | IFN-epsilon | 0.050 | 0.00 |  |
| IL-4 | -4.67 | 0.865 | 0.490 | 0.375 | 0.420 | CD40L | 0.025 | 0.00 | SHALLOW |
| IL-17D | +0.30 | 0.413 | 0.038 | 0.375 | 0.008 | IL-8 | 0.062 | 0.00 |  |
| IL-9 | +0.93 | 0.384 | 0.009 | 0.374 | 0.000 | IL-5 | 0.057 | 0.00 |  |
| IL-27 | +0.97 | 0.388 | 0.020 | 0.368 | 0.005 | FasL | 0.047 | 0.00 |  |
| IL-19 | -0.14 | 0.379 | 0.012 | 0.367 | 0.000 | CT-1 | 0.049 | 0.00 |  |
| BAFF | +2.23 | 0.383 | 0.018 | 0.365 | 0.013 | C5a | 0.055 | 0.00 |  |
| IL-17C | +2.21 | 0.374 | 0.011 | 0.363 | 0.000 | IGF-1 | 0.082 | 0.33 |  |
| TNF-alpha | +0.27 | 0.362 | 0.009 | 0.353 | 0.000 | IFN-lambda1 | 0.056 | 0.00 | SHALLOW |
| IL-35 | +1.78 | 0.375 | 0.022 | 0.353 | 0.013 | IL-27 | 0.038 | 0.00 |  |
| C5a | +2.00 | 0.552 | 0.202 | 0.351 | 0.172 | IL-1-beta | 0.049 | 0.00 |  |
| IL-11 | +1.38 | 0.364 | 0.015 | 0.349 | 0.007 | IL-17B | 0.051 | 0.00 |  |
| RANKL | +0.86 | 0.359 | 0.012 | 0.348 | 0.000 | EGF | 0.050 | 0.00 |  |
| IL-16 | +1.21 | 0.360 | 0.013 | 0.347 | 0.001 | TWEAK | 0.067 | 0.00 |  |
| IL-36-alpha | +2.00 | 0.356 | 0.011 | 0.345 | 0.001 | IL-9 | 0.069 | 0.00 |  |
| G-CSF | +0.97 | 0.354 | 0.012 | 0.342 | 0.002 | PBS | 0.046 | 0.00 | SHALLOW |
| IL-34 | +0.17 | 0.382 | 0.041 | 0.341 | 0.012 | IL-23 | 0.041 | 0.00 |  |
| IL-33 | +2.25 | 0.346 | 0.012 | 0.333 | 0.003 | FasL | 0.054 | 0.00 |  |
| IL-17F | -1.02 | 0.337 | 0.012 | 0.325 | 0.003 | IL-33 | 0.032 | 0.00 |  |
| Noggin | +0.84 | 0.335 | 0.011 | 0.325 | 0.000 | GM-CSF | 0.037 | 0.00 |  |
| LT-alpha1-beta2 | +1.20 | 0.332 | 0.009 | 0.323 | 0.000 | IL-8 | 0.072 | 0.00 |  |
| IGF-1 | +0.12 | 0.330 | 0.011 | 0.319 | 0.000 | IL-17C | 0.071 | 0.00 |  |
| IFN-omega | -2.41 | 0.765 | 0.454 | 0.311 | 0.364 | IFN-beta | 0.146 | 0.00 |  |
| FasL | -0.64 | 0.333 | 0.027 | 0.307 | 0.013 | TL1A | 0.064 | 0.00 |  |
| IL-23 | +0.36 | 0.322 | 0.016 | 0.306 | 0.003 | IL-18 | 0.059 | 0.00 |  |
| IL-1-beta | -1.74 | 0.754 | 0.458 | 0.297 | 0.230 | C5a | 0.046 | 0.00 | SHALLOW |
| OSM | -1.57 | 0.304 | 0.008 | 0.296 | 0.002 | IGF-1 | 0.050 | 0.00 | DEEP |
| TRAIL | +0.66 | 0.301 | 0.010 | 0.291 | 0.001 | APRIL | 0.053 | 0.00 |  |
| HGF | +1.13 | 0.296 | 0.009 | 0.286 | 0.002 | OX40L | 0.114 | 0.00 | DEEP |
| ADSF | +0.90 | 0.295 | 0.011 | 0.284 | 0.000 | FGF-beta | 0.030 | 0.00 |  |
| CD30L | -0.53 | 0.284 | 0.008 | 0.276 | 0.000 | Decorin | 0.035 | 0.00 |  |
| TWEAK | +1.30 | 0.282 | 0.011 | 0.271 | 0.004 | C3a | 0.046 | 0.00 |  |
| IL-20 | +0.48 | 0.275 | 0.011 | 0.264 | 0.001 | OX40L | 0.030 | 0.00 |  |
| APRIL | +1.19 | 0.271 | 0.010 | 0.261 | 0.000 | EPO | 0.055 | 0.00 |  |
| EPO | +0.31 | 0.269 | 0.009 | 0.260 | 0.000 | GDNF | 0.067 | 0.00 |  |
| IL-31 | +0.95 | 0.268 | 0.009 | 0.259 | 0.001 | IL-8 | 0.074 | 0.00 |  |
| FLT3L | +1.19 | 0.268 | 0.010 | 0.258 | 0.000 | TPO | 0.062 | 0.00 |  |
| C3a | -1.28 | 0.270 | 0.013 | 0.257 | 0.001 | LT-alpha2-beta1 | 0.056 | 0.00 |  |
| IL-18 | +0.40 | 0.267 | 0.014 | 0.253 | 0.005 | IL-23 | 0.045 | 0.00 |  |
| IFN-lambda2 | -0.79 | 0.262 | 0.010 | 0.253 | 0.001 | IL-20 | 0.035 | 0.00 |  |
| VEGF | +0.86 | 0.264 | 0.013 | 0.251 | 0.002 | IL-17B | 0.052 | 0.00 | DEEP |
| PSPN | +0.18 | 0.262 | 0.012 | 0.251 | 0.001 | LT-alpha1-beta2 | 0.071 | 0.00 |  |
| IL-5 | +2.27 | 0.256 | 0.008 | 0.247 | 0.001 | TRAIL | 0.046 | 0.00 |  |
| IL-17E | +0.43 | 0.252 | 0.011 | 0.241 | 0.001 | FLT3L | 0.040 | 0.00 |  |
| GDNF | +0.45 | 0.251 | 0.010 | 0.241 | 0.003 | Decorin | 0.070 | 0.00 |  |
| LT-alpha2-beta1 | +0.51 | 0.233 | 0.011 | 0.221 | 0.001 | C3a | 0.062 | 0.00 |  |
| SCF | -0.82 | 0.229 | 0.012 | 0.217 | 0.001 | TGF-beta1 | 0.061 | 0.00 |  |
| IL-17A | +0.97 | 0.225 | 0.009 | 0.216 | 0.000 | IFN-lambda1 | 0.076 | 0.00 |  |
| IL-22 | -1.65 | 0.223 | 0.008 | 0.215 | 0.002 | SCF | 0.041 | 0.00 | DEEP |
| IL-26 | -0.21 | 0.220 | 0.011 | 0.209 | 0.001 | IL-1Ra | 0.064 | 0.00 |  |
| Decorin | +0.72 | 0.219 | 0.011 | 0.208 | 0.000 | GDNF | 0.075 | 0.00 |  |
| TGF-beta1 | +0.79 | 0.209 | 0.010 | 0.198 | 0.000 | SCF | 0.059 | 0.00 | DEEP |
| IFN-lambda1 | +0.11 | 0.192 | 0.012 | 0.180 | 0.000 | IL-17A | 0.071 | 0.00 |  |
| IL-32-beta | -5.33 | 0.917 | 0.802 | 0.116 | 0.760 | IL-12 | 0.012 | 0.00 | DEEP |
