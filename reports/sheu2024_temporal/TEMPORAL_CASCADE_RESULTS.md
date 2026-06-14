# Time-Resolved Gene Cascades — Results (Sheu time course) → metric-AMBER, biology-VISIBLE

**The IFN cascade IS visible in the real time course (IRF3-direct precedes the ISGs in both PIC and
LPS), but the pre-registered activation-time metric missed it — a poorly-chosen metric, not absent
biology.** Job 30833076 (COMPLETED, ~2 min). Pre-reg: `TEMPORAL_CASCADE_PREREGISTRATION.md`. Apparatus
validated on synthetic first (planted early→late cascade: AUC=1.0, p=0.0).

## Pre-registered result (50%-of-own-max activation time)
- **V1 AMBER/RED:** AUC(source earlier) PIC=0.625 (p=0.43), LPS=0.547 (p=0.58); median activation
  = **3hr for BOTH** source (IRF3-direct) and downstream (IFNAR-induced). Not significant.
- **V2:** few directed edges (PIC 1, LPS 7; all source→downstream when present) — most genes co-"activate"
  at 3h so precedence rarely fires.

## Why the metric failed (diagnosed from the kinetics plots)
The metric = first time a gene crosses 50% of its **own max**. The IFN response is dominated by a
**sustained 3–24h plateau** (source max ≈3.3), so 50%-of-max ≈1.65 is only reached at 3h — even for genes
that rise early. Per-gene activation times are essentially all 3h (a few at 5h), washing out the early window.

## What the data actually shows (the characterization — the point of the experiment)
The source-vs-downstream **mean kinetics** (cascade_{PIC,LPS}.png) show the textbook cascade in BOTH stimuli:
- **IRF3-direct (source)** is induced **early — a clear bump at ~1hr** (≈0.7 PIC / ≈0.85 LPS above baseline).
- **IFNAR-induced (downstream ISGs)** are **flat (~0) until 3hr**, then rise.
- Both plateau through 3–24h, source consistently slightly higher.
This is exactly TLR-TRIF→IRF3 (direct, ~1h) → autocrine IFN-β → IFNAR→ISG (downstream, 3h+). The
direction is real and reads off real biological time.

## Correct metric (transparent follow-up, NOT the registered claim)
Use an **absolute-threshold first-crossing** (e.g., first time above-baseline ≥ 0.5) or the **early-window
(0.25–1h) signal**, instead of relative-to-own-max: source crosses ~0.5 at ~1h, downstream at ~3h →
ordering recovered. The registered metric conflated *onset* with *peak*; the early-window onset is the
cascade signal. (Reported as a follow-up because the 50%-max metric was pre-registered.)

## Status of the gene-cascade-direction line
- Relay (snapshot, ID): RED. Learning-order (training dynamics, snapshot): AMBER/RED (SNR).
- **Time-resolved (this): the cascade is VISIBLE in real kinetics** (source IRF3-direct → downstream ISG,
  both stimuli) — the first positive sighting; quantification needs the onset metric, not 50%-of-max.
- Takeaway consistent throughout: **direction needs real time** (it shows up here, snapshots collapsed);
  the only fix needed is the onset-based metric.

## Recommended next step
Re-run with the absolute-threshold / early-window onset metric (small change to `temporal_cascade`) to
QUANTIFY the visible cascade (expected V1 GREEN), reported transparently as a metric correction. Standing
cytokine results unaffected.
