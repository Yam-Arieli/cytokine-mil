# M1 — The biology & the challenge

## 0. Plain-language primer (zero biology assumed)
- A cell has a panel of **buttons**. Pressing a button switches on a set of genes — and
  scRNA-seq lets us *read* which genes are on. So each pressed button leaves a readable
  "fingerprint".
- A **stimulus** = pressing one button from outside (we choose which).
- Some stimuli make the cell **produce a second signal** that then presses **another**
  button by itself (this is "autocrine/paracrine"). That chain is a **cascade**: press A,
  and A *also* causes button B to be pressed.
- Two of our stimuli — **TNF** and **IFN-β (IFNb)** — *are* those second signals. We can
  press them directly too.
- **Direction** = cause vs consequence. If pressing A makes the cell produce B's signal,
  then **A is upstream**.
- The two cascades, one line each (these are the only facts you need):
  - **polyIC / LPS → IFN-β** — pressing polyIC or LPS makes the cell produce IFN-β.
  - **LPS / LPSlo / P3CSK / CpG → TNF** — pressing any of these makes the cell produce TNF.
- **The sign rule:** pairs are written alphabetically as `(a, b)`. `cross_asym` is `+` when
  the alphabetically-first stimulus `a` is upstream, `−` when `b` is upstream.

The sections below (1–8) restate the same thing with the real receptor names.

## 1. The objects
- **Cell → gene expression → scRNA-seq.** Single-cell RNA-seq gives, per cell, a vector of
  counts over genes (here normalized + log1p). A cell's *state* is this expression vector.
- **Cytokine** — a secreted signaling protein. A cell senses it via a **receptor**, which
  triggers an intracellular **signaling pathway** (adaptor → kinases → transcription
  factor), which switches on a **transcriptional program** (a set of genes). That program
  is what scRNA-seq sees.
- **Stimulus** — what we add to the dish: a TLR ligand (LPS, polyIC, CpG, Pam3CSK4) or a
  cytokine (TNF, IFN-β). Each leaves a characteristic program.

## 2. What a cascade is
A **cascade A→B**: stimulus A, through its program, *induces* production of mediator B,
which then acts on cells (autocrine/paracrine) and switches on B's program. A is
**upstream**, B **downstream**. **Direction = which is upstream.**

## 3. The challenge — direction from ONE frame
Direction is a *causal/temporal* statement (A's effect precedes B's). The obvious tool is
a **time-course**: watch A's genes rise before B's. We deliberately forbid that — each
time point is analyzed alone (**no cross-time leakage**, the hard constraint). From a
single frozen frame there is no time axis, so naively direction is unidentifiable.

**The idea that rescues it — the asymmetric fingerprint:**
> An upstream stimulus's cells carry **both** programs — their own *and* the downstream
> one they induce (autocrine). The downstream ligand's cells carry **mainly their own**.
> Therefore `s(upstream-tube, S_downstream) > s(downstream-tube, S_upstream)`.

That inequality is directional information visible in a single frame. Quantified, it is
**`cross_asym`** (M7).

*Caveat planted:* this only works if `S_X` actually captures the upstream-*specific*
genes. When the discovered signature collapses onto the shared (downstream) program, the
asymmetry vanishes or flips. That is exactly the **polyIC→IFNb failure** (S_polyIC becomes
ISG-dominated ≈ S_IFNb) — see M5/M8.

## 4. Why it matters
- Time-courses are expensive and often unavailable; most perturbation atlases are single
  snapshots.
- Direction ≈ causality ≈ *which node to drug* (target the upstream).
- The construction is generic — any labeled cascade system, not cytokine-specific.

## 5. The test biology (two canonical macrophage cascades)
Macrophages sense microbial cues via **Toll-like receptors (TLRs)**. Two adaptor arms:

| arm | adaptor | transcription factor | product | triggered by |
|---|---|---|---|---|
| IFN | **TRIF** | IRF3 | IFN-β → (IFNAR) → ISGs | TLR3 (polyIC), TLR4 (LPS) |
| inflammatory | **MyD88** | NF-κB | TNF → (TNFR) → more NF-κB | TLR2 (P3CSK), TLR4 (LPS/LPSlo), TLR9 (CpG) |

Per Sheu stimulus:

| stimulus | sensor | adaptor | direct product | role |
|---|---|---|---|---|
| PIC (polyIC, dsRNA mimic) | TLR3 | TRIF | IRF3 → IFN-β | **upstream** of IFN |
| LPS | TLR4 | TRIF **+** MyD88 | IFN-β **and** TNF | upstream of both |
| LPSlo (low-dose LPS) | TLR4 | TRIF+MyD88 (weaker) | IFN-β, TNF | upstream (weaker) |
| P3CSK (Pam3CSK4) | TLR2 | MyD88 only | NF-κB → TNF | upstream of TNF; **no IFN arm** |
| CpG | TLR9 | MyD88 | NF-κB → TNF | upstream of TNF; IFN is pDC-restricted |
| TNF | TNFR | (NF-κB) | inflammatory program | **downstream node** (added as ligand) |
| IFNb | IFNAR | (JAK-STAT) | ISGs | **downstream node** (added as ligand) |

Cascades: `polyIC→IFNb`, `LPS→IFNb`, `LPSlo→IFNb` (IFN arm); `LPS/LPSlo/P3CSK/CpG→TNF`
(NF-κB arm). TNF and IFNb are the downstream nodes.

## 6. Role of cell types
A cascade is a **multicellular relay**: cell α secretes IFN-β, neighbor β responds via
IFNAR. We **stratify by cell type** (`mac_c0..c3`) so that (a) composition differences
don't confound, (b) the **relay** cell type can surface, (c) we aggregate the direction
call by **median across cell types + sign consensus** — agreement across cell types is the
evidence. Cell-type labels are used for stratification + post-hoc analysis only; **the
model never sees them**.

## 7. The 21-pair benchmark (real, from `reports/sheu_cascade/sheu_axes_labeled.csv`)
7 stimuli → C(7,2)=21 pairs, hand-labeled from receptor biology (not web search):

- **7 directional (`counts_in_benchmark=True`):** 2 IFN_MUST (`IFNb/PIC`, `IFNb/LPS`),
  1 IFN_SHOULD (`IFNb/LPSlo`), 4 NFKB_SHOULD (`CpG/TNF`, `LPS/TNF`, `LPSlo/TNF`,
  `P3CSK/TNF`).
- **4 NEGATIVE (no cascade):** `CpG/IFNb`, `IFNb/P3CSK`, `IFNb/TNF`, `LPS/LPSlo`.
- **10 UNKNOWN (excluded):** cross-receptor or same-adaptor-parallel pairs with no clean
  direction (e.g. `CpG/LPS`, `P3CSK/PIC`, `PIC/TNF`).

`MUST` = §24 precondition holds (the two programs are transcriptionally distinct).
`SHOULD` = NF-κB/TNFR programs overlap → precondition **risk** (weaker; may fail).
`NEGATIVE` = specificity test (should read ≈ no cascade — though see the M8 caveat that
cross_asym gives direction, not existence).

## 8. The sign convention (subtle — get this right)
Pairs are stored **alphabetically** (`axis_a < axis_b`). `cross_asym = s(a, S_b) − s(b, S_a)`.

> The alphabetical order is **arbitrary bookkeeping we invented**, not biology. A cell has
> no notion of "alphabetical"; biology only knows *who is upstream*. We need *some* fixed
> orientation so that a single signed number can encode direction, and alphabetical is just
> deterministic + reproducible. Flip the convention and every sign flips too — the
> *conclusion* (e.g. CpG upstream of TNF) is unchanged. The only hard rule: the code and
> the benchmark `expected_sign` must use the **same** orientation (both use alphabetical),
> or accuracy is meaningless. (Classic sign-convention bug source — verified consistent in M7.)

- `+` → a engages b's program more → **a upstream** → `a_to_b` → `expected_sign = +1`.
- `−` → b upstream → `b_to_a` → `expected_sign = −1`.

"Upstream" alone isn't the answer — map it through the alphabetical position:
- `LPS,TNF`: LPS upstream, LPS is `a` → `a_to_b` → **+1**.
- `IFNb,PIC`: PIC upstream, PIC is `b` → `b_to_a` → **−1**.

This is where sign mistakes hide; always resolve "who is upstream" *and* "is it a or b".
