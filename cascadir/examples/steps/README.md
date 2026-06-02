# cascadir — per-step walkthrough

Each script demonstrates **one step** of the method using the `cascadir` API, on
the small synthetic planted-cascade dataset (`../synthetic_data.py`, a known
`CytA -> CytB` cascade). They are self-contained — run them in order or individually.

```bash
cd cascadir/examples/steps
python 01_validate.py            # data-suitability contract (strict)
python 02_preprocess.py          # normalize + log1p + HVG (raw vs lognorm)
python 03_build_pseudotubes.py   # in-memory bags of cells, stratified by cell type
python 04_train_encoder.py       # Stage-1 cell-type encoder
python 05_path_a_coupling.py     # Path A — which pairs are coupled (existence)
python 06_bridge_signatures.py   # binary AB-MIL + Integrated-Gradients signatures
python 07_path_b_direction.py    # Path B — who is upstream (cross_asym)
```

The **analysis step** is a notebook (it is the most visual): see
[`../analysis/analysis.ipynb`](../analysis/analysis.ipynb), which scores and plots
the cross_asym direction results of the Oesinghaus dataset (the best of the three,
88%).

For the whole thing in one script see [`../full_experiment.py`](../full_experiment.py).
