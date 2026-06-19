"""Throwaway probe: what does per-donor Sheu loading return at 5hr?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells

MAN = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json"
HVG = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json"
gn = json.load(open(HVG))

for donor in ["M0_rep1", "M0_rep2"]:
    c, _ = load_phase1_cells(MAN, gene_names=gn, time_filter="5hr", donors=[donor])
    cyts = sorted(set(k[0] for k in c))
    cts = sorted(set(k[1] for k in c))
    counts = {k: len(v) for k, v in sorted(c.items())}
    print(f"\n=== donor={donor} ===")
    print("n_groups:", len(c))
    print("cyts:", cyts)
    print("cell_types:", cts)
    print("has_PBS:", any(k[0] == "PBS" for k in c))
    print("counts (first 12):", dict(list(counts.items())[:12]))

# all-donor pooled (what coupling normally uses)
c, _ = load_phase1_cells(MAN, gene_names=gn, time_filter="5hr", donors=None)
print("\n=== pooled (donors=None) ===")
print("cyts:", sorted(set(k[0] for k in c)))
print("cell_types:", sorted(set(k[1] for k in c)))
print("PBS groups:", [(k, len(v)) for k, v in c.items() if k[0] == "PBS"])
