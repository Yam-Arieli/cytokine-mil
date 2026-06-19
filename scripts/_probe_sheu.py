"""Throwaway probe: what time_point values exist in Sheu stimulated tubes?"""
import json
import sys
from pathlib import Path

import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cytokine_mil.analysis.eda_pair_benchmark import load_phase1_cells

MAN = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/manifest.json"
HVG = "/cs/labs/mornitzan/yam.arieli/datasets/Sheu2024_pseudotubes/hvg_list.json"
gn = json.load(open(HVG))
man = json.load(open(MAN))
entries = man if isinstance(man, list) else man.get("entries", man.get("tubes", []))

# inspect a few stimulated (non-PBS) tubes
stim = [e for e in entries if str(e.get("cytokine")) != "PBS"][:4]
print("manifest fields:", sorted(entries[0].keys()))
for e in stim:
    a = ad.read_h5ad(e["path"])
    tp_col = next((c for c in a.obs.columns if "time" in c.lower()), None)
    tps = sorted(map(str, a.obs[tp_col].unique())) if tp_col else "NO time col"
    print(f"cyt={e.get('cytokine')} donor={e.get('donor')} time_col={tp_col} "
          f"time_values={tps} obs_cols={list(a.obs.columns)[:8]}")

# what tokens actually yield stimulated cells?
for tok in ["5hr", "5h", "5", "5.0", "3hr"]:
    c, _ = load_phase1_cells(MAN, gene_names=gn, time_filter=tok, donors=None)
    print(f"time_filter={tok!r} -> cyts={sorted(set(k[0] for k in c))}")
