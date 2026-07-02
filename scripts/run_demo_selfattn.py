"""
Local demo de-risk for the §34 self-attention (CytokineSelfAttnMIL) pipeline.

Runs the WHOLE pipeline end-to-end on the synthetic demo fixture (no cluster):
  build demo data -> Stage-1 encoder -> Stage-2 train_mil with the self-attention
  model + per-epoch checkpoints -> extract_selfattn_trajectory.py (both pkls) ->
  analyze_attention_dynamics.py (pooling head, §33) + analyze_selfattn_interaction.py.

Validates harness correctness (shapes, file layout, cells actually interact) ONLY —
the demo data is random, so the numbers are NOT biologically meaningful.

Usage:
    python scripts/run_demo_selfattn.py [out_dir]
"""

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import make_demo_data as mdd  # noqa: E402
from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset  # noqa: E402
from cytokine_mil.data.label_encoder import CytokineLabel  # noqa: E402
from cytokine_mil.experiment_setup import (  # noqa: E402
    build_encoder, build_selfattn_model, split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder  # noqa: E402
from cytokine_mil.training.train_mil import train_mil  # noqa: E402

STAGE2_EPOCHS = 6
CKPT_EPOCHS = list(range(1, STAGE2_EPOCHS + 1))


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix="selfattn_demo_"))
    data_dir = out_dir / "data"
    run_dir = out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"[demo] out_dir={out_dir}")

    manifest_path = mdd.make_demo_data(str(data_dir))
    gene_names = [f"gene_{i:04d}" for i in range(mdd.N_GENES)]
    gene_list_path = run_dir / "gene_list.json"
    gene_list_path.write_text(json.dumps(gene_names))
    with open(manifest_path) as f:
        manifest = json.load(f)
    label_enc = CytokineLabel().fit(manifest)

    # Stage-1 encoder
    cellds = CellDataset(manifest_path, gene_names=gene_names, preload=True)
    loader = DataLoader(cellds, batch_size=128, shuffle=True, num_workers=0)
    encoder = build_encoder(mdd.N_GENES, n_cell_types=cellds.n_cell_types(), embed_dim=128)
    train_encoder(encoder, loader, n_epochs=3, device=device, verbose=False)
    print("[demo] Stage-1 encoder pretrained.")

    # Run-dir bookkeeping (mirrors train_oesinghaus_full.py)
    train_manifest, val_manifest = split_manifest_by_donor(manifest, val_donors=["Donor3"])
    (run_dir / "manifest_train.json").write_text(json.dumps(train_manifest))
    (run_dir / "manifest_val.json").write_text(json.dumps(val_manifest))
    (run_dir / "label_encoder.json").write_text(
        json.dumps({"cytokines": list(label_enc.cytokines)}))

    # Stage-2 MIL with the self-attention model + checkpoints
    mil = build_selfattn_model(encoder, embed_dim=128, attention_hidden_dim=64,
                               n_classes=label_enc.n_classes(), encoder_frozen=True,
                               sab_heads=4, sab_layers=1)
    assert type(mil).__name__ == "CytokineSelfAttnMIL"
    train_ds = PseudoTubeDataset(str(run_dir / "manifest_train.json"), label_enc,
                                 gene_names=gene_names, preload=True)
    import anndata
    cell_type_obs = {
        i: list(anndata.read_h5ad(e["path"]).obs["cell_type"].astype(str))
        for i, e in enumerate(train_ds.get_entries())
    }
    dynamics = train_mil(
        mil, train_ds, n_epochs=STAGE2_EPOCHS, lr=0.01, momentum=0.9,
        log_every_n_epochs=1, device=device, seed=42, verbose=False,
        checkpoint_dir=str(run_dir / "checkpoints"), checkpoint_epochs=CKPT_EPOCHS,
        cell_type_obs=cell_type_obs,
    )
    with open(run_dir / "dynamics.pkl", "wb") as fh:
        pickle.dump({"records": dynamics["records"],
                     "logged_epochs": dynamics["logged_epochs"],
                     "label_encoder_cytokines": list(label_enc.cytokines)}, fh)
    torch.save(mil.state_dict(), run_dir / "model_stage2.pt")
    print(f"[demo] Stage-2 trained; checkpoints: "
          f"{sorted(p.name for p in (run_dir / 'checkpoints').glob('epoch_*.pt'))}")

    # Extract (both trajectories)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "extract_selfattn_trajectory.py"),
                    "--run_dir", str(run_dir), "--hvg_path", str(gene_list_path),
                    "--device", "cpu"], check=True)

    with open(run_dir / "attention_trajectory.pkl", "rb") as fh:
        at = pickle.load(fh)
    for k in ("epochs", "trajectory", "trajectory_per_donor", "concentration",
              "cell_types", "cytokines"):
        assert k in at, f"missing {k} in attention_trajectory.pkl"
    assert len(at["epochs"]) == len(CKPT_EPOCHS)
    with open(run_dir / "interaction_trajectory.pkl", "rb") as fh:
        it = pickle.load(fh)
    for k in ("epochs", "interaction", "interaction_per_donor", "offdiag",
              "cell_types", "cytokines", "pair_sep"):
        assert k in it, f"missing {k} in interaction_trajectory.pkl"
    # cells must actually interact: some off-diagonal (cross-type) attention present
    offd_final = [float(v[-1]) for v in it["offdiag"].values() if len(v)]
    assert offd_final and max(offd_final) > 0.0, "no cross-cell-type attention (SAB inert?)"
    print(f"[demo] pkls OK: {len(at['cytokines'])} cytokines, {len(at['cell_types'])} cell types; "
          f"max final off-diagonal attention fraction = {max(offd_final):.3f}")

    # Analyze — pooling head (§33) + interaction (§34)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "analyze_attention_dynamics.py"),
                    "--run_dir", str(run_dir),
                    "--report", str(run_dir / "RESULTS_pooling_demo.md")], check=True)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "analyze_selfattn_interaction.py"),
                    "--run_dir", str(run_dir),
                    "--report", str(run_dir / "RESULTS_interaction_demo.md")], check=True)

    for fn in ["offdiag_mass.png", "relay_direction.png"]:
        p = run_dir / "plots" / fn
        print(f"[demo] figure {'OK' if p.exists() else 'MISSING'}: {p}")
    assert (run_dir / "RESULTS_pooling_demo.md").exists()
    assert (run_dir / "RESULTS_interaction_demo.md").exists()
    print(f"\n[demo] PASS — self-attention harness ran end-to-end. Reports in {run_dir}")


if __name__ == "__main__":
    main()
