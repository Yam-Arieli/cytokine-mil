"""
Local demo de-risk for the §33 attention training-dynamics pipeline.

Runs the WHOLE pipeline end-to-end on the synthetic demo fixture (no cluster):
  build demo data -> Stage-1 encoder pretrain -> Stage-2 train_mil with per-epoch
  checkpoints -> extract_attention_trajectory.py -> analyze_attention_dynamics.py.

This validates harness correctness (shapes, file layout, no crashes) ONLY — the
demo data is random, so P1–P4 numbers are NOT biologically meaningful.

Usage:
    python scripts/run_demo_attention_dynamics.py [out_dir]
"""

import json
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
    build_encoder, build_mil_model, split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder  # noqa: E402
from cytokine_mil.training.train_mil import train_mil  # noqa: E402

# Every-epoch checkpoints so the demo exercises the reconstruct-from-params extractor.
STAGE2_EPOCHS = 6
CKPT_EPOCHS = list(range(1, STAGE2_EPOCHS + 1))


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix="attn_dyn_demo_"))
    data_dir = out_dir / "data"
    run_dir = out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"[demo] out_dir={out_dir}")

    # 1. Demo data + gene list
    manifest_path = mdd.make_demo_data(str(data_dir))
    gene_names = [f"gene_{i:04d}" for i in range(mdd.N_GENES)]
    gene_list_path = run_dir / "gene_list.json"
    gene_list_path.write_text(json.dumps(gene_names))
    with open(manifest_path) as f:
        manifest = json.load(f)

    label_enc = CytokineLabel().fit(manifest)

    # 2. Stage-1 encoder pretrain (tiny)
    cellds = CellDataset(manifest_path, gene_names=gene_names, preload=True)
    loader = DataLoader(cellds, batch_size=128, shuffle=True, num_workers=0)
    encoder = build_encoder(mdd.N_GENES, n_cell_types=cellds.n_cell_types(), embed_dim=128)
    train_encoder(encoder, loader, n_epochs=3, device=device, verbose=False)
    print("[demo] Stage-1 encoder pretrained.")

    # 3. Run-dir bookkeeping (mirrors train_oesinghaus_full.py layout)
    train_manifest, val_manifest = split_manifest_by_donor(manifest, val_donors=["Donor3"])
    (run_dir / "manifest_train.json").write_text(json.dumps(train_manifest))
    (run_dir / "manifest_val.json").write_text(json.dumps(val_manifest))
    (run_dir / "label_encoder.json").write_text(
        json.dumps({"cytokines": list(label_enc.cytokines)}))

    # 4. Stage-2 MIL with checkpoints
    mil = build_mil_model(encoder, embed_dim=128, attention_hidden_dim=64,
                          n_classes=label_enc.n_classes(), encoder_frozen=True)
    train_ds = PseudoTubeDataset(str(run_dir / "manifest_train.json"), label_enc,
                                 gene_names=gene_names, preload=True)
    # Build cell_type_obs (tube-idx -> per-cell types, file order) to exercise the
    # §33 cell-exclusion mask end-to-end, and turn on the entropy penalty.
    import anndata
    cell_type_obs = {
        i: list(anndata.read_h5ad(e["path"]).obs["cell_type"].astype(str))
        for i, e in enumerate(train_ds.get_entries())
    }
    EXCLUDE = {"B_cell"}
    dynamics = train_mil(
        mil, train_ds, n_epochs=STAGE2_EPOCHS, lr=0.01, momentum=0.9,
        log_every_n_epochs=1, device=device, seed=42, verbose=False,
        checkpoint_dir=str(run_dir / "checkpoints"), checkpoint_epochs=CKPT_EPOCHS,
        cell_type_obs=cell_type_obs, attn_entropy_lambda=0.1, exclude_cell_types=EXCLUDE,
    )
    import pickle
    with open(run_dir / "dynamics.pkl", "wb") as fh:
        pickle.dump({"records": dynamics["records"],
                     "logged_epochs": dynamics["logged_epochs"],
                     "label_encoder_cytokines": list(label_enc.cytokines)}, fh)
    torch.save(mil.state_dict(), run_dir / "model_stage2.pt")
    print(f"[demo] Stage-2 trained; checkpoints: "
          f"{sorted(p.name for p in (run_dir / 'checkpoints').glob('epoch_*.pt'))}")

    # 5. Extract attention trajectory (subprocess — exercises the real CLI +
    #    cell-type exclusion, matching the training-time hygiene).
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "extract_attention_trajectory.py"),
                    "--run_dir", str(run_dir), "--hvg_path", str(gene_list_path),
                    "--exclude_cell_types", "B_cell", "--device", "cpu"], check=True)

    # 6. Sanity-check the pkl structure
    with open(run_dir / "attention_trajectory.pkl", "rb") as fh:
        at = pickle.load(fh)
    for k in ("epochs", "trajectory", "trajectory_per_donor", "concentration",
              "cell_types", "cytokines"):
        assert k in at, f"missing key {k} in attention_trajectory.pkl"
    assert len(at["epochs"]) == len(CKPT_EPOCHS)
    some_cyt = at["cytokines"][0]
    some_ct = next(iter(at["trajectory"][some_cyt]))
    assert len(at["trajectory"][some_cyt][some_ct]) == len(CKPT_EPOCHS)
    assert at["trajectory_per_donor"][some_cyt][some_ct]  # per-donor present
    assert "B_cell" not in at["cell_types"], "exclusion failed: B_cell still present"
    print(f"[demo] attention_trajectory.pkl OK: {len(at['cytokines'])} cytokines, "
          f"{len(at['cell_types'])} cell types (B_cell excluded), epochs={at['epochs']}")

    # 7. Analyze (subprocess — exercises readouts, figures, report)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "analyze_attention_dynamics.py"),
                    "--run_dir", str(run_dir),
                    "--report", str(run_dir / "RESULTS_demo.md")], check=True)

    for fn in ["recruitment_ladder.png", "relay_lag.png",
               "primary_secondary_map.png", "concentration.png"]:
        p = run_dir / "plots" / fn
        print(f"[demo] figure {'OK' if p.exists() else 'MISSING'}: {p}")
    assert (run_dir / "RESULTS_demo.md").exists()
    print(f"\n[demo] PASS — end-to-end harness ran. Report: {run_dir / 'RESULTS_demo.md'}")


if __name__ == "__main__":
    main()
