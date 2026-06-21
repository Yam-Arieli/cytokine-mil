"""
Recurrent-IG Oesinghaus experiment driver (CLAUDE.md §31, Part B).

Reproduces the §26 binary-IG signature training FAITHFULLY (one shared Stage-1
encoder per seed + the WIDE HP config + the committed pseudo-tubes) and adds
recurrent Integrated Gradients: every ``--checkpoint_every`` epochs of the
full-model (Stage-2 binary) training the model is checkpointed and IG is run, so
each static signature S_X becomes a per-epoch TRAJECTORY.

Scope: all 45 cytokines in reports/cascade_pairs/cytokine_axes.csv that are present
in the Oesinghaus manifest, one binary AB-MIL (cytokine vs PBS) each, on the shared
frozen encoder. Run once per seed.

Faithfulness (the §27.6 lesson): a SINGLE shared encoder per seed (NOT per-chunk) and
the wide config (embed=512, hidden=(512,512), attn=128, Stage1 20@0.005, Stage2
250@3e-5). The IG itself is computed with cascadir.integrated_gradients (dogfood).

Checkpointing reuses train_mil's built-in checkpoint_dir/checkpoint_epochs, so all 250
epochs run as ONE optimization (momentum preserved) and we attribute the saved
states post-hoc. Per-cytokine checkpoint files are deleted after IG extraction unless
--keep_checkpoints.

Outputs (per --output_dir):
    ig_traj.parquet         long: cytokine, gene, epoch, ig, rank_ig, seed
    final_signatures.parquet  epoch==STAGE2_EPOCHS, rank_ig < top_n (regression anchor)
    encoder_shared_stage1.pt  the shared encoder
    run_log.txt

Usage:
    python scripts/run_recurrent_ig_oesinghaus.py --seed 42 --output_dir results/recurrent_ig/seed_42
"""

import argparse
import copy
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# Dogfood cascadir for the IG core (pure-python; deps are all in biovenv).
sys.path.insert(0, str(REPO_ROOT / "cascadir" / "src"))

from cytokine_mil.data.dataset import CellDataset, PseudoTubeDataset  # noqa: E402
from cytokine_mil.models.instance_encoder import InstanceEncoder  # noqa: E402
from cytokine_mil.experiment_setup import (  # noqa: E402
    build_mil_model,
    build_stage1_manifest,
    filter_manifest,
    make_binary_manifest,
    split_manifest_by_donor,
)
from cytokine_mil.training.train_encoder import train_encoder  # noqa: E402
from cytokine_mil.training.train_mil import train_mil  # noqa: E402

from cascadir.signatures import integrated_gradients  # noqa: E402


# ---------------------------------------------------------------------------
# Constants (faithful to §26)
# ---------------------------------------------------------------------------

MANIFEST_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/manifest.json"
HVG_PATH = "/cs/labs/mornitzan/yam.arieli/datasets/Oesinghaus_pseudotubes/hvg_list.json"
AXES_CSV = REPO_ROOT / "reports" / "cascade_pairs" / "cytokine_axes.csv"
VAL_DONORS = ["Donor2", "Donor3"]
CONTROL = "PBS"

# WIDE config — reproduces the §26 published signatures.
EMBED_DIM = 512
HIDDEN_DIMS = (512, 512)
ATTENTION_HIDDEN_DIM = 128
STAGE1_EPOCHS = 20
STAGE1_LR = 0.005
STAGE1_MOMENTUM = 0.9
STAGE2_EPOCHS = 250
STAGE2_LR = 0.00003
STAGE2_MOMENTUM = 0.90
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _target_cytokines(manifest_cytokines: set) -> list:
    """45 cytokines from cytokine_axes.csv that are present in the manifest."""
    df = pd.read_csv(AXES_CSV)
    cyts = sorted(set(df["axis_a"]).union(set(df["axis_b"])))
    present = [c for c in cyts if c in manifest_cytokines]
    return present


def _tubes_by_class(dataset: PseudoTubeDataset, max_tubes: int):
    """Split a binary PseudoTubeDataset's tubes into (cytokine_X_list, pbs_X_list)."""
    cyt_X, pbs_X = [], []
    for i in range(len(dataset)):
        X, label, _donor, _name = dataset[i]
        X = X.detach().cpu().numpy().astype(np.float32)
        if int(label) == 0:  # positive = cytokine
            cyt_X.append(X)
        else:  # negative = PBS
            pbs_X.append(X)
    return cyt_X[:max_tubes], pbs_X[:max_tubes]


def _pbs_baseline(pbs_X_list, device) -> torch.Tensor:
    """Per-gene baseline = mean over PBS tubes of each tube's gene-mean (G,)."""
    tube_means = np.stack([X.mean(axis=0) for X in pbs_X_list], axis=0)
    base = tube_means.mean(axis=0).astype(np.float32)
    return torch.from_numpy(base).to(device)


def _ig_ranking(model, cyt_X_list, baseline_t, gene_names, *, n_steps, top_n, device):
    """Accumulate IG over a cytokine's tubes; return top_n (gene, ig, rank) rows."""
    g = baseline_t.shape[0]
    ig_accum = np.zeros(g, dtype=np.float64)
    n_used = 0
    model.eval()
    for X in cyt_X_list:
        Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
        base = baseline_t.unsqueeze(0).expand_as(Xt).contiguous()
        ig = integrated_gradients(model, Xt, target_class=0, baseline=base, n_steps=n_steps)
        ig_accum += ig.mean(dim=0).detach().cpu().numpy()
        n_used += 1
    ig_mean = ig_accum / max(n_used, 1)
    order = np.argsort(-ig_mean)
    k = min(top_n, g)
    return [(gene_names[i], float(ig_mean[i]), rank) for rank, i in enumerate(order[:k])]


# ---------------------------------------------------------------------------
# Per-cytokine training + recurrent IG
# ---------------------------------------------------------------------------


def _run_one_cytokine(
    target, manifest, gene_names, shared_encoder, out_dir, ckpt_root, device, seed,
    *, embed_dim, attention_hidden_dim, checkpoint_epochs, n_steps, traj_top_n,
    max_tubes, keep_checkpoints, log,
):
    safe = target.replace("/", "_")
    bin_manifest, label_enc = make_binary_manifest(manifest, target, control=CONTROL)
    train_m, _val_m = split_manifest_by_donor(bin_manifest, VAL_DONORS)
    if len(train_m) < 4:
        log(f"  SKIP {target}: only {len(train_m)} train tubes")
        return []

    train_m_path = out_dir / f"_manifest_train_{safe}.json"
    with open(train_m_path, "w") as fh:
        json.dump(train_m, fh)
    train_dataset = PseudoTubeDataset(
        str(train_m_path), label_enc, gene_names=gene_names, preload=True
    )

    encoder = copy.deepcopy(shared_encoder)
    model = build_mil_model(
        encoder, embed_dim=embed_dim, attention_hidden_dim=attention_hidden_dim,
        n_classes=2, encoder_frozen=True,
    )

    ckpt_dir = ckpt_root / safe
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_mil(
        model, train_dataset, n_epochs=STAGE2_EPOCHS, lr=STAGE2_LR,
        momentum=STAGE2_MOMENTUM, log_every_n_epochs=STAGE2_EPOCHS,  # minimal dyn logging
        device=device, seed=seed, verbose=False,
        checkpoint_dir=str(ckpt_dir), checkpoint_epochs=checkpoint_epochs,
    )

    # ---- recurrent IG over the saved checkpoints ----
    cyt_X, pbs_X = _tubes_by_class(train_dataset, max_tubes)
    if not cyt_X or not pbs_X:
        log(f"  SKIP {target}: missing cytokine/PBS tubes for IG")
        return []
    baseline_t = _pbs_baseline(pbs_X, device)

    ig_model = build_mil_model(
        copy.deepcopy(shared_encoder), embed_dim=embed_dim,
        attention_hidden_dim=attention_hidden_dim, n_classes=2, encoder_frozen=True,
    ).to(device)

    rows = []
    for epoch in checkpoint_epochs:
        ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        if not ckpt_path.exists():
            log(f"  WARN {target}: missing {ckpt_path.name}")
            continue
        ig_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        ranked = _ig_ranking(
            ig_model, cyt_X, baseline_t, gene_names,
            n_steps=n_steps, top_n=traj_top_n, device=device,
        )
        for gene, ig, rank in ranked:
            rows.append(
                {"cytokine": target, "gene": gene, "epoch": int(epoch),
                 "ig": ig, "rank_ig": int(rank), "seed": int(seed)}
            )

    if not keep_checkpoints:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    train_m_path.unlink(missing_ok=True)
    log(f"  {target}: {len(checkpoint_epochs)} checkpoints x top-{traj_top_n} IG rows")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Recurrent-IG Oesinghaus experiment (Part B)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--manifest_path", type=str, default=MANIFEST_PATH)
    p.add_argument("--hvg_path", type=str, default=HVG_PATH)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--n_ig_steps", type=int, default=20)
    p.add_argument("--traj_top_n", type=int, default=300,
                   help="Genes stored per checkpoint (the tracked band).")
    p.add_argument("--max_tubes", type=int, default=10)
    p.add_argument("--embed_dim", type=int, default=EMBED_DIM)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=list(HIDDEN_DIMS))
    p.add_argument("--attention_hidden_dim", type=int, default=ATTENTION_HIDDEN_DIM)
    p.add_argument("--keep_checkpoints", action="store_true")
    p.add_argument("--limit_cytokines", type=int, default=None,
                   help="Debug: cap number of cytokines.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(str(msg) + "\n")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    embed_dim = args.embed_dim
    hidden_dims = tuple(args.hidden_dims)
    attention_hidden_dim = args.attention_hidden_dim
    checkpoint_epochs = list(range(args.checkpoint_every, STAGE2_EPOCHS + 1, args.checkpoint_every))

    log("=" * 60)
    log("RECURRENT-IG OESINGHAUS EXPERIMENT (Part B)")
    log(f"seed={seed}  checkpoint_every={args.checkpoint_every}  epochs={checkpoint_epochs}")
    log(f"wide config: embed={embed_dim} hidden={hidden_dims} attn={attention_hidden_dim}")
    log(f"Stage1 {STAGE1_EPOCHS}@{STAGE1_LR}  Stage2 {STAGE2_EPOCHS}@{STAGE2_LR}")
    log(f"traj_top_n={args.traj_top_n}  max_tubes={args.max_tubes}")
    log(f"started {datetime.now().isoformat(timespec='seconds')}")
    log("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    with open(args.manifest_path) as fh:
        manifest = json.load(fh)
    with open(args.hvg_path) as fh:
        gene_names = json.load(fh)
    manifest_cytokines = {e["cytokine"] for e in manifest}

    targets = _target_cytokines(manifest_cytokines)
    if args.limit_cytokines:
        targets = targets[: args.limit_cytokines]
    log(f"Targets present in manifest: {len(targets)} / 45")
    log(f"  {targets}")

    filtered_manifest = filter_manifest(manifest, targets, include_pbs=True)

    # ---- shared Stage-1 encoder (once per seed) ----
    log("\nSHARED STAGE 1 — encoder pre-training")
    stage1_path = out_dir / "manifest_stage1_shared.json"
    stage1_manifest = build_stage1_manifest(filtered_manifest, save_path=str(stage1_path))
    log(f"Stage1 manifest: {len(stage1_manifest)} entries")
    cell_dataset = CellDataset(str(stage1_path), gene_names=gene_names, preload=True)
    cell_loader = torch.utils.data.DataLoader(
        cell_dataset, batch_size=256, shuffle=True, num_workers=0
    )
    encoder = InstanceEncoder(
        input_dim=len(gene_names), embed_dim=embed_dim,
        n_cell_types=cell_dataset.n_cell_types(), hidden_dims=hidden_dims,
    )
    train_encoder(
        encoder, cell_loader, n_epochs=STAGE1_EPOCHS, lr=STAGE1_LR,
        momentum=STAGE1_MOMENTUM, device=device,
    )
    torch.save(encoder.state_dict(), out_dir / "encoder_shared_stage1.pt")
    log("Saved encoder_shared_stage1.pt")

    # ---- per-cytokine training + recurrent IG ----
    ckpt_root = out_dir / "_checkpoints"
    all_rows = []
    for i, target in enumerate(targets, 1):
        log(f"\n[{i}/{len(targets)}] {target}")
        all_rows.extend(
            _run_one_cytokine(
                target, filtered_manifest, gene_names, encoder, out_dir, ckpt_root,
                device, seed, embed_dim=embed_dim,
                attention_hidden_dim=attention_hidden_dim,
                checkpoint_epochs=checkpoint_epochs, n_steps=args.n_ig_steps,
                traj_top_n=args.traj_top_n, max_tubes=args.max_tubes,
                keep_checkpoints=args.keep_checkpoints, log=log,
            )
        )

    if ckpt_root.exists() and not args.keep_checkpoints:
        shutil.rmtree(ckpt_root, ignore_errors=True)

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_dir / "ig_traj.parquet", index=False)
    final = df[(df["epoch"] == STAGE2_EPOCHS) & (df["rank_ig"] < 50)].copy()
    final.to_parquet(out_dir / "final_signatures.parquet", index=False)
    log(f"\nSaved ig_traj.parquet ({len(df)} rows) + final_signatures.parquet ({len(final)} rows)")
    log(f"done {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
