"""Step 6 — bridge: per-condition binary models -> Integrated-Gradients signatures.

For each stimulus we train a stimulus-vs-control AB-MIL, then attribute its logit
back to genes with Integrated Gradients. The top genes are that condition's
discovered signature S_X (no curated gene lists).

Run:  python 06_bridge_signatures.py
"""

from __future__ import annotations

from _common import banner, preprocessed, tube_set, trained_encoder, CONTROL


def main() -> None:
    banner("Step 6 — train_all_binary + derive_signatures (bridge)")
    proc = preprocessed()
    ts = tube_set()
    encoder = trained_encoder(proc)

    import cascadir as cd

    models = cd.train_all_binary(
        ts, encoder, control_label=CONTROL, epochs=40, device="cpu", seed=42
    )
    print(f"trained {len(models)} binary models: {list(models)}")

    signatures = cd.derive_signatures(models, ts, top_n=10, device="cpu")
    for cond, sig in signatures.items():
        print(f"\nS_{cond} (top {len(sig.genes)} genes by IG):")
        print(f"  {list(sig.genes)}")


if __name__ == "__main__":
    main()
