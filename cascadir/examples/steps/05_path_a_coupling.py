"""Step 5 — Path A: discover coupling axes (which pairs are linked).

Latent-space geometry on the encoder embeddings (PBS-RC + per-donor Wilcoxon).
Direction-agnostic existence test. NOTE: the donor-level test needs several
donors; on the 3-donor demo it is underpowered (flagged), so rank by
axis_strength rather than trusting `coupled`.

Run:  python 05_path_a_coupling.py
"""

from __future__ import annotations

from _common import banner, preprocessed, tube_set, trained_encoder


def main() -> None:
    banner("Step 5 — discover_axes (Path A)")
    proc = preprocessed()
    ts = tube_set()
    encoder = trained_encoder(proc)

    import cascadir as cd

    axes = cd.discover_axes(ts, encoder, device="cpu")
    print(axes.summary())
    print()
    print(axes.axes.to_string(index=False))


if __name__ == "__main__":
    main()
