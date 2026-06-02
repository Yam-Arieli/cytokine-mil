"""Step 2 — preprocess (normalize + log1p + HVG), handling raw vs log-normalized.

`preprocess` detects the X state and applies the right branch; it refuses to
guess when it cannot tell (NotPreprocessedError with instructions).

Run:  python 02_preprocess.py
"""

from __future__ import annotations

from _common import banner, raw_anndata

import cascadir as cd


def main() -> None:
    banner("Step 2 — preprocess")
    adata = raw_anndata()

    print(f"raw input: is_raw_counts={cd.is_raw_counts(adata)}  "
          f"is_lognormalized={cd.is_lognormalized(adata)}")

    # raw counts -> stash counts, HVG, normalize_total + log1p
    proc = cd.preprocess(adata, assume="raw")
    print(f"after preprocess: {proc.n_obs} cells x {proc.n_vars} genes; "
          f"is_lognormalized={cd.is_lognormalized(proc)}  "
          f"has counts layer={'counts' in proc.layers}")

    # re-running on already-log-normalized data is a no-op (won't double-log)
    proc2 = cd.preprocess(proc, assume="auto")
    import numpy as np
    print(f"idempotent re-run unchanged: {np.allclose(proc.X, proc2.X, atol=1e-5)}")

    print("\nTip: if your data is ALREADY normalize_total+log1p'd, call "
          "preprocess(adata, assume='lognorm', "
          "preprocess... flavor='seurat') — seurat_v3 needs raw counts.")


if __name__ == "__main__":
    main()
