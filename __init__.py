"""clrmappy — Unsupervised coloring of multi-dimensional data.

This file is the package entry point. When the user runs::

    import clrmappy as cm

Python loads THIS file. Anything imported here becomes accessible as
``cm.<name>``. Anything NOT imported here is still reachable via the
sub-module path (e.g. ``clrmappy._okhsl_utils._find_cusp``) but isn't part
of the documented public API.

Internal modules (prefixed with `_`) hold implementation details:
  * ``_okhsl_utils``   — internal OKhsl PCA / rotation / sRGB conversion
  * ``_plotting``      — plotting helpers (notebook-tauglich)
  * ``clrmap_main``    — the 3 main coloring entry points
  * ``single_cell_helper_functions`` — scanpy-based preprocessing + UMAP

Typical use:

    import clrmappy as cm

    # 1. Optional: preprocess single-cell data
    adata = cm.preprocess(adata, min_genes=20, max_genes=200,
                          min_cells=100, mt_cutoff=5)

    # 2. Compute UMAPs (or bring your own)
    res = cm.compute_umaps(adata)
    emb_2d, emb_3d = res['umap_2d'], res['umap_3d']

    # 3. Color with one of the 3 colorspaces
    rgb = cm.emb_to_okhsl(emb_3d=emb_3d, emb_2d=emb_2d)['OKhsl_rgb_array']
    rgb = cm.emb_to_rgb(emb_3d)['rgb']
    rgb = cm.emb_to_cielab(emb_3d)

    # 4. Plot
    cm.plot_emb_2d(emb_2d, rgb, title='Brain UMAP')
    cm.plot_emb_3d(emb_3d, rgb)
    cm.plot_spatial(adata, rgb)
"""
from __future__ import annotations

# ── 3 main coloring entry points ────────────────────────────────────────────
from .clrmap_main import (
    emb_to_rgb,
    emb_to_okhsl,
    emb_to_cielab,
)

# ── Plot functions (notebook-tauglich, work with any 2D/3D embedding) ───────
from ._plotting import (
    plot_emb_2d,
    plot_emb_3d,
    plot_spatial,
    plot_emb_2d_vs_celltype,
    plot_spatial_vs_celltype,
    plot_okhsl_fit,
    load_csv_annotations,
)

# ── Single-cell helpers (preprocessing + UMAP via scanpy) ───────────────────
from .single_cell_helper_functions import (
    preprocess,
    compute_umaps,
)

# Public API list — explicit so `from clrmappy import *` is well-defined and
# tools like Pylance / Pyright don't complain about unused imports above.
__all__ = [
    # coloring
    "emb_to_rgb",
    "emb_to_okhsl",
    "emb_to_cielab",
    # plotting
    "plot_emb_2d",
    "plot_emb_3d",
    "plot_spatial",
    "plot_emb_2d_vs_celltype",
    "plot_spatial_vs_celltype",
    "plot_okhsl_fit",
    "load_csv_annotations",
    # single-cell helpers
    "preprocess",
    "compute_umaps",
]
