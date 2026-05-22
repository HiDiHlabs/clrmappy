"""Shared compute logic for the clrmappy Streamlit app + headless batch
script. Both `app.py` (foreground mode) and `compute_batch.py` (background
mode) import from here so file-naming and computation results stay identical.

Phase 2 caching model:
- Only the UMAPs (``emb2d.npy`` + ``emb3d.npy``) are cached on disk per combo.
- Colors are recomputed live by the app (cheap) and optionally persisted
  via the 💾 button in Section 4 — using the filename schema in
  ``color_file_stem``.
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# matplotlib must be configured BEFORE pyplot import
import matplotlib
matplotlib.use("Agg")

# Make the parent folder discoverable so `import clrmappy as cm` works no
# matter where the entry-point script is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scanpy as sc  # noqa: E402

import clrmappy as cm  # noqa: E402


# ── defaults — match what the OKhsl pipeline considers "no extra tweaks" ─────
DEFAULT_BRIGHTNESS_RANGE = [0.2, 0.8]
DEFAULT_SATURATION_RANGE = [0.0, 1.0]


def combo_key(md, nn, metric):
    """Folder name for one UMAP combination."""
    return f"md{md}_nn{nn}_m{metric}"


def _is_default_brightness(b):
    return b is None or list(b) == DEFAULT_BRIGHTNESS_RANGE


def _is_default_saturation(s):
    return s is None or list(s) == DEFAULT_SATURATION_RANGE


def _bright_sat_suffix(brightness, saturation):
    """Filename suffix encoding non-default brightness / saturation."""
    if _is_default_brightness(brightness) and _is_default_saturation(saturation):
        return ""
    b = brightness or DEFAULT_BRIGHTNESS_RANGE
    s = saturation or DEFAULT_SATURATION_RANGE
    return (f"_b{int(round(b[0]*100)):02d}-{int(round(b[1]*100)):02d}"
            f"_s{int(round(s[0]*100)):02d}-{int(round(s[1]*100)):02d}")


def color_file_stem(spec: dict) -> str:
    """Filename stem (no .npy extension) for a saved color array.

    Standard spec → short name (``okhsl_3d_iso1_pc0_eqv0``, ``rgb_eqv0``,
    ``cielab``). Custom brightness / saturation appends a suffix like
    ``_b35-85_s40-100``. When ``sat_enhance=False`` the saturation suffix
    is replaced with ``_noSat``.
    """
    mode = spec["mode"]
    if mode == "okhsl":
        base = spec.get("base", "3d")
        if base == "3d":
            iso = int(bool(spec.get("iso", True)))
            pc = int(bool(spec.get("pc_from_2d", False)))
            eqv = int(bool(spec.get("equal_var", False)))
            stem = f"okhsl_3d_iso{iso}_pc{pc}_eqv{eqv}"
        else:
            stem = "okhsl_2d"
        if spec.get("sat_enhance", True) is False:
            stem += "_noSat"
            if not _is_default_brightness(spec.get("brightness")):
                b = spec.get("brightness") or DEFAULT_BRIGHTNESS_RANGE
                stem += (f"_b{int(round(b[0]*100)):02d}"
                         f"-{int(round(b[1]*100)):02d}")
            return stem
        return stem + _bright_sat_suffix(
            spec.get("brightness"), spec.get("saturation"))
    if mode == "rgb":
        return f"rgb_eqv{int(bool(spec.get('equal_var', False)))}"
    if mode == "cielab":
        return "cielab"
    raise ValueError(f"Unknown color mode: {mode!r}")


def compute_color_full(spec: dict, emb2d, emb3d, center_around):
    """Compute ``(rgb_array, emb_fit_or_None)`` for one coloring spec
    using the public clrmappy package API.

    Returns
    -------
    (rgb : (N, 3) ndarray in [0, 1], emb_fit : (N, 2) | (N, 3) | None)
        ``emb_fit`` is the rotated cloud the OKhsl / RGB conversion was
        applied to. None for CIELab (U-CIE fits internally in R).
    """
    mode = spec["mode"]
    if mode == "okhsl":
        base = spec.get("base", "3d")
        brightness = spec.get("brightness", DEFAULT_BRIGHTNESS_RANGE)
        saturation = spec.get("saturation", DEFAULT_SATURATION_RANGE)
        sat_enhance = bool(spec.get("sat_enhance", True))
        if base == "3d":
            res = cm.emb_to_okhsl(
                emb_3d=emb3d,
                emb_2d=emb2d if spec.get("pc_from_2d") else None,
                iso_rot_scale=bool(spec.get("iso", True)),
                equal_variance_mode=bool(spec.get("equal_var", False)),
                pc1_and_2_from_2d=bool(spec.get("pc_from_2d", False)),
                brightness_range=list(brightness),
                saturation_enhancement=sat_enhance,
                saturation_range=list(saturation),
                center_around=center_around,
            )
        else:
            res = cm.emb_to_okhsl(
                emb_3d=None, emb_2d=emb2d,
                iso_rot_scale=False, equal_variance_mode=False,
                pc1_and_2_from_2d=False,
                brightness_range=list(brightness),
                saturation_enhancement=sat_enhance,
                saturation_range=list(saturation),
                center_around=center_around,
            )
        plt.close("all")
        return res["clrmappy"], res["emb_fit"]
    if mode == "rgb":
        res = cm.emb_to_rgb(
            emb3d, equal_variance_mode=bool(spec.get("equal_var", False)))
        return res["clrmappy"], res.get("emb3d_fit")
    if mode == "cielab":
        return cm.emb_to_cielab(embedding=emb3d), None
    raise ValueError(f"Unknown color mode: {mode!r}")


def preproc_subdir_name(setup) -> str:
    """Subdirectory under the user's output dir, one per preprocessing config."""
    return (
        f"preproc_mg{setup.get('min_genes', 0)}"
        f"_xg{setup.get('max_genes', 0)}"
        f"_mt{setup.get('mt_cutoff', 0)}"
        f"_mc{setup.get('min_cells', 0)}"
        f"_ntg{setup.get('n_top_genes', 0)}"
        f"_np{setup.get('n_pcs', 50)}"
    )


def compute_umaps_fast(adata, min_dist, n_neighbors, metric, n_pcs):
    """3D + 2D UMAP via scanpy. Requires ``X_pca`` already in adata."""
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric)
    sc.tl.umap(adata, n_components=3, min_dist=min_dist, random_state=10)
    emb3d = adata.obsm["X_umap"].copy()
    adata.obsm["X_umap_3d"] = emb3d.copy()

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, metric=metric)
    sc.tl.umap(adata, n_components=2, min_dist=min_dist, random_state=10)
    emb2d = adata.obsm["X_umap"].copy()
    adata.obsm["X_umap_2d"] = emb2d.copy()
    return emb2d, emb3d


# ── setup fingerprint (cache-validity tracking) ─────────────────────────────
SETUP_FILE = "setup.json"

SETUP_COMPARE_KEYS = [
    "input_filename",
    "skip_preprocess",
    "min_genes",
    "max_genes",
    "mt_cutoff",
    "min_cells",
    "n_top_genes",
    "n_pcs",
    "adata_fingerprint",
]


def compute_adata_fingerprint(adata, *, n_pcs=None) -> str:
    """Fast deterministic fingerprint of preprocessed adata + PCA state."""
    h = hashlib.sha256()
    h.update(f"shape={adata.shape}\n".encode())
    names = list(adata.obs_names.astype(str))
    h.update(b"obs_head:" + "|".join(names[:200]).encode())
    h.update(b"\nobs_tail:" + "|".join(names[-200:]).encode())
    if "X_pca" in adata.obsm:
        pca = adata.obsm["X_pca"]
        h.update(f"\npca_shape={pca.shape}\n".encode())
        h.update(pca[:5].tobytes())
        if len(pca) > 10:
            mid = len(pca) // 2
            h.update(pca[mid:mid + 5].tobytes())
            h.update(pca[-5:].tobytes())
        h.update(f"\npca_mean={float(pca.mean()):.8f}".encode())
        h.update(f"\npca_std={float(pca.std()):.8f}".encode())
    if n_pcs is not None:
        h.update(f"\nn_pcs={int(n_pcs)}".encode())
    return h.hexdigest()[:16]


def build_setup_dict(adata_base, *, input_filename, skip_preprocess,
                     min_genes, max_genes, mt_cutoff, min_cells, n_top_genes,
                     n_pcs) -> dict:
    return {
        "input_filename": input_filename,
        "skip_preprocess": bool(skip_preprocess),
        "min_genes": int(min_genes),
        "max_genes": int(max_genes),
        "mt_cutoff": int(mt_cutoff),
        "min_cells": int(min_cells),
        "n_top_genes": int(n_top_genes),
        "n_pcs": int(n_pcs),
        "adata_shape": list(adata_base.shape),
        "adata_fingerprint": compute_adata_fingerprint(
            adata_base, n_pcs=n_pcs),
        "created_at": time.time(),
    }


def write_setup(out_dir, setup: dict):
    """Atomic write of setup.json into out_dir."""
    p = Path(out_dir) / SETUP_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(setup, indent=2, default=str))
    os.replace(tmp, p)


def read_setup(out_dir):
    """Return the stored setup dict, or None if missing/unreadable."""
    p = Path(out_dir) / SETUP_FILE
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def setup_diff(stored, current) -> list:
    """Differing (key, stored_value, current_value) tuples."""
    if stored is None:
        return []
    diffs = []
    for k in SETUP_COMPARE_KEYS:
        sv = stored.get(k)
        cv = current.get(k)
        if sv != cv:
            diffs.append((k, sv, cv))
    return diffs


def process_combo(adata_base, md, nn, metric, n_pcs, out_dir,
                  *, skip_existing=True):
    """Compute (or load cached) UMAPs for ONE parameter combination.

    Phase 2: only ``emb2d.npy`` + ``emb3d.npy`` are persisted — colors are
    recomputed live by the app on demand. With ``skip_existing=True``
    (default), an existing pair on disk skips computation.

    Returns ``{"2d", "3d", "colors": {}, "stats": …}`` — the empty ``colors``
    dict is kept for API symmetry with callers that still expect the key.
    """
    ck = combo_key(md, nn, metric)
    cdir = Path(out_dir) / ck
    cdir.mkdir(parents=True, exist_ok=True)

    emb2d_path = cdir / "emb2d.npy"
    emb3d_path = cdir / "emb3d.npy"
    embeddings_on_disk = (
        skip_existing and emb2d_path.exists() and emb3d_path.exists())
    need_umap_compute = not embeddings_on_disk

    adata_local = None
    if need_umap_compute:
        adata_local = adata_base.copy()
        emb2d, emb3d = compute_umaps_fast(
            adata_local, md, nn, metric, n_pcs)
        np.save(emb2d_path, emb2d)
        np.save(emb3d_path, emb3d)
    else:
        emb2d = np.load(emb2d_path)
        emb3d = np.load(emb3d_path)

    if adata_local is not None:
        del adata_local

    return {
        "2d": emb2d,
        "3d": emb3d,
        "colors": {},
        "stats": {
            "computed_umap": need_umap_compute,
            "computed_colors": 0,
            "loaded_colors": 0,
        },
    }
