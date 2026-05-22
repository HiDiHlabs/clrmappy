from __future__ import annotations
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

# IMPORTANT: matplotlib.use() must run BEFORE pyplot is imported.
import matplotlib
matplotlib.use("Agg")

# Make the `clrmappy` package (parent folder of this file) importable:
# this file lives INSIDE the package folder, so the parent dir must be on
# sys.path so Python can find the package by name. compute_core lives in
# the SAME folder as app.py, so the folder itself also needs to be on path.
_HERE = os.path.dirname(os.path.abspath(__file__))
# parent → enables `import clrmappy`
sys.path.insert(0, os.path.dirname(_HERE))
# self   → enables `import compute_core`
sys.path.insert(0, _HERE)

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import scanpy as sc  # noqa: E402
import streamlit as st  # noqa: E402
import colorcet as cc  # noqa: E402

import clrmappy as cm  # noqa: E402
from compute_core import (  # noqa: E402
    build_setup_dict,
    color_file_stem,
    combo_key,
    compute_color_full,
    preproc_subdir_name,
    process_combo,
    read_setup,
    setup_diff,
    write_setup,
)

st.set_page_config(page_title="clrmappy explorer", layout="wide")

# ── widget-default seeding ────────────────────────────────────────────────────
# Sidebar widgets are keyed (sb_*) so the Quick-View load flow can override
# their defaults by writing to st.session_state before they render. These
# defaults are only used the first time the app starts.
_DEFAULTS = {
    "sb_n_pcs": 50,
    "sb_center_around": "mid",
    "sb_min_dist_str": "0.01, 0.1, 0.3",
    "sb_n_neighbors_str": "15, 30, 50",
    "sb_m_euclidean": True,
    "sb_m_cosine": False,
    "sb_m_correlation": False,
    "sb_out_dir": "results/?",
    "qv_path": "",
    # Preprocessing widget defaults — also targets of the Quick-View
    # "_qv_pending_*" pattern below so loading an existing results folder
    # restores the exact preprocessing settings that produced it.
    "sb_skip_preprocess": False,
    "sb_min_genes": 20,
    "sb_max_genes": 200,
    "sb_mt_cutoff": 5,
    "sb_min_cells": 100,
    "sb_n_top_genes": 2000,
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# Apply any pending Quick-View overrides BEFORE the sidebar widgets render.
# (Streamlit refuses to mutate session_state[key] AFTER a widget with that
# key has been instantiated, so we stash overrides under `_qv_pending_*`
# during the Load click and apply them here at the top of the next run.)
_QV_PENDING_MAP = {
    "_qv_pending_min_dist_str": "sb_min_dist_str",
    "_qv_pending_n_neighbors_str": "sb_n_neighbors_str",
    "_qv_pending_m_euclidean": "sb_m_euclidean",
    "_qv_pending_m_cosine": "sb_m_cosine",
    "_qv_pending_m_correlation": "sb_m_correlation",
    "_qv_pending_out_dir": "sb_out_dir",
    "_qv_pending_n_pcs": "sb_n_pcs",
    "_qv_pending_qv_path": "qv_path",
    # Preprocessing widgets — required so the Save-coloring path writes
    # into the EXACT preproc folder that was loaded (otherwise the
    # _effective_out_dir is rebuilt from stale sidebar defaults).
    "_qv_pending_skip_preprocess": "sb_skip_preprocess",
    "_qv_pending_min_genes": "sb_min_genes",
    "_qv_pending_max_genes": "sb_max_genes",
    "_qv_pending_mt_cutoff": "sb_mt_cutoff",
    "_qv_pending_min_cells": "sb_min_cells",
    "_qv_pending_n_top_genes": "sb_n_top_genes",
}
for _pending, _target in _QV_PENDING_MAP.items():
    if _pending in st.session_state:
        st.session_state[_target] = st.session_state.pop(_pending)

# ── constants ─────────────────────────────────────────────────────────────────
SAT_LABELS = [
    "no enhancement",
    "range min–1.0 (std)",
    "range 0.2–1.0",
    "range 0.4–1.0",
    "range 0.6–1.0",
    "range 0.8–1.0",
]
SAT_MINS = [None, 0.0, 0.2, 0.4, 0.6, 0.8]


def color_descriptor(colorspace, pc_flag, sat_idx):
    """Human-readable descriptor used in plot titles and expander headers."""
    if colorspace == "okhsl":
        base = "OKhsl"
        if sat_idx == 0:
            sat_part = "no saturation enhancement"
        elif sat_idx == 1:
            sat_part = "saturation range: min–1.0 (std)"
        else:
            sat_part = f"saturation range: {SAT_MINS[sat_idx]}–1.0"
        parts = [sat_part]
    else:
        base = "RGB naive"
        parts = []
    if pc_flag:
        parts.append("pc1&2 from 2D UMAP")
    return f"{base} ({', '.join(parts)})" if parts else base


# ── pure helpers ──────────────────────────────────────────────────────────────
# combo_key, color_key, compute_* live in compute_core.py (shared with batch script)


def detect_celltype_col(adata):
    candidates = [
        "cell_type", "celltype", "class_name", "subcluster",
        "leiden", "louvain", "annotation", "label", "cluster",
    ]
    for c in candidates:
        if c in adata.obs.columns:
            return c
    return None


def glasbey_map(celltypes):
    unique = sorted(set(celltypes))
    cmap = cc.glasbey
    cmap_map = {ct: cmap[i % len(cmap)] for i, ct in enumerate(unique)}
    return unique, cmap_map, [cmap_map[ct] for ct in celltypes]


def to_hex(rgb_arr):
    rgb_arr = np.clip(rgb_arr, 0.0, 1.0)
    return [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b in rgb_arr
    ]


def _auto_dot(n):
    if n > 100_000:
        return 0.3
    if n > 50_000:
        return 0.6
    if n > 20_000:
        return 1.0
    return 2.0


def _fmt_duration(seconds: float) -> str:
    """Format seconds as human-readable duration: '45s', '12m 3s', '1h 5m'."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m"


def _slug(text: str) -> str:
    """Filesystem-safe lowercase-ish slug used in download filenames."""
    text = text.replace("–", "-").replace("→", "to").replace("&", "and")
    text = re.sub(r"[^\w.\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def fig_to_png_bytes(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes (uses the figure's own DPI)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def plot_filename(plot_kind, md, nn, metric, *,
                  color_descriptor=None, celltype_col=None) -> str:
    """Build a descriptive PNG filename.

    Examples:
      UMAP2D_md0.1_nn30_euclidean_OKhsl_saturation_range_0.2-1.0.png
      Spatial_md0.1_nn30_euclidean_RGB_naive_pc1and2_from_2D_UMAP.png
      UMAP2D_by_subcluster_md0.1_nn30_euclidean.png
    """
    parts = [plot_kind, f"md{md}", f"nn{nn}", str(metric)]
    if celltype_col:
        parts.insert(1, f"by_{celltype_col}")
    if color_descriptor:
        parts.append(color_descriptor)
    return _slug("_".join(parts)) + ".png"


def _render_plot(fig, filename: str):
    """Display fig in Streamlit, then offer a PNG download button. Closes fig."""
    st.pyplot(fig, use_container_width=True)
    png = fig_to_png_bytes(fig)
    st.download_button(
        "Download PNG",
        data=png,
        file_name=filename,
        mime="image/png",
        key=f"dl_{filename}",
        use_container_width=True,
    )
    plt.close(fig)


# Uniform figure size + axes layout so all 4 plots in the explorer (OKhsl/RGB
# colored UMAP+Spatial, celltype UMAP+Spatial) render at the exact same on-screen
# size. The axes occupy the same fraction in all figures; celltype panels reserve
# the bottom strip for a multi-column legend instead of taking that space off
# the plot.
_PLOT_FIGSIZE = (6, 7.0)        # uniform figsize for ALL four panels
_AXES_RECT = (0.10, 0.30, 0.86, 0.62)  # left, bottom, width, height (fraction)
# Legend top edge (figure coords). Spatial panels need a HIGHER value than
# UMAP panels to land at the visually-same position. Tweak independently.
_LEGEND_TOP_Y_UMAP = 0.21
_LEGEND_TOP_Y_SPATIAL = 0.26


def _make_axes(dpi):
    """Create a figure + axes at the fixed rect used by every plot in §4."""
    fig = plt.figure(figsize=_PLOT_FIGSIZE, dpi=dpi)
    ax = fig.add_axes(_AXES_RECT)
    return fig, ax


def _style_axes(ax, spatial):
    if spatial:
        # adjustable="datalim" keeps the AXES BOX at the size set by add_axes()
        # and pads the data with whitespace instead of shrinking the box.
        # → spatial panel ends up the same width × height as the UMAP panel.
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    else:
        ax.set_xlabel("UMAP 1", fontsize=7)
        ax.set_ylabel("UMAP 2", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6)


def make_scatter(coords, colors, title, spatial=False, dot_size=None, dpi=220):
    ds = dot_size or _auto_dot(len(coords))
    fig, ax = _make_axes(dpi)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, s=ds, linewidths=0,
        alpha=(1.0 if spatial else 0.7),
        rasterized=True,
    )
    ax.set_title(title, fontsize=9, pad=6, linespacing=1.3)
    _style_axes(ax, spatial)
    return fig


def make_celltype_scatter(coords, celltypes, title, spatial=False, dot_size=None, dpi=220):
    unique, cmap_map, ann_colors = glasbey_map(celltypes)
    ds = dot_size or _auto_dot(len(coords))
    fig, ax = _make_axes(dpi)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=ann_colors, s=ds, linewidths=0,
        alpha=(1.0 if spatial else 0.7),
        rasterized=True,
    )
    ax.set_title(title, fontsize=9, pad=6, linespacing=1.3)
    _style_axes(ax, spatial)
    # Legend anchored in FIGURE coordinates (not axes coords) so it sits at
    # the EXACT same y-position for both UMAP and Spatial panels — ax-relative
    # coords would otherwise differ because UMAP has xticks + "UMAP 1" xlabel
    # below the axes while Spatial has axis("off").
    patches = [mpatches.Patch(color=cmap_map[ct], label=ct) for ct in unique]
    ncol = min(5, max(2, len(unique)))
    ax.legend(
        handles=patches, title="Cell type",
        bbox_to_anchor=(
            0.5, _LEGEND_TOP_Y_SPATIAL if spatial else _LEGEND_TOP_Y_UMAP),
        bbox_transform=fig.transFigure,
        loc="upper center", ncol=ncol,
        fontsize=5, title_fontsize=6, markerscale=2, frameon=False,
    )
    return fig


# ── computation helpers ───────────────────────────────────────────────────────
# compute_umaps_fast, compute_okhsl_variants, compute_rgb_variant, process_combo
# all live in compute_core.py (imported at the top).


def run_all(adata_base, combos, n_pcs, out_dir, skip_existing=True):
    """Foreground (Streamlit) compute. Delegates per-combo UMAP computation to
    ``compute_core.process_combo`` (shared with ``compute_batch.py``). Colors
    are recomputed live in Section 4 — only UMAPs are persisted to disk."""
    color_cache, embedding_cache = {}, {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    prog = st.progress(0)
    status = st.empty()
    eta_box = st.empty()

    t_total_start = time.time()
    combo_durations: list[float] = []
    n_umap_skipped = 0
    n_umap_computed = 0

    for i, (md, nn, metric) in enumerate(combos):
        t_combo_start = time.time()

        if combo_durations:
            mean_t = sum(combo_durations) / len(combo_durations)
            eta_seconds = mean_t * (len(combos) - i)
            eta_box.info(
                f"⏱  **{_fmt_duration(eta_seconds)} remaining** "
                f"· avg {_fmt_duration(mean_t)}/combo · "
                f"elapsed {_fmt_duration(time.time() - t_total_start)}"
            )
        else:
            eta_box.info(
                "⏱  Estimating time after first combination finishes…")

        status.markdown(
            f"**{i+1}/{len(combos)}** — `min_dist={md}` "
            f"`n_neighbors={nn}` `metric={metric}`"
        )

        result = process_combo(
            adata_base, md, nn, metric, n_pcs, out_dir,
            skip_existing=skip_existing,
        )
        ck = combo_key(md, nn, metric)
        embedding_cache[ck] = {"2d": result["2d"], "3d": result["3d"]}
        color_cache[ck] = result["colors"]

        if result.get("stats", {}).get("computed_umap"):
            n_umap_computed += 1
        else:
            n_umap_skipped += 1

        combo_durations.append(time.time() - t_combo_start)
        prog.progress((i + 1) / len(combos))

    total_elapsed = time.time() - t_total_start
    eta_box.success(
        f"⏱  Total time: {_fmt_duration(total_elapsed)} · "
        f"UMAPs computed: {n_umap_computed} · "
        f"skipped (cached): {n_umap_skipped}"
    )
    status.success(f"Done! Results saved to `{out_dir}/`")
    return color_cache, embedding_cache


def load_from_disk(out_dir, combos):
    """Load precomputed UMAP arrays for the given combos from disk. Returns
    ``(color_cache, embedding_cache)`` — color_cache is empty (Phase 2:
    colors are recomputed live)."""
    out = Path(out_dir)
    color_cache, embedding_cache = {}, {}
    try:
        for md, nn, metric in combos:
            ck = combo_key(md, nn, metric)
            cdir = out / ck
            embedding_cache[ck] = {
                "2d": np.load(cdir / "emb2d.npy"),
                "3d": np.load(cdir / "emb3d.npy"),
            }
            color_cache[ck] = {}  # live recompute → no cached colors
    except Exception as e:
        st.error(f"Load failed: {e}")
        return None, None
    return color_cache, embedding_cache


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ── App title + subtitle (rendered ABOVE the setup expander) ─────────────────
st.title("🎨 Clrmappy — Single-Cell / Spatial-Omics App")
st.markdown(
    "<p style='font-size:1.05rem; color:#666; margin-top:-0.6em; "
    "margin-bottom:1.2em;'>"
    "Unsupervised coloring of multi-dimensional data"
    "</p>",
    unsafe_allow_html=True,
)

# ── SETUP block (was the sidebar in earlier versions) ────────────────────────
# Pre-compute parameters — preprocessing filters, UMAP combos, output dir.
# Wrapped in a top-of-page expander so it doesn't take up screen real estate
# during the exploration phase. The actual Section-4 explorer controls live
# in the sidebar instead (see below) so you can adjust them while scrolling
# through the plots.
with st.expander(
    "⚙️ Setup — preprocessing & UMAP parameters (one-time config)",
    expanded=False,
):
    st.markdown("##### Preprocessing")
    st.caption(
        "Quality control + normalization + HVG selection. Filters out low-quality "
        "cells/genes, log-normalizes counts, picks variable genes, regresses out "
        "library size & mt%, and z-scales the matrix."
    )
    skip_preprocess = st.checkbox(
        "Skip (data already preprocessed)",
        key="sb_skip_preprocess",
        help="Tick this if your dataset already has a `scaled` layer and/or "
             "`X_pca`. Skips filtering, normalization and HVG selection.",
    )
    pp1, pp2, pp3 = st.columns(3)
    with pp1:
        min_genes = st.number_input(
            "min_genes",
            min_value=0, step=1,
            key="sb_min_genes",
            help="Cells with fewer than this many detected genes are dropped "
                 "(low-quality / empty droplets).",
        )
        min_cells = st.number_input(
            "min_cells",
            min_value=1, step=1,
            key="sb_min_cells",
            help="Genes detected in fewer than this many cells are removed.",
        )
    with pp2:
        max_genes = st.number_input(
            "max_genes",
            min_value=1, step=1,
            key="sb_max_genes",
            help="Cells with MORE than this many detected genes are dropped "
                 "(likely doublets or contamination).",
        )
        n_top_genes = st.number_input(
            "n_top_genes",
            min_value=100, step=100,
            key="sb_n_top_genes",
            help="Number of highly variable genes (HVGs) for PCA & UMAP. "
                 "2000 is the scanpy default.",
        )
    with pp3:
        mt_cutoff = st.slider(
            "mt_cutoff (%)",
            min_value=0, max_value=100,
            key="sb_mt_cutoff",
            help="Cells with mitochondrial fraction above this percent are "
                 "dropped (stressed / dying cells). 5% is a common default.",
        )

    st.markdown("##### UMAP parameters")
    st.caption(
        "Compute UMAP for every `min_dist` × `n_neighbors` × `metric` "
        "combination. Section 4 below picks one combo at a time to render."
    )
    up1, up2 = st.columns(2)
    with up1:
        n_pcs = int(st.number_input(
            "n_pcs", min_value=5, step=5, key="sb_n_pcs",
            help="Number of PCA components used to build the UMAP neighbor "
                 "graph. 50 is the scanpy default.",
        ))
        min_dist_str = st.text_input(
            "min_dist (list separated by comma)",
            key="sb_min_dist_str",
            help="UMAP `min_dist`: small (0.0–0.1) → tight clusters, "
                 "large (0.3–0.99) → looser spread. Comma-separated.",
        )
    with up2:
        center_around = st.radio(
            "center_around (OKhsl PCA)",
            ["mid", "mean"], key="sb_center_around",
            help="`mid` (default): subtract midpoint of min/max. "
                 "`mean`: subtract per-axis mean.",
        )
        nn_str = st.text_input(
            "n_neighbors (list separated by comma)",
            key="sb_n_neighbors_str",
            help="UMAP `n_neighbors`: small (5–15) → local structure, "
                 "large (50–200) → global / continuous. Comma-separated.",
        )

    st.markdown("**Metrics** — distance metric(s) for the UMAP k-NN graph. "
                "Pick one or more.")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        m_euclidean = st.checkbox(
            "euclidean", key="sb_m_euclidean",
            help="L2 distance — default, fast, good baseline.",
        )
    with mc2:
        m_cosine = st.checkbox(
            "cosine", key="sb_m_cosine",
            help="Angle between vectors — popular for sparse data.",
        )
    with mc3:
        m_correlation = st.checkbox(
            "correlation", key="sb_m_correlation",
            help="1 − Pearson correlation. Captures co-expression patterns.",
        )

    st.markdown("##### Output directory")
    out_dir = st.text_input(
        "Path",
        key="sb_out_dir",
        help="Where UMAP embeddings (.npy) are saved. **Replace the `?`** "
             "with a name (e.g. `results/brain` or `results/my-run`).",
    )

# Parse comma-separated sidebar values into combo lists.
# COMPUTE iterates over the cartesian product; DISPLAY (Section 4) shows
# exactly ONE combo at a time, chosen via single-value pickers in Section 4.
try:
    min_dists = sorted({float(x.strip())
                        for x in min_dist_str.split(",") if x.strip()})
except Exception:
    min_dists = [0.1]
try:
    nn_list = sorted({int(x.strip())
                      for x in nn_str.split(",") if x.strip()})
except Exception:
    nn_list = [30]

metrics = [m for m, sel in [
    ("euclidean", m_euclidean), ("cosine", m_cosine),
    ("correlation", m_correlation)
] if sel] or ["euclidean"]

combos = list(product(min_dists, nn_list, metrics))

# ── Output-dir hard block: refuse to run while the placeholder is unchanged ──
out_dir_unset = (
    not out_dir.strip()
    or "?" in out_dir
    or out_dir.strip().rstrip("/") in {"results", "Results", "."}
)

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
for k in ["raw_adata", "adata_base", "color_cache", "embedding_cache",
          "celltype_col", "loaded_filename"]:
    if k not in st.session_state:
        st.session_state[k] = None


# ── QUICK-VIEW: load an existing results folder ───────────────────────────────
# Skip Sections 1-3 entirely by pointing the app at a previously-computed
# preproc folder (one that has `_input.h5ad` next to combo subdirs).
def _parse_combo_dir_name(name: str):
    """``md0.01_nn15_meuclidean`` → (0.01, 15, ``euclidean``) or None."""
    m = re.match(r"^md([\d.]+)_nn(\d+)_m(.+)$", name)
    if not m:
        return None
    try:
        return float(m.group(1)), int(m.group(2)), m.group(3)
    except ValueError:
        return None


def _pick_folder_native():
    """Native macOS Finder folder picker via AppleScript subprocess.

    AppleScript via ``osascript`` runs out-of-process, so it does NOT block
    or crash the Streamlit server (which tkinter would on macOS). Returns
    the absolute folder path, or None if the user cancelled.
    """
    if sys.platform != "darwin":
        return None
    try:
        script = ('POSIX path of '
                  '(choose folder with prompt "Choose preproc folder")')
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            return result.stdout.strip().rstrip("/")
    except Exception:
        pass
    return None


def _quick_view_load(folder_str: str):
    """Load adata + every combo subdir from a preproc folder. Returns
    ``(ok: bool, message: str, payload: dict | None)``."""
    p = Path(folder_str).expanduser()
    if not p.exists():
        return False, f"Path does not exist: `{p}`", None

    # If the user pointed at a combo folder (has embeddings), use its parent.
    if (p / "emb2d.npy").exists() or (p / "emb3d.npy").exists():
        preproc_dir = p.parent
    else:
        preproc_dir = p

    adata_path = preproc_dir / "_input.h5ad"
    if not adata_path.exists():
        return False, (
            f"No `_input.h5ad` found in `{preproc_dir}`. The Quick-View flow "
            "requires the preprocessed AnnData next to the combo folders."
        ), None

    # Scan combo subdirs
    found = []
    for sub in sorted(preproc_dir.iterdir()):
        if not sub.is_dir():
            continue
        parsed = _parse_combo_dir_name(sub.name)
        if not parsed:
            continue
        if (sub / "emb2d.npy").exists() and (sub / "emb3d.npy").exists():
            found.append((*parsed, sub))
    if not found:
        return False, (
            f"No combo subdirs with `emb2d.npy` + `emb3d.npy` found in "
            f"`{preproc_dir}`."
        ), None

    # Read setup.json if available (for n_pcs + preproc params display)
    stored_setup = read_setup(str(preproc_dir))

    return True, "", {
        "preproc_dir": preproc_dir,
        "adata_path": adata_path,
        "found_combos": found,  # list of (md, nn, metric, Path)
        "stored_setup": stored_setup,
    }


with st.expander(
    "📂 **Quick view** — load an existing results folder directly",
    expanded=False,
):
    st.caption(
        "Skip Sections 1–3 by pointing at a preproc folder that already has "
        "`_input.h5ad` + combo subdirs. The app loads the AnnData and all "
        "UMAP embeddings in one click — colorings are recomputed live in "
        "Section 4. Useful for "
        "revisiting past runs without re-uploading raw data."
    )
    col_browse, col_path = st.columns([1, 4])
    with col_browse:
        st.markdown("&nbsp;")  # vertical alignment with text_input label
        if st.button("📂 Browse…", key="qv_browse_btn",
                     help="Open the native macOS folder picker"):
            folder = _pick_folder_native()
            if folder:
                # Stash as pending so the text_input picks it up on rerun
                # (avoids the "cannot mutate after widget instantiated" error).
                st.session_state["_qv_pending_qv_path"] = folder
                st.rerun()
            else:
                st.info("No folder selected (or picker unavailable on this OS).")
    with col_path:
        qv_path = st.text_input(
            "Path to preproc folder (or any combo inside it)",
            placeholder=("e.g. Results/liver/"
                         "preproc_mg100_xg1000_mt5_mc100_ntg2000_np50"),
            key="qv_path",
        )
    if st.button(
        "⟳ Load this folder",
        disabled=(not qv_path.strip()),
        type="primary",
        key="qv_load_btn",
    ):
        ok, msg, payload = _quick_view_load(qv_path.strip())
        if not ok:
            st.error(msg)
        else:
            preproc_dir = payload["preproc_dir"]
            found_combos = payload["found_combos"]

            with st.spinner(f"Loading AnnData ({payload['adata_path'].name})…"):
                qv_adata = sc.read_h5ad(payload["adata_path"])

            # Load UMAPs only — colors are recomputed live in Section 4.
            cc_loaded = {}
            ec_loaded = {}
            for md, nn, m, sub in found_combos:
                ck = combo_key(md, nn, m)
                try:
                    ec_loaded[ck] = {
                        "2d": np.load(sub / "emb2d.npy"),
                        "3d": np.load(sub / "emb3d.npy"),
                    }
                except Exception as e:
                    st.warning(f"Skipping `{ck}`: {e}")
                    continue
                cc_loaded[ck] = {}

            # Push core data into session_state (these are NOT widget keys)
            st.session_state["raw_adata"] = qv_adata
            st.session_state["adata_base"] = qv_adata
            st.session_state["color_cache"] = cc_loaded
            st.session_state["embedding_cache"] = ec_loaded
            st.session_state["loaded_filename"] = payload["adata_path"].name
            st.session_state["filter_stats"] = None

            # Sidebar widget overrides: stash as `_qv_pending_*` so they're
            # applied at the top of the NEXT run (before the widgets render).
            all_mds = sorted({md for md, _, _, _ in found_combos})
            all_nns = sorted({nn for _, nn, _, _ in found_combos})
            all_metrics = {m for _, _, m, _ in found_combos}

            st.session_state["_qv_pending_min_dist_str"] = ", ".join(
                str(x) for x in all_mds)
            st.session_state["_qv_pending_n_neighbors_str"] = ", ".join(
                str(x) for x in all_nns)
            st.session_state["_qv_pending_m_euclidean"] = (
                "euclidean" in all_metrics)
            st.session_state["_qv_pending_m_cosine"] = "cosine" in all_metrics
            st.session_state["_qv_pending_m_correlation"] = (
                "correlation" in all_metrics)
            st.session_state["_qv_pending_out_dir"] = str(preproc_dir.parent)
            # Restore the preprocessing settings from setup.json so the
            # _effective_out_dir resolves back to THIS preproc folder when
            # the user later saves a coloring or re-runs computation —
            # otherwise stale sidebar defaults would create a sibling
            # `preproc_…` folder next to the loaded one.
            ss = payload["stored_setup"]
            if ss:
                _SETUP_TO_PENDING = {
                    "n_pcs": "_qv_pending_n_pcs",
                    "min_genes": "_qv_pending_min_genes",
                    "max_genes": "_qv_pending_max_genes",
                    "mt_cutoff": "_qv_pending_mt_cutoff",
                    "min_cells": "_qv_pending_min_cells",
                    "n_top_genes": "_qv_pending_n_top_genes",
                    "skip_preprocess": "_qv_pending_skip_preprocess",
                }
                for sk, pk in _SETUP_TO_PENDING.items():
                    if sk in ss and ss[sk] is not None:
                        v = ss[sk]
                        # Cast numerics to int; skip_preprocess stays bool
                        if sk != "skip_preprocess":
                            try:
                                v = int(v)
                            except (TypeError, ValueError):
                                continue
                        st.session_state[pk] = v

            st.success(
                f"Loaded **{len(found_combos)}** combo(s) from "
                f"`{preproc_dir}`. Sidebar updated. Scroll to "
                "Section 4 below to pick which combo to display."
            )
            st.rerun()

    # Show available combos in the currently-loaded folder, if any
    if st.session_state.get("color_cache") and st.session_state.get("embedding_cache"):
        _loaded_cks = list(st.session_state["color_cache"].keys())
        if len(_loaded_cks) > 1:
            st.caption(
                f"💡 {len(_loaded_cks)} combos cached — switch via the "
                "sidebar's `min_dist` / `n_neighbors` / `metric` controls. "
                "Available: " + ", ".join(f"`{ck}`" for ck in _loaded_cks[:10])
                + (" …" if len(_loaded_cks) > 10 else "")
            )


# ── PACKAGE DEMO — color any 2D / 3D embedding (self-contained box) ──────────
# Visually + functionally set apart from the single-cell pipeline above.
# Has its OWN inline color picker and renders output IN-BOX (uploads do NOT
# touch the main pipeline's session state — Sections 1-4 remain untouched).
st.divider()
with st.container(border=True):
    st.markdown(
        "### 🧪 Demonstration of the `clrmappy` package — color any 2D / 3D data"
    )
    st.markdown(
        "This branch is **fully independent from the single-cell pipeline "
        "above**. Drop in any coordinate array — UMAP, PCA, t-SNE, MDS, your "
        "own embedding — and clrmappy colors it right here in this box. "
        "Works for biology, physics, materials, genomics, anything with a "
        "2D / 3D point cloud."
    )
    st.caption(
        "Drop a 2D file alone → OKhsl-2D coloring · drop a 3D file alone → "
        "OKhsl-3D / RGB / CIELab · drop both → everything available."
    )

    qu_cols = st.columns(2)
    with qu_cols[0]:
        qu_2d_file = st.file_uploader(
            "2D embedding (.npy)  — shape (N, 2)",
            type=["npy"], key="qu_2d_uploader",
            help="Any 2D coordinate array (UMAP / PCA / t-SNE / MDS / …).",
        )
    with qu_cols[1]:
        qu_3d_file = st.file_uploader(
            "3D embedding (.npy)  — shape (N, 3)",
            type=["npy"], key="qu_3d_uploader",
            help="Any 3D coordinate array.",
        )

    if st.button(
        "⟳ Use this embedding",
        type="primary",
        disabled=(qu_2d_file is None and qu_3d_file is None),
        key="qu_load_btn",
    ):
        try:
            qu_2d = np.load(qu_2d_file) if qu_2d_file is not None else None
            qu_3d = np.load(qu_3d_file) if qu_3d_file is not None else None
            if qu_2d is not None and (qu_2d.ndim != 2 or qu_2d.shape[1] != 2):
                st.error(
                    f"2D embedding shape must be (N, 2), got {qu_2d.shape}.")
                st.stop()
            if qu_3d is not None and (qu_3d.ndim != 2 or qu_3d.shape[1] != 3):
                st.error(
                    f"3D embedding shape must be (N, 3), got {qu_3d.shape}.")
                st.stop()
            if (qu_2d is not None and qu_3d is not None
                    and qu_2d.shape[0] != qu_3d.shape[0]):
                st.error(
                    f"2D and 3D files have different N: "
                    f"{qu_2d.shape[0]} vs {qu_3d.shape[0]}.")
                st.stop()
            # Stash in demo-only session state — does NOT touch raw_adata /
            # adata_base / color_cache used by the main pipeline.
            st.session_state["_demo_emb2d"] = qu_2d
            st.session_state["_demo_emb3d"] = qu_3d
            st.rerun()
        except Exception as e:
            st.error(f"Could not parse uploaded file(s): {e}")

    # ── If a demo embedding is loaded → inline color picker + plots ──
    _demo_e2 = st.session_state.get("_demo_emb2d")
    _demo_e3 = st.session_state.get("_demo_emb3d")
    if _demo_e2 is not None or _demo_e3 is not None:
        n_pts = (_demo_e2.shape[0] if _demo_e2 is not None
                 else _demo_e3.shape[0])
        st.divider()
        st.markdown(
            f"**Loaded:** {n_pts:,} points ·  "
            f"{'2D ✓' if _demo_e2 is not None else '2D —'}  "
            f"{'3D ✓' if _demo_e3 is not None else '3D —'}"
        )

        # Available colorspaces depend on what was uploaded.
        # OKhsl always available (uses whichever embedding has data, with a
        # 2D / 3D base picker below); RGB / CIELab need the 3D embedding.
        _opts = ["okhsl"]
        if _demo_e3 is not None:
            _opts.extend(["rgb", "cielab"])
        _labels = {
            "okhsl": "OKhsl",
            "rgb": "RGB (naive)",
            "cielab": "CIELab (U-CIE)",
        }
        d_top = st.columns([2, 1])
        with d_top[0]:
            demo_cs = st.radio(
                "Colorspace", _opts,
                format_func=lambda x: _labels[x],
                horizontal=True, key="demo_cs",
            )
        with d_top[1]:
            demo_dpi = int(st.number_input(
                "DPI", value=180, min_value=80, max_value=400, step=20,
                key="demo_dpi"))

        # For OKhsl: choose 2D or 3D base (independent of what's uploaded —
        # if only one is present, the other base option is disabled).
        if demo_cs == "okhsl":
            # Decide which base options are available
            _base_opts = []
            if _demo_e3 is not None:
                _base_opts.append("3d")
            if _demo_e2 is not None:
                _base_opts.append("2d")
            if len(_base_opts) >= 2:
                _demo_base_choice = st.radio(
                    "Calculate colors from",
                    _base_opts,
                    format_func=lambda x: {"3d": "3D embedding",
                                           "2d": "2D embedding"}[x],
                    horizontal=True, key="demo_okhsl_base",
                )
            else:
                _demo_base_choice = _base_opts[0]
                st.caption(
                    f"OKhsl base: **{_demo_base_choice.upper()} embedding** "
                    f"(the other dimensionality wasn't uploaded).")
        else:
            _demo_base_choice = "3d"  # rgb/cielab always 3D

        # Per-mode params + spec assembly
        demo_spec = None
        if demo_cs == "okhsl":
            # 3D-only toggles — only when base is 3D
            d_iso = True
            d_pc = False
            d_eqv = False
            if _demo_base_choice == "3d":
                t_cols = st.columns(3)
                with t_cols[0]:
                    d_iso = st.toggle(
                        "Saturation optimization algorithm",
                        value=True, key="demo_okhsl_iso",
                        help="Rotates the PCA to maximize saturation without "
                             "distorting the brightness range. (iso_rot_scale)",
                    )
                with t_cols[1]:
                    d_pc = st.toggle(
                        "Use 2D PCs for hue",
                        value=False,
                        disabled=(_demo_e2 is None), key="demo_okhsl_pc",
                        help="Replace PC1/PC2 with the 2D embedding's PCs "
                             "(needs the 2D file). (pc1_and_2_from_2dumap)",
                    )
                with t_cols[2]:
                    d_eqv = st.toggle(
                        "Equal-variance rotation",
                        value=False, key="demo_okhsl_eqv",
                        help="Fixed 45° rotation around all axes for balanced "
                             "channel variance. (equal_variance_mode)",
                    )
            s_cols = st.columns(2)
            with s_cols[0]:
                d_sat_on = st.checkbox(
                    "Apply saturation enhancement",
                    value=True, key="demo_okhsl_sat_on")
                d_sat = list(st.slider(
                    "Saturation range",
                    min_value=0.0, max_value=1.0, value=(0.0, 1.0),
                    step=0.05, key="demo_okhsl_sat",
                    disabled=not d_sat_on))
            with s_cols[1]:
                _bright_label = (
                    "Brightness range"
                    if _demo_base_choice == "3d"
                    else "Brightness range (constant = mid of range)"
                )
                d_bright = list(st.slider(
                    _bright_label,
                    min_value=0.0, max_value=1.0, value=(0.2, 0.8),
                    step=0.05, key="demo_okhsl_bright"))
            demo_spec = {
                "mode": "okhsl", "base": _demo_base_choice,
                "iso": d_iso, "pc_from_2d": d_pc, "equal_var": d_eqv,
                "brightness": d_bright, "saturation": d_sat,
                "sat_enhance": d_sat_on,
            }
        elif demo_cs == "rgb":
            d_eqv = st.toggle(
                "Equal-variance rotation",
                value=False, key="demo_rgb_eqv",
                help="Rotates the 3D PCA so every R/G/B channel ends up "
                     "with similar variance. (equal_variance_mode)",
            )
            demo_spec = {"mode": "rgb", "equal_var": d_eqv}
        else:  # cielab
            st.caption(
                "CIELab via the R `ucie` package "
                "(Kourmpetis et al., mikelkou/ucie) — no tunable parameters.")
            demo_spec = {"mode": "cielab"}

        # Compute color (live, no caching for demo)
        try:
            demo_rgb, demo_fit = compute_color_full(
                demo_spec, _demo_e2, _demo_e3, "mid")
        except Exception as _de:
            st.error(f"Color computation failed: {_de}")
            demo_rgb, demo_fit = None, None

        if demo_rgb is not None:
            demo_hex = to_hex(demo_rgb)
            st.divider()
            st.markdown("#### Output")
            # 2D scatter (only if 2D available). Place it in a narrow column
            # so it doesn't dominate the container — the 3D plot below is the
            # main attraction in the demo.
            if _demo_e2 is not None:
                _l_col, _ = st.columns([1, 1])
                with _l_col:
                    fig_demo_2d = make_scatter(
                        _demo_e2, demo_hex, "2D embedding — colored",
                        dot_size=None, dpi=demo_dpi,
                    )
                    st.pyplot(fig_demo_2d, use_container_width=True)
                    plt.close(fig_demo_2d)
            # 3D interactive (only if 3D available)
            if _demo_e3 is not None:
                _dn = len(_demo_e3)
                _ds = 1.5 if _dn < 30_000 else (1.0 if _dn < 100_000 else 0.6)
                fig_demo_3d = go.Figure(data=[go.Scatter3d(
                    x=_demo_e3[:, 0], y=_demo_e3[:, 1], z=_demo_e3[:, 2],
                    mode="markers",
                    marker=dict(size=_ds, color=demo_hex, opacity=0.85),
                    hovertemplate=(
                        "x: %{x:.2f}<br>"
                        "y: %{y:.2f}<br>"
                        "z: %{z:.2f}<extra></extra>"
                    ),
                )])
                fig_demo_3d.update_layout(
                    title="3D embedding — colored (rotate / zoom / pan)",
                    scene=dict(aspectmode="data"),
                    height=600, margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig_demo_3d, use_container_width=True,
                                key="demo_3d_plot")
            # Fit embedding (OKhsl + RGB only; CIELab has no fit)
            if demo_fit is not None and demo_spec.get("mode") in ("okhsl", "rgb"):
                with st.expander("Fit embedding (what the colors are based on)",
                                 expanded=False):
                    if demo_fit.shape[1] == 3:
                        _ax_titles = (("r", "g", "b")
                                      if demo_spec["mode"] == "rgb"
                                      else ("x", "y", "z (brightness)"))
                        fig_fit = go.Figure(data=[go.Scatter3d(
                            x=demo_fit[:, 0], y=demo_fit[:, 1],
                            z=demo_fit[:, 2],
                            mode="markers",
                            marker=dict(size=_ds, color=demo_hex,
                                        opacity=0.85),
                        )])
                        fig_fit.update_layout(
                            title="Fit embedding",
                            scene=dict(
                                xaxis_title=_ax_titles[0],
                                yaxis_title=_ax_titles[1],
                                zaxis_title=_ax_titles[2],
                                aspectmode="data",
                            ),
                            height=500, margin=dict(l=0, r=0, t=40, b=0),
                        )
                        st.plotly_chart(fig_fit, use_container_width=True,
                                        key="demo_fit_plot")
                    else:
                        fig_fit2d = make_scatter(
                            demo_fit, demo_hex,
                            "Fit embedding (2D)",
                            dot_size=None, dpi=demo_dpi,
                        )
                        st.pyplot(fig_fit2d, use_container_width=True)
                        plt.close(fig_fit2d)

        # Reset
        st.divider()
        if st.button("← Reset demo", key="demo_reset_btn"):
            for _k in ["_demo_emb2d", "_demo_emb3d"]:
                if _k in st.session_state:
                    st.session_state[_k] = None
            st.rerun()


st.header("1 — Load dataset")

uploaded = st.file_uploader("Drag & drop an .h5ad file", type=["h5ad"])

if uploaded and st.session_state["loaded_filename"] != uploaded.name:
    with st.spinner("Reading file…"):
        with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
            f.write(uploaded.read())
            tmp = f.name
        raw = sc.read_h5ad(tmp)
        os.unlink(tmp)
    st.session_state["raw_adata"] = raw
    st.session_state["loaded_filename"] = uploaded.name
    st.session_state["adata_base"] = None
    st.session_state["color_cache"] = None
    st.session_state["embedding_cache"] = None

if st.session_state["raw_adata"] is None:
    st.stop()

raw = st.session_state["raw_adata"]
has_spatial = "spatial" in raw.obsm
has_scaled = "scaled" in raw.layers
has_pca = "X_pca" in raw.obsm

c1, c2, c3 = st.columns(3)
c1.metric("Cells", f"{raw.shape[0]:,}")
c2.metric("Genes", f"{raw.shape[1]:,}")
c3.metric("Spatial", "yes" if has_spatial else "no")

# ── Optional: upload a separate CSV with cell annotations ─────────────────────
with st.expander("Upload separate annotation CSV (optional)", expanded=False):
    st.caption(
        "Use this when your cell-type / cluster labels live in a CSV instead of "
        "`adata.obs`. The CSV is merged into `obs` by matching cell IDs, then "
        "appears in the dropdown below."
    )
    ann_csv = st.file_uploader(
        "CSV file with annotations",
        type=["csv"],
        key="ann_csv_uploader",
    )

    if ann_csv is not None:
        try:
            ann_csv.seek(0)
            preview = pd.read_csv(ann_csv, nrows=5)
            st.write("Preview:")
            st.dataframe(preview, use_container_width=True)

            ann_csv.seek(0)
            full_cols = list(pd.read_csv(ann_csv, nrows=0).columns)

            ca, cb, cc_col = st.columns(3)
            with ca:
                id_choice = st.selectbox(
                    "Cell-ID column (must match `adata.obs_names`)",
                    ["(use first column)"] + full_cols,
                    index=0,
                )
            with cb:
                # default to last column (often the annotation)
                ann_default = len(full_cols) - 1 if full_cols else 0
                ann_choice = st.selectbox(
                    "Annotation column to import",
                    full_cols,
                    index=ann_default,
                )
            with cc_col:
                new_name = st.text_input(
                    "Name for new `obs` column",
                    value=ann_choice or "annotation",
                )

            replace_str = st.text_area(
                "Optional value replacements (one `old → new` per line)",
                value="",
                help="Same idea as the dict in `liver-pipeline.ipynb`. "
                     "Example: `Hepatocyte_1 → Hepa 1_3` per line. Leave empty to skip.",
                height=80,
            )

            if st.button("Add annotations to dataset", type="primary"):
                ann_csv.seek(0)
                df = pd.read_csv(ann_csv)
                if id_choice == "(use first column)":
                    df = df.set_index(df.columns[0])
                else:
                    df = df.set_index(id_choice)
                df.index = df.index.astype(str)

                # parse optional replacements
                if replace_str.strip():
                    rep_map = {}
                    for line in replace_str.splitlines():
                        if "→" in line:
                            k, v = line.split("→", 1)
                        elif "->" in line:
                            k, v = line.split("->", 1)
                        else:
                            continue
                        rep_map[k.strip()] = v.strip()
                    if rep_map:
                        df = df.replace(rep_map)

                merged_raw = (
                    df[ann_choice]
                    .reindex(raw.obs_names.astype(str))
                    .fillna("unknown")
                )
                raw.obs[new_name] = merged_raw.values
                n_match = int((merged_raw != "unknown").sum())

                # If preprocessing already ran, mirror into adata_base too
                ab = st.session_state.get("adata_base")
                if ab is not None:
                    merged_ab = (
                        df[ann_choice]
                        .reindex(ab.obs_names.astype(str))
                        .fillna("unknown")
                    )
                    ab.obs[new_name] = merged_ab.values

                st.success(
                    f"Added column `{new_name}` to `obs` — matched "
                    f"{n_match:,} / {raw.shape[0]:,} cells."
                )
                st.rerun()

        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

detected_ct = detect_celltype_col(raw)
obs_cols = list(raw.obs.columns)
default_ct_idx = (obs_cols.index(detected_ct) +
                  1) if detected_ct and detected_ct in obs_cols else 0
ct_col = st.selectbox(
    "Cell type annotation column (optional — leave blank if none)",
    ["— none —"] + obs_cols,
    index=default_ct_idx,
)
st.session_state["celltype_col"] = ct_col if ct_col != "— none —" else None


# ── SECTION 2: Preprocess + PCA ───────────────────────────────────────────────
st.header("2 — Preprocess & compute PCA")

if st.button("▶ Run preprocessing + PCA", type="primary"):
    n_cells_before = raw.shape[0]
    n_genes_before = raw.shape[1]
    filter_stats = None
    with st.spinner("Working…"):
        if skip_preprocess:
            if has_pca:
                adata_base = raw.copy()
                st.info("Using existing X_pca (skipped preprocessing and PCA).")
            elif has_scaled:
                adata_base = raw.copy()
                sc.pp.pca(adata_base, layer="scaled",
                          svd_solver="arpack", n_comps=n_pcs)
            else:
                st.error(
                    "Cannot skip preprocessing: dataset has neither `X_pca` nor `scaled` layer."
                )
                st.stop()
        else:
            adata_base, filter_stats = cm.preprocess(
                raw.copy(),
                min_genes=int(min_genes),
                min_cells=int(min_cells),
                max_genes=int(max_genes),
                mt_cutoff=int(mt_cutoff),
                n_top_genes=int(n_top_genes),
                return_stats=True,
            )
            sc.pp.pca(adata_base, layer="scaled",
                      svd_solver="arpack", n_comps=n_pcs)
    st.session_state["adata_base"] = adata_base
    st.session_state["filter_stats"] = filter_stats
    st.session_state["color_cache"] = None
    st.session_state["embedding_cache"] = None

# ── Display preprocessing summary (persists across reruns) ────────────────
if st.session_state["adata_base"] is not None:
    _ab = st.session_state["adata_base"]
    n_cells_before = raw.shape[0]
    n_genes_before = raw.shape[1]
    n_cells_after = _ab.shape[0]
    n_genes_after = _ab.shape[1]
    cells_dropped = n_cells_before - n_cells_after
    genes_dropped = n_genes_before - n_genes_after
    cells_pct = 100 * cells_dropped / n_cells_before if n_cells_before else 0
    genes_pct = 100 * genes_dropped / n_genes_before if n_genes_before else 0

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Cells kept",
        f"{n_cells_after:,}",
        delta=f"−{cells_dropped:,} ({cells_pct:.1f}%)" if cells_dropped else None,
        delta_color="inverse",
        help=f"Before filtering: {n_cells_before:,} cells.",
    )
    m2.metric(
        "Genes kept",
        f"{n_genes_after:,}",
        delta=f"−{genes_dropped:,} ({genes_pct:.1f}%)" if genes_dropped else None,
        delta_color="inverse",
        help=f"Before filtering: {n_genes_before:,} genes.",
    )
    m3.metric("PCA components", f"{n_pcs}")

    # Per-step filtering breakdown
    _fstats = st.session_state.get("filter_stats")
    if _fstats:
        st.markdown("**Per-step filtering breakdown**")
        rows = []
        for s in _fstats:
            rows.append({
                "Step": s["step"],
                "Cells after": f"{s['n_cells']:,}",
                "Cells dropped": (
                    f"−{s['dropped_cells']:,}" if s["dropped_cells"] else "—"
                ),
                "Genes after": f"{s['n_genes']:,}",
                "Genes dropped": (
                    f"−{s['dropped_genes']:,}" if s["dropped_genes"] else "—"
                ),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
        # Headline summary
        st.caption(
            f"Total: {cells_dropped:,} cells dropped ({cells_pct:.1f}%) and "
            f"{genes_dropped:,} genes dropped ({genes_pct:.1f}%) across all "
            f"filtering steps."
        )
    else:
        st.caption(
            f"Preprocessing skipped — using the dataset as loaded "
            f"({n_cells_after:,} cells × {n_genes_after:,} genes)."
        )

if st.session_state["adata_base"] is None:
    st.stop()

adata_base = st.session_state["adata_base"]


# ── SECTION 3: Compute ────────────────────────────────────────────────────────
st.header("3 — Compute UMAPs")

# Build the current setup (preprocessing fingerprint). center_around is NOT
# part of this — it lives in OKhsl filenames so multiple values coexist.
_current_setup = build_setup_dict(
    adata_base,
    input_filename=st.session_state.get("loaded_filename") or "",
    skip_preprocess=skip_preprocess,
    min_genes=min_genes,
    max_genes=max_genes,
    mt_cutoff=mt_cutoff,
    min_cells=min_cells,
    n_top_genes=n_top_genes,
    n_pcs=n_pcs,
)

# Each preprocessing config gets its own subdirectory under the user's out_dir
_preproc_subdir = preproc_subdir_name(_current_setup)
_effective_out_dir = str(Path(out_dir) / _preproc_subdir)

if out_dir_unset:
    st.error(
        "⚠️ **Output directory is not set.** Replace the `?` in the setup "
        "expander's 'Output directory' field with a real name (e.g. "
        "`results/brain` or `results/my-run`). All compute / load buttons "
        "stay disabled until you do."
    )
else:
    st.info(
        f"**{len(combos)} UMAP combination(s)** → saved to "
        f"`{_effective_out_dir}/`. "
        "Colors are recomputed live in Section 4 (no auto-caching); "
        "use the 💾 Save button there to persist a specific coloring."
    )
    st.caption(
        f"💡 The preprocessing subdirectory `{_preproc_subdir}/` is created "
        "automatically. Different preprocessing configs land in different "
        "subdirs — you can switch between them by changing the setup values."
    )

# Setup fingerprint check — warns if existing results in this folder were
# computed with different preprocessing settings.
_stored_setup = (read_setup(_effective_out_dir)
                 if Path(_effective_out_dir).exists() else None)
_setup_diffs = setup_diff(_stored_setup, _current_setup)

if _setup_diffs:
    with st.container(border=True):
        st.error(
            "⚠️ **Cache setup mismatch — existing results in this directory "
            "were computed with different settings.** Loading them would "
            "produce inconsistent plots. Either pick a different output "
            "directory, or tick **Force recompute** below to overwrite."
        )
        st.markdown("**Differences:**")
        diff_rows = []
        for k, sv, cv in _setup_diffs:
            diff_rows.append({
                "Parameter": k,
                "Stored (on disk)": "—" if sv is None else str(sv),
                "Current (sidebar)": "—" if cv is None else str(cv),
            })
        st.table(diff_rows)

force_recompute = st.checkbox(
    "Force recompute everything (ignore cache)",
    value=bool(_setup_diffs),  # auto-tick when setup mismatch
    help="Default OFF: UMAPs already on disk are reused instead of "
         "recomputed. Tick this to redo everything from scratch "
         "(e.g. after changing preprocessing). Auto-enabled when a setup "
         "mismatch is detected.",
)
# If setup mismatches, refuse to use cache even if user un-ticks force_recompute
_setup_block_cache = bool(_setup_diffs) and not force_recompute
skip_existing = not force_recompute and not _setup_block_cache

if _setup_block_cache:
    st.warning(
        "Setup mismatch unresolved — Force recompute will be required to "
        "start a new computation. Or change the **Output directory** in the "
        "sidebar to a fresh path."
    )


def _pid_is_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError, ValueError):
        return False


# ── Show any existing/running background job in this preproc subdir ─────────
_status_path = Path(_effective_out_dir) / "status.json"
if _status_path.exists():
    try:
        _status = json.loads(_status_path.read_text())
    except Exception:
        _status = None
else:
    _status = None

if _status:
    state = _status.get("state", "?")
    pid = _status.get("pid")
    alive = _pid_is_alive(pid)

    if state in ("running", "starting") and alive:
        completed = _status.get("completed", 0)
        total = _status.get("total_combos", 1) or 1
        elapsed = _status.get("elapsed_seconds", 0)
        eta = _status.get("eta_seconds") or 0
        current = _status.get("current") or "—"
        with st.container(border=True):
            st.markdown(f"🌙 **Background job running** · PID `{pid}`")
            st.progress(min(completed / total, 1.0))
            st.caption(
                f"{completed} / {total} combos · last: `{current}` · "
                f"elapsed {_fmt_duration(elapsed)} · ETA {_fmt_duration(eta)}"
            )
            bc1, bc2, bc3 = st.columns(3)
            if bc1.button("🔄 Refresh status"):
                st.rerun()
            if bc2.button("📂 Load partial results"):
                cc_loaded, ec_loaded = load_from_disk(
                    _effective_out_dir, combos)
                if cc_loaded is not None:
                    st.session_state["color_cache"] = cc_loaded
                    st.session_state["embedding_cache"] = ec_loaded
                    st.success("Loaded partial results from disk.")
            if bc3.button("⏹ Stop job"):
                try:
                    os.kill(int(pid), 15)  # SIGTERM
                    st.warning(f"Sent SIGTERM to PID {pid}.")
                except (OSError, ProcessLookupError, ValueError) as e:
                    st.info(f"Could not signal process: {e}")

    elif state == "done":
        with st.container(border=True):
            st.success(
                f"✓ Background job finished — {_status.get('total_combos', '?')} "
                f"combos in {_fmt_duration(_status.get('elapsed_seconds', 0))}."
            )
            if st.button("📂 Load results", type="primary"):
                cc_loaded, ec_loaded = load_from_disk(
                    _effective_out_dir, combos)
                if cc_loaded is not None:
                    st.session_state["color_cache"] = cc_loaded
                    st.session_state["embedding_cache"] = ec_loaded
                    st.success("Loaded from disk.")
                    st.rerun()

    elif state == "error":
        with st.container(border=True):
            st.error(f"Background job failed: "
                     f"{_status.get('error', 'unknown error')}")
            log_p = Path(_effective_out_dir) / "batch.log"
            if log_p.exists():
                st.caption(f"Log: `{log_p}`")

    elif state in ("running", "starting") and not alive:
        st.warning(
            f"`status.json` says state=`{state}` but PID {pid} is not running. "
            "Job likely crashed without finishing. Check `batch.log` or restart."
        )

# ── Run mode + buttons ───────────────────────────────────────────────────────
run_mode = st.radio(
    "Run mode",
    options=["foreground", "background"],
    index=0,
    horizontal=True,
    format_func=lambda x: (
        "🟢 Foreground (run inside Streamlit)" if x == "foreground"
        else "🌙 Background (overnight / headless)"
    ),
    help=(
        "**Foreground:** computation runs inside Streamlit; tab must stay open. "
        "Good for short runs (a few combos).\n\n"
        "**Background:** spawns `compute_batch.py` as a separate process. "
        "Streamlit can be closed; the job keeps running. Status is polled "
        "from `status.json`. Recommended for big datasets / many combos / "
        "overnight runs."
    ),
)

btn1, btn2 = st.columns(2)

_compute_disabled = _setup_block_cache or out_dir_unset

if run_mode == "foreground":
    if btn1.button(
        "▶ Start foreground computation",
        type="primary",
        disabled=_compute_disabled,
    ):
        Path(_effective_out_dir).mkdir(parents=True, exist_ok=True)
        write_setup(_effective_out_dir, _current_setup)
        cc_new, ec_new = run_all(
            adata_base, combos, n_pcs, _effective_out_dir,
            skip_existing=skip_existing)
        st.session_state["color_cache"] = cc_new
        st.session_state["embedding_cache"] = ec_new
else:
    if btn1.button(
        "🌙 Launch background job",
        type="primary",
        disabled=_compute_disabled,
    ):
        # If a job is already running here, refuse to start another one
        if _status and _status.get("state") in ("running", "starting") and \
                _pid_is_alive(_status.get("pid")):
            st.error(
                f"A background job is already running in `{_effective_out_dir}` "
                f"(PID {_status.get('pid')}). Stop it first or pick another "
                "output directory."
            )
        else:
            Path(_effective_out_dir).mkdir(parents=True, exist_ok=True)
            write_setup(_effective_out_dir, _current_setup)
            input_path = Path(_effective_out_dir) / "_input.h5ad"

            with st.spinner(
                "Saving preprocessed dataset (so the batch script can read it)…"
            ):
                adata_base.write_h5ad(input_path)

            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "compute_batch.py"),
                "--input", str(input_path),
                "--out", str(_effective_out_dir),
                "--min-dists", ",".join(str(x) for x in min_dists),
                "--n-neighbors", ",".join(str(x) for x in nn_list),
                "--metrics", ",".join(metrics),
                "--center-around", center_around,
                "--n-pcs", str(int(n_pcs)),
                "--skip-preprocess",  # adata_base already preprocessed
            ]
            if force_recompute:
                cmd.append("--force-recompute")

            log_path = Path(_effective_out_dir) / "batch.log"
            log_handle = open(log_path, "w")  # noqa: SIM115
            proc = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # detach: survives Streamlit close
                cwd=str(Path(__file__).resolve().parent),
            )

            st.success(
                f"🌙 Background job launched · PID `{proc.pid}` · "
                f"logs at `{log_path}`. You can close this tab.\n\n"
                "Re-open the app later and click **Load results** below "
                "(or just hit **Refresh status**)."
            )
            # give the subprocess a moment to write status.json
            time.sleep(1.0)
            st.rerun()

if btn2.button("⟳ Load existing results from disk", disabled=out_dir_unset):
    if _setup_diffs:
        st.error(
            "Refusing to load: the cache in this output directory was made "
            "with different preprocessing / settings (see mismatch above). "
            "Pick a different output directory, or **Force recompute** to "
            "overwrite."
        )
    else:
        cc_loaded, ec_loaded = load_from_disk(_effective_out_dir, combos)
        if cc_loaded is not None:
            st.session_state["color_cache"] = cc_loaded
            st.session_state["embedding_cache"] = ec_loaded
            st.success("Loaded from disk.")

if st.session_state["color_cache"] is None:
    st.stop()

color_cache = st.session_state["color_cache"]
embedding_cache = st.session_state["embedding_cache"]


# ── SECTION 4: Explorer ───────────────────────────────────────────────────────
st.header("4 — Explore")
ct_col_name = st.session_state["celltype_col"]
has_spatial_base = "spatial" in adata_base.obsm

# View controls: which plots to show + fit-on-screen toggle.
# ── Compare mode: panels labeled "compare" show either the celltype
# annotation (default) or a second user-chosen coloring (when enabled).
with st.sidebar:
    st.markdown('### 🎨 Explorer controls')
    st.caption('Display + coloring knobs (Section 4). Adjust live without scrolling away from the plots — none of these trigger UMAP recomputation.')

    compare_mode = st.radio(
        "Compare panels with:",
        options=["celltype", "coloring"],
        format_func=lambda x: ("Cell-type annotation (Glasbey)"
                               if x == "celltype" else "Another coloring"),
        horizontal=True, key="compare_mode",
        help="Default: side panels show the cell-type annotation (Glasbey "
             "palette) for visual reference. Switch to **Another coloring** "
             "to compare two clrmappy colorings (e.g. OKhsl vs RGB) on the "
             "same UMAP / spatial layout.",
    )

    # Second color picker (only when compare_mode == 'coloring')
    # Mirrors the main coloring panel one-to-one so the comparison can use any
    # OKhsl algorithm + brightness/saturation combination, not just defaults.
    compare_spec = None
    compare_label = "compare"
    if compare_mode == "coloring":
        with st.container(border=True):
            st.markdown("**Compare coloring settings**")
            cmp_top = st.columns([2, 1.2, 1.2])
            with cmp_top[0]:
                cmp_cs = st.radio(
                    "Colorspace",
                    ["okhsl", "rgb", "cielab"],
                    format_func=lambda x: {"okhsl": "OKhsl",
                                           "rgb": "RGB (naive)",
                                           "cielab": "CIELab (U-CIE)"}[x],
                    horizontal=True, key="cmp_color_mode",
                )

            # Per-mode defaults (kept up-to-date so we can build a spec at the end)
            cmp_okhsl_base_3d = True
            cmp_okhsl_iso = True
            cmp_okhsl_pc_from_2d = False
            cmp_okhsl_equal_var = False
            cmp_brightness = [0.2, 0.8]
            cmp_saturation = [0.0, 1.0]
            cmp_sat_enhance = True
            cmp_rgb_equal_var = False

            if cmp_cs == "okhsl":
                base_col, _ = st.columns([2, 3])
                with base_col:
                    cmp_base_choice = st.radio(
                        "Calculate compare colors from",
                        ["3D UMAP", "2D UMAP"],
                        horizontal=True, key="cmp_okhsl_base",
                    )
                    cmp_okhsl_base_3d = (cmp_base_choice == "3D UMAP")

                if cmp_okhsl_base_3d:
                    st.markdown("**Compare 3D OKhsl options**")
                    t_cols = st.columns(3)
                    with t_cols[0]:
                        cmp_okhsl_iso = st.toggle(
                            "Saturation optimization algorithm",
                            value=True, key="cmp_okhsl_iso",
                        )
                    with t_cols[1]:
                        cmp_okhsl_pc_from_2d = st.toggle(
                            "Use 2D-Embedding PCs for hue",
                            value=False, key="cmp_okhsl_pc_from_2d",
                        )
                    with t_cols[2]:
                        cmp_okhsl_equal_var = st.toggle(
                            "Equal-variance rotation",
                            value=False, key="cmp_okhsl_equal_var",
                        )

                sl_cols = st.columns(2)
                with sl_cols[0]:
                    cmp_sat_enhance = st.checkbox(
                        "Apply saturation enhancement",
                        value=True, key="cmp_okhsl_sat_enabled",
                    )
                    cmp_saturation = list(st.slider(
                        "Compare saturation range",
                        min_value=0.0, max_value=1.0,
                        value=tuple(st.session_state.get(
                            "cmp_okhsl_sat_range", (0.0, 1.0))),
                        step=0.05,
                        key="cmp_okhsl_sat_range",
                        disabled=not cmp_sat_enhance,
                    ))
                with sl_cols[1]:
                    cmp_brightness = list(st.slider(
                        "Compare brightness range",
                        min_value=0.0, max_value=1.0,
                        value=tuple(st.session_state.get(
                            "cmp_okhsl_bright_range", (0.2, 0.8))),
                        step=0.05,
                        key="cmp_okhsl_bright_range",
                    ))

                compare_spec = {
                    "mode": "okhsl",
                    "base": "3d" if cmp_okhsl_base_3d else "2d",
                    "iso": cmp_okhsl_iso,
                    "pc_from_2d": cmp_okhsl_pc_from_2d,
                    "equal_var": cmp_okhsl_equal_var,
                    "brightness": cmp_brightness,
                    "saturation": cmp_saturation,
                    "sat_enhance": cmp_sat_enhance,
                }
                # Build short compare label for plot titles + multiselect
                algo_bits = []
                if cmp_okhsl_base_3d:
                    if cmp_okhsl_iso:
                        algo_bits.append("sat-opt")
                    if cmp_okhsl_pc_from_2d:
                        algo_bits.append("hue-2D")
                    if cmp_okhsl_equal_var:
                        algo_bits.append("eqv-rot")
                compare_label = (
                    f"OKhsl {'3D' if cmp_okhsl_base_3d else '2D'}"
                    + (" " + ",".join(algo_bits) if algo_bits else "")
                )

            elif cmp_cs == "rgb":
                cmp_rgb_equal_var = st.toggle(
                    "Equal-variance rotation",
                    value=False, key="cmp_rgb_equal_var",
                )
                compare_spec = {
                    "mode": "rgb", "equal_var": cmp_rgb_equal_var,
                }
                compare_label = (
                    "RGB equal-var" if cmp_rgb_equal_var else "RGB naive")

            else:  # cielab
                st.caption(
                    "CIELab (U-CIE) — no tunable parameters; the `ucie` "
                    "R package fits the gamut internally.")
                compare_spec = {"mode": "cielab"}
                compare_label = "CIELab"

    view_c1, view_c2 = st.columns([3, 2])
    with view_c1:
        # Show the active colorspace inline so the multiselect chip reads
        # "UMAP (OKhsl)" instead of the generic "UMAP (colored)".
        _MAIN_LABEL = {
            "okhsl": "OKhsl",
            "rgb": "RGB naive",
            "cielab": "CIELab",
        }.get(st.session_state.get("color_mode", "okhsl"), "colored")
        _CMP_LABEL = (compare_label if compare_mode == "coloring"
                      else "cell types")
        _PLOT_LABELS = {
            "umap_color": f"UMAP ({_MAIN_LABEL})",
            "spatial_color": f"Spatial ({_MAIN_LABEL})",
            "umap_compare": f"UMAP ({_CMP_LABEL})",
            "spatial_compare": f"Spatial ({_CMP_LABEL})",
        }
        _valid_opts = list(_PLOT_LABELS.keys())
        # Migrate legacy ids (umap_celltype / spatial_celltype were the old
        # names for what's now umap_compare / spatial_compare).
        _legacy_id_map = {
            "umap_celltype": "umap_compare",
            "spatial_celltype": "spatial_compare",
        }
        _stored = st.session_state.get("_selected_plots", _valid_opts)
        _stored_migrated = [_legacy_id_map.get(p, p) for p in _stored]
        _stored_filtered = [p for p in _stored_migrated
                            if p in _valid_opts] or _valid_opts
        # Detect compare_mode change (celltype ↔ coloring) and force-reset the
        # multiselect to all 4 panels. Streamlit's widget keeps its previous
        # state from session_state even if we change `default=`, so we have to
        # mutate session_state AND st.rerun() to make the widget re-read it.
        _prev_cmp_mode = st.session_state.get("_prev_compare_mode")
        st.session_state["_prev_compare_mode"] = compare_mode
        if _prev_cmp_mode is not None and _prev_cmp_mode != compare_mode:
            st.session_state["_selected_plots"] = list(_valid_opts)
            st.rerun()
        if _stored != _stored_filtered:
            st.session_state["_selected_plots"] = _stored_filtered
        selected_plot_ids = st.multiselect(
            "Plots to display",
            options=_valid_opts,
            default=_stored_filtered,
            format_func=lambda k: _PLOT_LABELS[k],
            key="_selected_plots",
            help="Pick which panels to render. Two plots are shown side-by-side "
                 "per row — the order you pick = the order they're rendered.",
        )
    with view_c2:
        _fit_on_screen = st.toggle(
            "🖥️ Fit on screen",
            value=st.session_state.get("_fit_on_screen", False),
            help="Caps each plot's on-screen height (via CSS) so the visible "
                 "panels fit in the viewport without scrolling. Plots shrink "
                 "proportionally — no distortion. The figures themselves are "
                 "unchanged (downloaded PNG keeps full resolution).",
        )
        st.session_state["_fit_on_screen"] = _fit_on_screen
    if _fit_on_screen:
        st.markdown(
            """
            <style>
            /* Cap pyplot/image height to fit panels in viewport. */
            [data-testid="stPyplotChart"] img,
            [data-testid="stImage"] img {
                max-height: 38vh !important;
                width: auto !important;
                object-fit: contain !important;
                display: block !important;
                margin-left: auto !important;
                margin-right: auto !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # ── Color-mode controls (3 modes: OKhsl / RGB / CIELab) ────────────────────
    # All colors are recomputed LIVE on every interaction (no per-variant cache).
    # Only the underlying UMAPs are cached on disk (they're expensive); colors
    # are cheap and let the sliders update in real time.
    st.subheader("Coloring")
    top = st.columns([2, 1.2, 1.2])
    with top[0]:
        colorspace_sel = st.radio(
            "Colorspace",
            ["okhsl", "rgb", "cielab"],
            format_func=lambda x: {
                "okhsl": "OKhsl",
                "rgb": "RGB (naive)",
                "cielab": "CIELab (U-CIE)",
            }[x],
            horizontal=True,
            key="color_mode",
            help=(
                "**OKhsl** — Björn Ottosson's perceptually uniform HSL-like color "
                "space. The clrmappy pipeline:\n"
                "1. Fit a PCA on the 3D UMAP (or 2D UMAP for the 2D variant) and "
                "center it (`mid` or `mean`).\n"
                "2. Optionally apply a rotation algorithm (*saturation "
                "optimization*) that scales the z-axis to the requested "
                "brightness range while keeping saturation maximal — minimizing "
                "the distortion that pure saturation enhancement would otherwise "
                "introduce.\n"
                "3. Optionally enhance saturation by mapping the radial "
                "distance `r_z` onto the requested saturation range.\n"
                "4. Convert (h = PC1/PC2 angle, s = r_z, l = PC3) → OKhsl → sRGB.\n\n"
                "**RGB (naive)** — direct min-max scaling of the 3D UMAP axes to "
                "the R / G / B channels. Quick baseline but perceptually uneven.\n\n"
                "**CIELab (U-CIE)** — fits the 3D UMAP cloud into the "
                "perceptually uniform CIELab gamut via the R `ucie` package "
                "(Kourmpetis et al., "
                "[mikelkou/ucie](https://github.com/mikelkou/ucie))."
            ),
        )
    with top[1]:
        dot_size_val = st.number_input(
            "Dot size (0 = auto)", value=0.0, min_value=0.0, step=0.2,
            help="Scatter dot size in points². 0 = auto-scaled based on cell count "
                 "(smaller dots for large datasets).",
        )
        dot_size = float(dot_size_val) if dot_size_val > 0 else None
    with top[2]:
        plot_dpi = int(st.number_input(
            "DPI", value=220, min_value=80, max_value=500, step=20,
            help="Higher = sharper, slower. 100=draft, 220=default, 300+=print",
        ))

    # ── Per-mode parameter block ─────────────────────────────────────────────────
    # Initialize parameters with safe defaults; only the active mode's widgets
    # are rendered, so values from inactive modes carry forward unchanged.
    okhsl_base_3d = True       # True → compute coloring from 3D UMAP, False → 2D
    okhsl_equal_var = False
    okhsl_pc_from_2d = False
    okhsl_iso_rot = True
    brightness_range = [0.2, 0.8]
    saturation_range = [0.0, 1.0]
    saturation_enabled = True
    rgb_equal_var = False

    if colorspace_sel == "okhsl":
        base_col, _ = st.columns([2, 3])
        with base_col:
            okhsl_base_choice = st.radio(
                "Calculate colors from",
                ["3D UMAP", "2D UMAP"],
                horizontal=True, key="okhsl_base_choice",
                help="**3D UMAP** (recommended) — hue from PC1/PC2 angle, lightness "
                     "from PC3. Three coloring axes → richer distinction. "
                     "**2D UMAP** — only hue + saturation from 2D PCA, lightness "
                     "is constant. Use when no 3D UMAP is available.",
            )
            okhsl_base_3d = (okhsl_base_choice == "3D UMAP")

        if okhsl_base_3d:
            st.markdown("**3D OKhsl options**")
            t_cols = st.columns(3)
            with t_cols[0]:
                okhsl_iso_rot = st.toggle(
                    "Saturation optimization algorithm",
                    value=True, key="okhsl_iso_rot",
                    help="**Distinct from the 'Apply saturation enhancement' "
                         "checkbox below** — this is a rotation algorithm.\n\n"
                         "Searches for the optimal rotation of the 3D-PCA cloud "
                         "around the y-axis so that the saturation (radial "
                         "distance `r_z`) is **as large as possible without "
                         "distortion** while still mapping the z-axis into the "
                         "requested brightness range. Effectively it minimizes "
                         "the distortion that pure saturation enhancement would "
                         "otherwise introduce — the rotation trades unused "
                         "brightness headroom for extra saturation. "
                         "Off → simple min-max z-axis scaling instead. "
                         "(Code: `iso_rot_scale`)",
                )
            with t_cols[1]:
                okhsl_pc_from_2d = st.toggle(
                    "Use 2D-Embedding PCs for hue",
                    value=False, key="okhsl_pc_from_2d",
                    help="If ON: replace PC1+PC2 of the 3D-PCA with the 2D-UMAP's "
                         "PCs (only PC3 / lightness still comes from the 3D UMAP). "
                         "Aligns color hue with what you visually see in 2D — "
                         "useful when 2D and 3D UMAP layouts disagree. "
                         "(Code: `pc1_and_2_from_2dumap`)",
                )
            with t_cols[2]:
                okhsl_equal_var = st.toggle(
                    "Equal-variance rotation",
                    value=False, key="okhsl_equal_var",
                    help="Applies a fixed 45° rotation around all three axes so "
                         "every axis ends up with similar variance. Gives a more "
                         "balanced color distribution when one UMAP axis dominates. "
                         "(Code: `equal_variance_mode`)",
                )

        # Saturation + brightness range sliders
        sl_cols = st.columns(2)
        with sl_cols[0]:
            saturation_enabled = st.checkbox(
                "Apply saturation enhancement",
                value=True, key="okhsl_sat_enabled",
                help="When OFF, the radial distance `r_z` is used as-is (raw "
                     "saturation from the PCA) and the slider below is ignored. "
                     "When ON, `r_z` is min-max-scaled into the picked range, "
                     "which boosts low-saturation points and makes colors more "
                     "vivid (at the cost of some distortion).",
            )
            saturation_range = list(st.slider(
                "Saturation range",
                min_value=0.0, max_value=1.0,
                value=tuple(st.session_state.get(
                    "okhsl_sat_range", (0.0, 1.0))),
                step=0.05,
                key="okhsl_sat_range",
                disabled=not saturation_enabled,
                help="Min / max saturation after radial-distance scaling. "
                     "`[0.0, 1.0]` stretches the data's r_z to span the full "
                     "saturation range (most vivid). `[min, 1.0]` only stretches "
                     "the top end. Move the bottom handle up to force a minimum "
                     "saturation on washed-out points. Disabled when "
                     "**Apply saturation enhancement** is off.",
            ))
        with sl_cols[1]:
            brightness_range = list(st.slider(
                "Brightness range",
                min_value=0.0, max_value=1.0,
                value=tuple(st.session_state.get(
                    "okhsl_bright_range", (0.2, 0.8))),
                step=0.05,
                key="okhsl_bright_range",
                help="Lightness scaling for PC3 (z-axis in OKhsl). `[0, 1]` would "
                     "produce pure black at the bottom and pure white at the top "
                     "(loses detail). The default `[0.2, 0.8]` keeps colors "
                     "readable. Narrow the range to flatten brightness, widen it "
                     "for more dramatic shading.",
            ))

    elif colorspace_sel == "rgb":
        rgb_cols = st.columns([2, 3])
        with rgb_cols[0]:
            rgb_equal_var = st.toggle(
                "Equal-variance rotation",
                value=False, key="rgb_equal_var",
                help="If ON, a 3D PCA is fit to the UMAP and then rotated 45° "
                     "around all axes so every R/G/B channel ends up with similar "
                     "variance (more balanced colors). If OFF, the 3D UMAP axes "
                     "are min-max scaled directly to R/G/B. "
                     "(Code: `equal_variance_mode`)",
            )

    else:  # cielab
        st.info(
            "**CIELab (U-CIE)** — perceptually uniform color space. The 3D UMAP "
            "cloud is fit into the CIELab gamut via the R "
            "[`ucie`](https://github.com/mikelkou/ucie) package "
            "(Kourmpetis et al.). No tunable parameters — the package handles "
            "rotation, scaling and gamut clipping internally."
        )

    # Single-combo display: pick exactly one (min_dist, n_neighbors, metric)
    # from what's currently in the cache. Range-sliders intentionally not used —
    # only one combo is shown at a time. Switch via the pickers below.
    st.subheader("Pick a combination to display")
    st.caption(
        "Single combo at a time. Compute multiple combos via the sidebar's "
        "comma-separated `min_dist` / `n_neighbors`, then switch between them here."
    )

    # Show ALL combos in the cache (not just those matching the sidebar's product),
    # so the Quick-View flow surfaces every loaded combo. Parse the combo keys back
    # into (md, nn, metric) tuples.
    _combo_pat = re.compile(r"^md([\d.]+)_nn(\d+)_m(.+)$")
    _avail = set()
    for _ck in color_cache.keys():
        _m = _combo_pat.match(_ck)
        if _m:
            try:
                _avail.add((float(_m.group(1)), int(_m.group(2)), _m.group(3)))
            except ValueError:
                pass
    _avail = sorted(_avail)
    avail_mds = sorted({p[0] for p in _avail})
    avail_nns = sorted({p[1] for p in _avail})
    avail_ms = sorted({p[2] for p in _avail})

    if not (avail_mds and avail_nns and avail_ms):
        st.warning("Cache is empty for the current sidebar settings. "
                   "Run computation or change the sidebar values to match cached "
                   "combos.")
        st.stop()

    pk1, pk2, pk3 = st.columns(3)
    with pk1:
        if len(avail_mds) >= 2:
            sel_md = st.select_slider(
                "min_dist", options=avail_mds,
                value=avail_mds[0], key="explorer_md",
                help="Single value — switch with the slider.",
            )
        else:
            st.markdown(f"**min_dist:** `{avail_mds[0]}`")
            sel_md = avail_mds[0]
    with pk2:
        if len(avail_nns) >= 2:
            sel_nn = st.select_slider(
                "n_neighbors", options=avail_nns,
                value=avail_nns[0], key="explorer_nn",
                help="Single value — switch with the slider.",
            )
        else:
            st.markdown(f"**n_neighbors:** `{avail_nns[0]}`")
            sel_nn = avail_nns[0]
    with pk3:
        if len(avail_ms) >= 2:
            # Default to 'euclidean' when it's in the cache.
            _default_metric_idx = (avail_ms.index("euclidean")
                                   if "euclidean" in avail_ms else 0)
            sel_metric = st.radio(
                "metric", options=avail_ms,
                index=_default_metric_idx,
                key="explorer_metric",
                help="Single metric — pick one.",
            )
        else:
            st.markdown(f"**metric:** `{avail_ms[0]}`")
            sel_metric = avail_ms[0]

    _only_ck = combo_key(sel_md, sel_nn, sel_metric)
    if _only_ck not in color_cache:
        st.warning(
            f"Combo `{_only_ck}` was not computed. Pick a different value above "
            "or run computation for this combination."
        )
        st.stop()

    # ── Optional: separate UMAP for DISPLAY (panels + interactive 3D) ──
    # Colors are always computed from the picker above. The DISPLAY UMAP can
    # come from a different combo if you want to see how your coloring lands
    # on a different layout (different min_dist / n_neighbors / metric).
    with st.expander(
        "🎯 Use a different UMAP for display (optional)", expanded=False,
    ):
        st.caption(
            "By default, the same combo is used for **color computation** and "
            "for the **displayed 2D / 3D UMAP**. Tick below to override the "
            "displayed UMAP independently — useful when the layout you used "
            "for color computation isn't the cleanest one to *look* at."
        )
        _override_display = st.checkbox(
            "Override displayed UMAP",
            key="display_override_on", value=False,
        )
        if _override_display:
            dp1, dp2, dp3 = st.columns(3)
            with dp1:
                disp_md = (st.select_slider(
                    "Display min_dist", options=avail_mds,
                    value=sel_md, key="display_md")
                    if len(avail_mds) >= 2 else avail_mds[0])
            with dp2:
                disp_nn = (st.select_slider(
                    "Display n_neighbors", options=avail_nns,
                    value=sel_nn, key="display_nn")
                    if len(avail_nns) >= 2 else avail_nns[0])
            with dp3:
                disp_metric = (st.radio(
                    "Display metric", options=avail_ms,
                    index=avail_ms.index(sel_metric)
                    if sel_metric in avail_ms else 0,
                    key="display_metric")
                    if len(avail_ms) >= 2 else avail_ms[0])
            display_ck = combo_key(disp_md, disp_nn, disp_metric)
            if display_ck not in embedding_cache:
                st.warning(f"Display combo `{display_ck}` not in cache — falling "
                           "back to the coloring combo.")
                display_ck = _only_ck
        else:
            display_ck = _only_ck

selected_combos = [_only_ck]
st.markdown(
    f"<small>Showing combo <code>min_dist={sel_md}</code>, "
    f"<code>n_neighbors={sel_nn}</code>, <code>metric={sel_metric}</code> · "
    f"adjust controls in the sidebar →</small>",
    unsafe_allow_html=True,
)
st.divider()

# Render the single picked combo with LIVE color recomputation
# (colors are cheap → no need to cache per-variant on disk).
for ck in selected_combos:
    md, nn, met = sel_md, sel_nn, sel_metric
    # Colors come from `ck` (the coloring combo). The displayed UMAPs come
    # from `display_ck` (potentially different — see "Override displayed
    # UMAP" expander above).
    emb2d = embedding_cache[display_ck]["2d"]
    emb3d = embedding_cache[display_ck]["3d"]
    # We still need the COLORING UMAPs to feed into compute_color_full,
    # since the colors are derived from those (not from the display ones).
    color_emb2d = embedding_cache[ck]["2d"]
    color_emb3d = embedding_cache[ck]["3d"]

    # ── Live color compute on every render. Colors are NOT cached on disk
    # by default — recompute is fast enough and we save disk space. The
    # "💾 Save this coloring" button below lets you persist a specific
    # combination explicitly when you want to.
    # Build the spec describing the current UI state (used both for compute
    # and for the optional Save button).
    if colorspace_sel == "okhsl":
        spec = {
            "mode": "okhsl",
            "base": "3d" if okhsl_base_3d else "2d",
            "iso": okhsl_iso_rot,
            "pc_from_2d": okhsl_pc_from_2d,
            "equal_var": okhsl_equal_var,
            "brightness": list(brightness_range),
            "saturation": list(saturation_range),
            "sat_enhance": bool(saturation_enabled),
        }
    elif colorspace_sel == "rgb":
        spec = {"mode": "rgb", "equal_var": rgb_equal_var}
    else:  # cielab
        spec = {"mode": "cielab"}

    try:
        rgb_arr, emb_fit = compute_color_full(
            spec, color_emb2d, color_emb3d, center_around)
    except Exception as e:
        st.error(f"Color computation failed: {e}")
        continue
    color_stem = color_file_stem(spec)

    # Build a human-readable descriptor for plot titles + download filenames
    if colorspace_sel == "okhsl":
        base_lbl = "3D" if okhsl_base_3d else "2D"
        algo_bits = []
        if okhsl_base_3d:
            if okhsl_iso_rot:
                algo_bits.append("sat-opt algo")
            if okhsl_pc_from_2d:
                algo_bits.append("hue from 2D")
            if okhsl_equal_var:
                algo_bits.append("equal-var rot")
        if not saturation_enabled:
            algo_bits.append("no sat enhance")
        algo_part = ", " + ", ".join(algo_bits) if algo_bits else ""
        # OKhsl 2D has no PC3, so brightness is constant for every cell —
        # show the constant value (mid of the brightness range) instead of
        # a range, since the slider's min/max collapse to a single L value.
        if okhsl_base_3d:
            bright_part = (
                f"bright {brightness_range[0]:.2f}–{brightness_range[1]:.2f}"
            )
        else:
            _const_bright = (brightness_range[0] + brightness_range[1]) / 2
            bright_part = f"bright = {_const_bright:.2f} (constant)"
        descriptor = (
            f"OKhsl {base_lbl}{algo_part} · "
            f"sat {saturation_range[0]:.2f}–{saturation_range[1]:.2f} · "
            f"{bright_part}"
        )
    elif colorspace_sel == "rgb":
        descriptor = (
            "RGB (equal-var rot)" if rgb_equal_var else "RGB naive")
    else:
        descriptor = "CIELab (U-CIE)"

    umap_params = f"min_dist={md}, n_neighbors={nn}, metric={met}"
    header = f"{umap_params}  ·  {descriptor}"

    with st.expander(header, expanded=True):
        if rgb_arr is None:
            st.warning(f"Color computation returned nothing.")
            continue

        hex_colors = to_hex(rgb_arr)

        # Save-to-cache button: persists the current rgb_arr as a .npy in
        # the combo's folder, using the spec-encoded filename. Useful when
        # you've found a specific OKhsl/RGB/CIELab config you want to keep
        # without re-running the algorithm next session.
        _save_target = Path(_effective_out_dir) / ck / f"{color_stem}.npy"
        _save_exists = _save_target.exists()
        sv_col, sv_caption = st.columns([1.2, 4])
        with sv_col:
            if st.button(
                ("✓ Cached on disk" if _save_exists
                 else "💾 Save this coloring"),
                key=f"save_color_{ck}_{color_stem}",
                disabled=_save_exists,
                help=(f"Saved already: `{_save_target.name}`."
                      if _save_exists
                      else "Persist the current color array to disk as "
                           f"`{_save_target.name}` so future loads from this "
                           "folder pick it up instantly. Re-computing is "
                           "fast enough that caching is optional."),
            ):
                _save_target.parent.mkdir(parents=True, exist_ok=True)
                np.save(_save_target, rgb_arr)
                st.success(f"Saved → `{_save_target}`")
                st.rerun()
        with sv_caption:
            st.caption(
                f"File stem: `{color_stem}` — encodes algorithm + brightness "
                f"+ saturation settings for unique identification."
            )

        # Two-line title: descriptor on top, UMAP params below
        params_line = f"min_dist={md} · n_neighbors={nn} · metric={met}"
        title_umap = f"2D UMAP — {descriptor}\n{params_line}"
        title_spatial = f"Spatial transcriptomics — {descriptor}\n{params_line}"
        title_umap_ct = (
            f"2D UMAP — colored by cell-type column “{ct_col_name}”\n{params_line}"
            if ct_col_name else None
        )
        title_spatial_ct = (
            f"Spatial transcriptomics — colored by cell-type column "
            f"“{ct_col_name}”\n{params_line}"
            if ct_col_name else None
        )

        # Build only the panels the user picked, then render them in rows
        # of two columns each. With exactly 2 picked → one wide side-by-side
        # row (perfect for comparing two UMAPs or two Spatial plots).
        spatial_coords = (adata_base.obsm["spatial"]
                          if has_spatial_base else None)
        celltypes = (adata_base.obs[ct_col_name].values
                     if ct_col_name and ct_col_name in adata_base.obs.columns
                     else None)

        # Build a dict of available builders, then iterate over the user's
        # selected_plot_ids in their multiselect ORDER — that way the layout
        # follows the user's pick order (left→right, row-major). Panels that
        # require missing data (e.g. spatial / celltype) are silently skipped.
        _builders = {}
        _builders["umap_color"] = (
            lambda: make_scatter(
                emb2d, hex_colors, title_umap,
                dot_size=dot_size, dpi=plot_dpi),
            plot_filename(
                "UMAP2D", md, nn, met, color_descriptor=descriptor),
        ) if True else None
        if spatial_coords is not None:
            _builders["spatial_color"] = (
                lambda: make_scatter(
                    spatial_coords, hex_colors, title_spatial,
                    spatial=True, dot_size=dot_size, dpi=plot_dpi),
                plot_filename(
                    "Spatial", md, nn, met, color_descriptor=descriptor),
            )
        # Compare panel content depends on `compare_mode`.
        if compare_mode == "celltype":
            if celltypes is not None:
                _builders["umap_compare"] = (
                    lambda: make_celltype_scatter(
                        emb2d, celltypes, title_umap_ct,
                        dot_size=dot_size, dpi=plot_dpi),
                    plot_filename(
                        "UMAP2D", md, nn, met, celltype_col=ct_col_name),
                )
            if spatial_coords is not None and celltypes is not None:
                _builders["spatial_compare"] = (
                    lambda: make_celltype_scatter(
                        spatial_coords, celltypes, title_spatial_ct,
                        spatial=True, dot_size=dot_size, dpi=plot_dpi),
                    plot_filename(
                        "Spatial", md, nn, met, celltype_col=ct_col_name),
                )
        else:  # compare_mode == "coloring": run the second color pipeline
            try:
                rgb_compare, _ = compute_color_full(
                    compare_spec, color_emb2d, color_emb3d, center_around)
                hex_compare = to_hex(rgb_compare)
                ttl_umap_cmp = (
                    f"2D UMAP — {compare_label}\n{params_line}")
                ttl_spatial_cmp = (
                    f"Spatial — {compare_label}\n{params_line}")
                _builders["umap_compare"] = (
                    lambda: make_scatter(
                        emb2d, hex_compare, ttl_umap_cmp,
                        dot_size=dot_size, dpi=plot_dpi),
                    plot_filename(
                        "UMAP2D", md, nn, met,
                        color_descriptor=compare_label),
                )
                if spatial_coords is not None:
                    _builders["spatial_compare"] = (
                        lambda: make_scatter(
                            spatial_coords, hex_compare, ttl_spatial_cmp,
                            spatial=True, dot_size=dot_size, dpi=plot_dpi),
                        plot_filename(
                            "Spatial", md, nn, met,
                            color_descriptor=compare_label),
                    )
            except Exception as _cmp_err:
                st.warning(f"Compare coloring failed: {_cmp_err}")

        # Now follow the user's multiselect order
        panels = [_builders[pid] for pid in selected_plot_ids
                  if pid in _builders]

        if not panels:
            st.info(
                "No panels selected. Pick one or more in the **Plots to "
                "display** dropdown above.")
        else:
            # Render in rows of 2. Trailing single panel gets an empty
            # right column so it stays at half width (consistent with the
            # paired-comparison size when exactly 2 panels are selected).
            for row_start in range(0, len(panels), 2):
                row = panels[row_start:row_start + 2]
                cols = st.columns(2)
                for col, (build_fig, fname) in zip(cols, row):
                    with col:
                        _render_plot(build_fig(), fname)

        # ── Phase 3 — interactive 3D UMAP + OKhsl fit embedding ──────────
        st.divider()
        st.subheader("🔄 Interactive 3D view")
        st.caption(
            "Rotate / zoom / pan with your mouse. Colored with the currently "
            "selected colorspace (no celltype reference here)."
            + (" The **OKhsl fit embedding** plot below shows the calculated, fitted"
               "cloud the OKhsl conversion was applied to. This is primarily for surveillance — useful for "
               "judging how strongly saturation optimization or saturation "
               "enhancement deformed the data or for seeing what the causes were for specific color gradients ."
               if colorspace_sel == "okhsl" else "")
        )

        # Fit-embedding is available for both OKhsl (PCA-rotated cloud) and
        # RGB (min-max-scaled cloud — same coords as the colors).
        # CIELab fits inside R, no exposed fit array.
        _has_fit = (colorspace_sel in ("okhsl", "rgb")
                    and emb_fit is not None)
        if _has_fit:
            viz_left, viz_right = st.columns(2)
        else:
            viz_left = st.container()
            viz_right = None

        # ── Left: the raw 3D UMAP, colored by the current rgb_arr ──
        with viz_left:
            _n = len(emb3d)
            _dot_3d = 1.5 if _n < 30_000 else (1.0 if _n < 100_000 else 0.6)
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=emb3d[:, 0], y=emb3d[:, 1], z=emb3d[:, 2],
                mode="markers",
                marker=dict(size=_dot_3d, color=to_hex(rgb_arr),
                            opacity=0.85),
                hovertemplate=(
                    "UMAP 1: %{x:.2f}<br>"
                    "UMAP 2: %{y:.2f}<br>"
                    "UMAP 3: %{z:.2f}<extra></extra>"
                ),
            )])
            fig_3d.update_layout(
                title=f"3D UMAP — {descriptor}",
                scene=dict(
                    xaxis_title="UMAP 1",
                    yaxis_title="UMAP 2",
                    zaxis_title="UMAP 3",
                    aspectmode="data",
                ),
                height=600,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_3d, use_container_width=True,
                            key=f"plotly3d_{ck}_{color_stem}")

        # ── Right: the fit embedding the coloring was computed on. ──
        # OKhsl 3D → PCA-rotated cloud (x, y, z axes; z = brightness)
        # OKhsl 2D → 2D matplotlib scatter (brightness is constant)
        # RGB    → min-max-scaled UMAP cloud (r, g, b axes mapping directly)
        if _has_fit:
            with viz_right:
                if colorspace_sel == "rgb":
                    # RGB fit: each axis was rescaled to [0, 1] and used as
                    # one of the R / G / B channels directly.
                    ax_titles = ("r", "g", "b")
                    fit_title = (
                        f"RGB fit embedding "
                        f"({'equal-var rot' if rgb_equal_var else 'min-max'})"
                    )
                else:
                    ax_titles = ("x", "y", "z (brightness)")
                    fit_title = "OKhsl fit embedding (PCA-rotated, 3D)"

                if emb_fit.shape[1] == 3:
                    fig_fit = go.Figure(data=[go.Scatter3d(
                        x=emb_fit[:, 0], y=emb_fit[:, 1], z=emb_fit[:, 2],
                        mode="markers",
                        marker=dict(size=_dot_3d, color=to_hex(rgb_arr),
                                    opacity=0.85),
                        hovertemplate=(
                            f"{ax_titles[0]}: %{{x:.3f}}<br>"
                            f"{ax_titles[1]}: %{{y:.3f}}<br>"
                            f"{ax_titles[2]}: %{{z:.3f}}<extra></extra>"
                        ),
                    )])
                    fig_fit.update_layout(
                        title=fit_title,
                        scene=dict(
                            xaxis_title=ax_titles[0],
                            yaxis_title=ax_titles[1],
                            zaxis_title=ax_titles[2],
                            aspectmode="data",
                        ),
                        height=600,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig_fit, use_container_width=True,
                                    key=f"plotly_fit_{ck}_{color_stem}")
                else:  # (N, 2) — 2D fit embedding, brightness is constant
                    _const_b = (brightness_range[0] + brightness_range[1]) / 2
                    fit_title_2d = (
                        f"OKhsl fit embedding (2D PCA after r_z scaling)\n"
                        f"brightness = {_const_b:.2f} (constant, no PC3)"
                    )
                    fig_fit_2d = make_scatter(
                        emb_fit, to_hex(rgb_arr), fit_title_2d,
                        spatial=False,
                        dot_size=dot_size, dpi=plot_dpi,
                    )
                    _render_plot(
                        fig_fit_2d,
                        plot_filename(
                            "OKhslFit2D", md, nn, met,
                            color_descriptor=descriptor),
                    )
