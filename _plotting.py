
# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
#
# Mirror of the Streamlit explorer's plot functions, exposed for notebook use.
# All embedding-plot functions are generic — they work with ANY 2D / 3D
# coordinate array (UMAP, t-SNE, PCA, MDS, your own embedding) — not just
# UMAPs. When given an AnnData, the `obsm_key` parameter picks which
# ``adata.obsm`` entry to plot (default ``X_umap_2d`` / ``X_umap_3d`` for
# back-compat with the single-cell pipeline).
#
# Conventions:
#   * Inputs: either an AnnData (with the expected ``obsm`` key) OR a raw
#     coordinate ndarray + a color array — pick whichever is convenient.
#   * Colors: (N, 3) RGB floats in [0, 1] or a list of '#rrggbb' hex strings.
#     RGB floats are auto-converted to hex.
#   * Spatial plots use ``set_aspect("equal", adjustable="datalim")`` so the
#     axes box stays the same size regardless of the data's actual extent —
#     the data is padded with whitespace instead.
#
# Functions:
#   plot_emb_2d              — 2D scatter of any embedding, colored
#   plot_emb_3d              — interactive 3D scatter (Plotly), colored
#   plot_emb_2d_vs_celltype  — side-by-side: colored vs celltype
#   plot_spatial                   — spatial coordinates scatter, colored
#   plot_spatial_vs_celltype       — side-by-side spatial: colored vs celltype
#   plot_okhsl_fit                 — OKhsl fit embedding (2D or 3D, auto)

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import colorcet as cc


def _auto_dot_size(n_cells):
    """Reasonable scatter marker size based on point count."""
    if n_cells > 100_000:
        return 0.3
    if n_cells > 50_000:
        return 0.6
    if n_cells > 20_000:
        return 1.0
    return 2.0


def _to_hex(color):
    """Accept either an (N, 3) RGB float array or a list of hex strings.
    Returns a list of '#rrggbb' strings."""
    arr = np.asarray(color)
    if arr.dtype.kind in ('f', 'i') and arr.ndim == 2 and arr.shape[1] == 3:
        arr = np.clip(arr, 0.0, 1.0)
        return [
            '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
            for r, g, b in arr
        ]
    return list(color)


def _glasbey_colors(celltypes):
    """Map every unique cell type to a distinct Glasbey color (perceptual,
    color-blind-aware). Returns ``(unique_sorted, color_map, per_cell_colors)``."""
    unique = sorted(set(celltypes))
    cmap = cc.glasbey
    color_map = {ct: cmap[i % len(cmap)] for i, ct in enumerate(unique)}
    return unique, color_map, [color_map[ct] for ct in celltypes]


def _resolve_coords(adata_or_coords, obsm_key, expected_ndim,
                    fn_label='plot_embedding'):
    """Extract a coordinate array from either an AnnData (via ``obsm_key``)
    or a raw ndarray. Validates shape.

    Falls back through a list of common keys when ``obsm_key`` is None —
    this makes the function "just work" for the standard single-cell
    pipeline but lets you override for any other embedding type.
    """
    if hasattr(adata_or_coords, 'obsm'):
        # Auto-detect: try the obsm_key first, then fall back to typical
        # embedding keys. Pick whichever has the expected dimensionality.
        if obsm_key is not None:
            candidate_keys = [obsm_key]
        else:
            if expected_ndim == 2:
                candidate_keys = ['X_umap_2d', 'X_umap', 'X_pca', 'X_tsne']
            else:
                candidate_keys = ['X_umap_3d']
        for k in candidate_keys:
            if k in adata_or_coords.obsm:
                arr = np.asarray(adata_or_coords.obsm[k])
                if arr.ndim == 2 and arr.shape[1] >= expected_ndim:
                    return arr[:, :expected_ndim]
        raise KeyError(
            f"{fn_label}: no {expected_ndim}D embedding found in adata.obsm. "
            f"Tried {candidate_keys}. Pass `obsm_key=...` explicitly to "
            f"select a different key, or pass the coordinates directly as "
            f"an (N, {expected_ndim}) ndarray."
        )
    arr = np.asarray(adata_or_coords)
    if arr.ndim != 2 or arr.shape[1] < expected_ndim:
        raise ValueError(
            f"{fn_label}: expected an (N, {expected_ndim}) ndarray, got "
            f"shape {arr.shape}."
        )
    return arr[:, :expected_ndim]


def load_csv_annotations(csv_path, adata, *,
                         id_col=None, ann_col, replace_map=None,
                         fill_unknown='unknown'):
    """Load celltype annotations from a CSV and align them to ``adata.obs_names``.

    Returns a 1D ndarray of labels (length ``adata.n_obs``) you can pass
    directly to ``plot_*_vs_celltype(... , celltype_col=labels)`` — or
    assign back to ``adata.obs[col]`` if you prefer.

    Parameters
    ----------
    csv_path : str | Path
        Path to the annotation CSV.
    adata : AnnData
        Used to reindex against ``obs_names``.
    id_col : str or None
        Column in the CSV that holds the cell IDs matching
        ``adata.obs_names``. If None, the first column is used as index.
    ann_col : str
        Column in the CSV with the celltype label.
    replace_map : dict or None
        Optional value remapping applied to ``ann_col`` (e.g.
        ``{'Hepatocyte_1': 'Hepa 1_3', 'Hepatocyte_2': 'Hepa 1_3'}``).
    fill_unknown : str
        Label assigned to cells whose ID is missing in the CSV.

    Example
    -------
    >>> labels = cm.load_csv_annotations(
    ...     'liver_annotations.csv', adata,
    ...     ann_col='subcluster',
    ...     replace_map={'Hepatocyte_1': 'Hepa 1_3', 'Hepatocyte_2': 'Hepa 1_3'},
    ... )
    >>> cm.plot_spatial_vs_celltype(adata, rgb, labels)
    """
    df = pd.read_csv(csv_path)
    if id_col is None:
        df = df.set_index(df.columns[0])
    else:
        df = df.set_index(id_col)
    df.index = df.index.astype(str)
    if replace_map:
        df = df.replace(replace_map)
    return (
        df[ann_col]
        .reindex(adata.obs_names.astype(str))
        .fillna(fill_unknown)
        .values
    )


def _resolve_celltypes(adata, celltype_col):
    """Accept either an ``adata.obs`` column name (str) or an array-like of
    per-cell labels. Returns a 1D ndarray of labels with length ``N``.

    Raises a helpful error if the caller passed a hex color list or an
    array of the wrong length (a common mistake — passing ``ann_colors``
    where ``celltype_col`` was expected)."""
    if isinstance(celltype_col, str):
        if celltype_col not in adata.obs.columns:
            raise KeyError(
                f"Column '{celltype_col}' not in adata.obs. Available: "
                f"{list(adata.obs.columns)}. "
                "Tip: you can also pass the celltype labels directly as a "
                "list / array (one entry per cell)."
            )
        return adata.obs[celltype_col].values
    arr = np.asarray(celltype_col)
    n = adata.n_obs if hasattr(adata, 'n_obs') else len(adata)
    if arr.ndim != 1 or len(arr) != n:
        raise ValueError(
            f"celltype_col must be either a str column name in adata.obs, "
            f"or a 1D array-like of length {n} (got shape {arr.shape}). "
            f"You may have accidentally passed `ann_colors` (a list of hex "
            "colors) — pass the raw celltype labels instead, or the obs "
            "column name as a string."
        )
    # Heuristic: catch the common mistake of passing hex-color list where
    # raw celltype labels were expected. >70% '#rrggbb' / '#rrggbbaa' entries
    # almost certainly means the caller passed a glasbey color list.
    if arr.dtype.kind in ('U', 'O') and len(arr) > 0:
        try:
            hex_like = sum(
                1 for v in arr
                if isinstance(v, str) and v.startswith('#') and len(v) in (4, 7, 9)
            )
            if hex_like / len(arr) > 0.7:
                raise ValueError(
                    "celltype_col looks like a list of hex color strings "
                    "(e.g. '#bcb6ff', '#03c600' …) — that's the OUTPUT of a "
                    "glasbey mapping, not the input. Pass the raw celltype "
                    "labels (column in adata.obs, or array of strings/ints), "
                    "and the plot function will do the glasbey mapping itself."
                )
        except TypeError:
            pass
    return arr


def _legend_in_axes(legend_ax, color_map, *, n_cols=5, title='Cell type',
                    fontsize=7, title_fontsize=8):
    """Render the celltype legend in its OWN dedicated axes (typically the
    bottom row of a gridspec). This works far better than bbox_to_anchor +
    tight_layout when the data plots use ``aspect='equal'``, because
    matplotlib otherwise leaves a huge gap between axes box and legend."""
    legend_ax.axis('off')
    patches = [mpatches.Patch(color=col, label=lbl)
               for lbl, col in color_map.items()]
    ncol = min(n_cols, max(2, len(patches)))
    legend_ax.legend(
        handles=patches, title=title,
        loc='center', ncol=ncol,
        fontsize=fontsize, title_fontsize=title_fontsize,
        markerscale=2, frameon=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2D embedding scatter (UMAP / t-SNE / PCA / MDS / anything 2D)
# ─────────────────────────────────────────────────────────────────────────────
def plot_emb_2d(adata_or_coords, color, *,
                obsm_key=None,
                title='2D embedding',
                dim_labels=('Dim 1', 'Dim 2'),
                dot_size=None,
                alpha=0.7,
                figsize=(7, 6),
                dpi=200,
                show=True):
    """Scatter any 2D embedding (UMAP, t-SNE, PCA, MDS, …) colored per point.

    Parameters
    ----------
    adata_or_coords : AnnData or (N, 2) ndarray
        Either an AnnData (the ``obsm_key`` is read from it), or the 2D
        coords directly.
    color : (N, 3) RGB float array or list of hex strings
        One color per point. Floats in [0, 1] get auto-converted to hex.
    obsm_key : str or None
        When ``adata_or_coords`` is an AnnData, picks which ``adata.obsm``
        entry to plot. ``None`` (default) auto-detects: tries
        ``X_umap_2d``, ``X_umap``, ``X_pca``, ``X_tsne`` in that order.
        Pass e.g. ``'X_tsne'`` to force a specific embedding.
    title : str
        Plot title. Override with e.g. ``f'2D PCA — {descriptor}'``.
    dim_labels : 2-tuple of str
        Axis labels — defaults to ``('Dim 1', 'Dim 2')``. Override with
        e.g. ``('UMAP 1', 'UMAP 2')`` or ``('PC1', 'PC2')`` for a more
        specific look.

    Returns the matplotlib ``Figure`` when ``show=False``, otherwise None
    (avoids Jupyter double-render).
    """
    coords = _resolve_coords(adata_or_coords, obsm_key, 2,
                             fn_label='plot_emb_2d')
    hex_colors = _to_hex(color)
    if dot_size is None:
        dot_size = _auto_dot_size(len(coords))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(coords[:, 0], coords[:, 1],
               c=hex_colors, s=dot_size, linewidths=0, alpha=alpha,
               rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(dim_labels[0], fontsize=9)
    ax.set_ylabel(dim_labels[1], fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    if show:
        plt.show()
        return None  # Jupyter would otherwise re-render the returned fig
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3D embedding scatter (interactive Plotly)
# ─────────────────────────────────────────────────────────────────────────────
def plot_emb_3d(adata_or_coords, color, *,
                obsm_key=None,
                title='3D embedding',
                dim_labels=('Dim 1', 'Dim 2', 'Dim 3'),
                dot_size=None,
                opacity=0.85,
                height=700,
                show=True):
    """Interactive 3D embedding scatter (Plotly) — rotate / zoom / pan.

    Parameters
    ----------
    adata_or_coords : AnnData or (N, 3) ndarray
        Either an AnnData (``obsm_key`` picks the entry) or 3D coords.
    color : (N, 3) RGB float array or list of hex strings
    obsm_key : str or None
        When AnnData: which obsm entry to plot. ``None`` defaults to
        ``X_umap_3d``. Pass any other 3D embedding key here.
    title : str
        Plot title.
    dim_labels : 3-tuple of str
        Axis labels — defaults to ``('Dim 1', 'Dim 2', 'Dim 3')``.

    Returns the plotly ``Figure``. With ``show=True`` (default) the figure
    is displayed via ``fig.show()`` and the function returns ``None``.
    Use ``show=False`` to get the Figure back (e.g. for Streamlit's
    ``st.plotly_chart``).
    """
    coords = _resolve_coords(adata_or_coords, obsm_key, 3,
                             fn_label='plot_emb_3d')
    hex_colors = _to_hex(color)
    if dot_size is None:
        n = len(coords)
        dot_size = 1.5 if n < 30_000 else (1.0 if n < 100_000 else 0.6)

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(size=dot_size, color=hex_colors, opacity=opacity),
        hovertemplate=(
            f'{dim_labels[0]}: %{{x:.2f}}<br>'
            f'{dim_labels[1]}: %{{y:.2f}}<br>'
            f'{dim_labels[2]}: %{{z:.2f}}<extra></extra>'
        ),
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=dim_labels[0],
            yaxis_title=dim_labels[1],
            zaxis_title=dim_labels[2],
            aspectmode='data',
        ),
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if show:
        fig.show()
        return None
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Spatial scatter (anchored to adata.obsm['spatial'])
# ─────────────────────────────────────────────────────────────────────────────
def plot_spatial(adata_or_coords, color, *,
                 title='Spatial transcriptomics',
                 dot_size=None,
                 alpha=1.0,
                 figsize=(8, 7),
                 dpi=200,
                 show=True):
    """Scatter the spatial coordinates, colored per point.

    Parameters
    ----------
    adata_or_coords : AnnData or (N, 2) ndarray
        Either an AnnData with ``obsm['spatial']`` set, or the spatial coords.
    color : (N, 3) RGB float array or hex strings.
    """
    if hasattr(adata_or_coords, 'obsm'):
        coords = adata_or_coords.obsm['spatial']
    else:
        coords = np.asarray(adata_or_coords)
    hex_colors = _to_hex(color)
    if dot_size is None:
        dot_size = _auto_dot_size(len(coords))

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(coords[:, 0], coords[:, 1],
               c=hex_colors, s=dot_size, linewidths=0, alpha=alpha,
               rasterized=True)
    ax.set_title(title, fontsize=11)
    # keep aspect 1:1 without shrinking the axes — pad data instead
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    plt.tight_layout()
    if show:
        plt.show()
        return None  # Jupyter would otherwise re-render the returned fig
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side: colored embedding vs celltype annotation
# ─────────────────────────────────────────────────────────────────────────────
def plot_emb_2d_vs_celltype(adata, color, celltype_col, *,
                            obsm_key=None,
                            title_color='2D embedding — colored',
                            title_cell=None,
                            dim_labels=('Dim 1', 'Dim 2'),
                            dot_size=None,
                            alpha=0.7,
                            figsize=(15, 7),
                            dpi=200,
                            show=True):
    """Side-by-side 2D embedding plot: left = unsupervised coloring,
    right = cell-type annotation (Glasbey). The cell-type legend sits
    BELOW the right panel so the colored axes box stays the same size as
    the celltype axes box.

    Parameters
    ----------
    adata : AnnData
        Must have a 2D embedding in ``obsm`` and (if ``celltype_col`` is
        a string) the matching column in ``adata.obs``.
    color : (N, 3) RGB float array or hex strings.
    celltype_col : str  OR  array-like of length N
        Either a column name in ``adata.obs``, OR the celltype labels
        directly as a 1D array (one entry per cell). Do NOT pass a list of
        hex colors — pass the raw labels, the function does its own Glasbey
        mapping.
    obsm_key : str or None
        Which 2D embedding to plot. ``None`` auto-detects (X_umap_2d,
        X_umap, X_pca, X_tsne). Pass any other 2D obsm key explicitly.
    dim_labels : 2-tuple of str
        Axis labels — default ``('Dim 1', 'Dim 2')``.
    """
    coords = _resolve_coords(adata, obsm_key, 2,
                             fn_label='plot_emb_2d_vs_celltype')
    hex_colors = _to_hex(color)
    celltypes = _resolve_celltypes(adata, celltype_col)
    _, color_map, ann_colors = _glasbey_colors(celltypes)
    if dot_size is None:
        dot_size = _auto_dot_size(len(coords))
    if title_cell is None:
        if isinstance(celltype_col, str):
            title_cell = f'2D embedding — colored by "{celltype_col}"'
        else:
            title_cell = '2D embedding — colored by cell type'

    # gridspec: two plot panels on top, dedicated legend axes underneath.
    # height_ratios picks how much vertical space the legend gets — 1 unit
    # legend for every ~5 units of plot is plenty for short celltype names.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1],
                          hspace=0.08, wspace=0.12,
                          left=0.06, right=0.97, top=0.95, bottom=0.04)
    ax_color = fig.add_subplot(gs[0, 0])
    ax_cell = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    for ax, c, ttl in [(ax_color, hex_colors, title_color),
                       (ax_cell, ann_colors, title_cell)]:
        ax.scatter(coords[:, 0], coords[:, 1], c=c,
                   s=dot_size, linewidths=0, alpha=alpha,
                   rasterized=True)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel(dim_labels[0], fontsize=9)
        ax.set_ylabel(dim_labels[1], fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    _legend_in_axes(ax_legend, color_map)
    if show:
        plt.show()
        return None  # Jupyter would otherwise re-render the returned fig
    return fig


def plot_spatial_vs_celltype(adata, color, celltype_col, *,
                             title_color='Spatial — colored',
                             title_cell=None,
                             dot_size=None,
                             alpha=1.0,
                             figsize=(15, 8),
                             dpi=200,
                             show=True):
    """Side-by-side spatial scatter: left = unsupervised coloring, right =
    cell-type annotation (Glasbey). Cell-type legend goes BELOW (same
    rationale as ``plot_emb_2d_vs_celltype``).

    ``celltype_col`` may be a str column in ``adata.obs`` OR a 1D array-like
    of celltype labels (length N). Pass raw labels, not hex colors.
    """
    coords = adata.obsm['spatial']
    hex_colors = _to_hex(color)
    celltypes = _resolve_celltypes(adata, celltype_col)
    _, color_map, ann_colors = _glasbey_colors(celltypes)
    if dot_size is None:
        dot_size = _auto_dot_size(len(coords))
    if title_cell is None:
        if isinstance(celltype_col, str):
            title_cell = f'Spatial — colored by "{celltype_col}"'
        else:
            title_cell = 'Spatial — colored by cell type'

    # gridspec layout: two spatial panels on top, dedicated legend axes
    # underneath. Avoids the giant gap that tight_layout + aspect='equal'
    # otherwise leaves between the plot box and a bbox_to_anchor legend.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1],
                          hspace=0.05, wspace=0.05,
                          left=0.02, right=0.98, top=0.96, bottom=0.04)
    ax_color = fig.add_subplot(gs[0, 0])
    ax_cell = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[1, :])
    for ax, c, ttl in [(ax_color, hex_colors, title_color),
                       (ax_cell, ann_colors, title_cell)]:
        ax.scatter(coords[:, 0], coords[:, 1], c=c,
                   s=dot_size, linewidths=0, alpha=alpha,
                   rasterized=True)
        ax.set_title(ttl, fontsize=11)
        ax.set_aspect('equal', adjustable='datalim')
        ax.axis('off')
    _legend_in_axes(ax_legend, color_map)
    if show:
        plt.show()
        return None  # Jupyter would otherwise re-render the returned fig
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# OKhsl fit-embedding viewer (the cloud OKhsl was applied to)
# ─────────────────────────────────────────────────────────────────────────────
def plot_okhsl_fit(emb_fit, color, *,
                   title=None,
                   constant_brightness=None,
                   dot_size=None,
                   alpha=0.85,
                   figsize=(7, 6),
                   dpi=200,
                   height=700,
                   show=True):
    """Plot the OKhsl fit embedding (the PCA-rotated cloud the OKhsl
    conversion was applied to). Auto-detects 2D vs 3D from the shape:

      * ``emb_fit.shape == (N, 3)`` → interactive 3D Plotly scatter
      * ``emb_fit.shape == (N, 2)`` → 2D matplotlib scatter (in this case
        the brightness was constant — pass it via ``constant_brightness``
        for the title annotation)

    Returns either a matplotlib ``Figure`` (2D) or a plotly ``Figure`` (3D).
    """
    emb_fit = np.asarray(emb_fit)
    hex_colors = _to_hex(color)

    if emb_fit.shape[1] == 3:
        if dot_size is None:
            n = len(emb_fit)
            dot_size = 1.5 if n < 30_000 else (1.0 if n < 100_000 else 0.6)
        fig = go.Figure(data=[go.Scatter3d(
            x=emb_fit[:, 0], y=emb_fit[:, 1], z=emb_fit[:, 2],
            mode='markers',
            marker=dict(size=dot_size, color=hex_colors, opacity=alpha),
            hovertemplate=(
                'x: %{x:.3f}<br>'
                'y: %{y:.3f}<br>'
                'z: %{z:.3f}<extra></extra>'
            ),
        )])
        fig.update_layout(
            title=title or 'OKhsl fit embedding (PCA-rotated, 3D)',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z (brightness)',
                aspectmode='data',
            ),
            height=height,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        if show:
            fig.show()
            return None
        return fig

    # 2D fit embedding — brightness is constant (no PC3 in OKhsl 2D pipeline)
    if dot_size is None:
        dot_size = _auto_dot_size(len(emb_fit))
    ttl = title or 'OKhsl fit embedding (2D PCA after r_z scaling)'
    if constant_brightness is not None:
        ttl += f'\nbrightness = {constant_brightness:.2f} (constant, no PC3)'
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(emb_fit[:, 0], emb_fit[:, 1],
               c=hex_colors, s=dot_size, linewidths=0, alpha=alpha,
               rasterized=True)
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('y', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    if show:
        plt.show()
        return None  # Jupyter would otherwise re-render the returned fig
    return fig
