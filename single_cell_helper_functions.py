# This python file stores the functions, that the user can use
# to generate umaps from single cell/ spatial omics data in order to color them later with clrmappy
# These functions are scanpy/anndata based

from __future__ import annotations
import scanpy as sc


# === PREPROCESSING THE SINGLE CELL/ SPATIAL OMICS DATA ===

def preprocess(adata, min_genes, min_cells, max_genes, mt_cutoff=5,
               n_top_genes=2000, return_stats=False):
    """Standard QC + normalization + HVG selection pipeline.

    If ``return_stats=True``, returns a tuple ``(adata, stats)`` where
    ``stats`` is a list of dicts — one per filtering step — recording
    ``n_cells``, ``n_genes``, ``dropped_cells`` and ``dropped_genes``
    after the step. Default behavior (``return_stats=False``) is unchanged.
    """
    adata.var_names_make_unique()
    adata.X = adata.X.astype("int32")

    stats = [{
        "step": "Initial (raw input)",
        "n_cells": int(adata.shape[0]),
        "n_genes": int(adata.shape[1]),
        "dropped_cells": 0,
        "dropped_genes": 0,
    }]

    # Step 1: filter cells with too few detected genes
    n_cells_before = adata.shape[0]
    sc.pp.filter_cells(adata, min_genes=min_genes)
    stats.append({
        "step": f"filter_cells (n_genes ≥ {min_genes})",
        "n_cells": int(adata.shape[0]),
        "n_genes": int(adata.shape[1]),
        "dropped_cells": int(n_cells_before - adata.shape[0]),
        "dropped_genes": 0,
    })

    # Step 2: filter genes detected in too few cells
    n_genes_before = adata.shape[1]
    sc.pp.filter_genes(adata, min_cells=min_cells)
    stats.append({
        "step": f"filter_genes (n_cells ≥ {min_cells})",
        "n_cells": int(adata.shape[0]),
        "n_genes": int(adata.shape[1]),
        "dropped_cells": 0,
        "dropped_genes": int(n_genes_before - adata.shape[1]),
    })

    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Step 3: drop cells with too many detected genes (likely doublets)
    n_cells_before = adata.shape[0]
    adata = adata[adata.obs.n_genes_by_counts <= max_genes, :].copy()
    stats.append({
        "step": f"max_genes filter (n_genes ≤ {max_genes})",
        "n_cells": int(adata.shape[0]),
        "n_genes": int(adata.shape[1]),
        "dropped_cells": int(n_cells_before - adata.shape[0]),
        "dropped_genes": 0,
    })

    # Step 4: drop cells with high mitochondrial fraction
    n_cells_before = adata.shape[0]
    adata = adata[adata.obs.pct_counts_mt < mt_cutoff, :].copy()
    stats.append({
        "step": f"mt_cutoff filter (mt% < {mt_cutoff})",
        "n_cells": int(adata.shape[0]),
        "n_genes": int(adata.shape[1]),
        "dropped_cells": int(n_cells_before - adata.shape[0]),
        "dropped_genes": 0,
    })

    print("After filtering:", adata.shape)

    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        n_top_genes=n_top_genes,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        flavor="seurat_v3",
    )

    adata.layers["scaled"] = adata.X.copy()
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"], layer="scaled")
    sc.pp.scale(adata, max_value=10, layer="scaled")

    if return_stats:
        return adata, stats
    return adata

# === COMPUTING UMAPs with Scanpy (anndata object) ===


def compute_umaps(adata, n_neighbors_2d=15, n_neighbors_3d=15, min_dist_2d=0.2, min_dist_3d=0.2, metric_2d='euclidean', metric_3d='euclidean', n_pcs=50):
    sc.pp.pca(adata, layer="scaled", svd_solver="arpack", n_comps=n_pcs)

    # 3d Umap
    sc.pp.neighbors(adata, n_neighbors=n_neighbors_3d,
                    n_pcs=n_pcs, metric=metric_3d)
    sc.tl.umap(adata, n_components=3, min_dist=min_dist_3d, random_state=10)
    umap_3d = adata.obsm['X_umap'].copy()
    adata.obsm['X_umap_3d'] = umap_3d.copy()

    # 2D UMAP: standard parameters for visualization
    sc.pp.neighbors(adata, n_neighbors=n_neighbors_2d,
                    n_pcs=n_pcs, metric=metric_2d)
    sc.tl.umap(adata, n_components=2, min_dist=min_dist_2d, random_state=10)

    umap_2d = adata.obsm['X_umap'].copy()
    adata.obsm['X_umap_2d'] = umap_2d.copy()

    return {
        'adata': adata,
        'umap_2d': umap_2d,
        'umap_3d': umap_3d,
    }
