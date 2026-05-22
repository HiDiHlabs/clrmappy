#!/usr/bin/env python
"""Headless batch script for clrmappy — same logic as Section 3 of the Streamlit
app, but without Streamlit / browser. Lets you run very large parameter sweeps
overnight via tmux / nohup / caffeinate.

Example
-------

    caffeinate -i -s python compute_batch.py \\
        --input ST-Data/nuclei.h5ad \\
        --out results/clrmappy/brain_overnight \\
        --min-dists 0.01,0.1,0.3,0.5,0.8 \\
        --n-neighbors 5,15,30,50,100 \\
        --metrics euclidean,cosine

…or let the Streamlit app launch it for you (Section 3 → run mode "Background").

A `status.json` is written next to the output, which the Streamlit app polls
to show live progress. Logs go to stdout (or `batch.log` when launched from
the app).

Phase 2: only UMAPs are persisted on disk (one ``emb2d.npy`` + ``emb3d.npy``
per combo). Colors are recomputed live in the explorer.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from itertools import product
from pathlib import Path

# Make the `clrmappy` package importable (parent of this folder) and the
# sibling `compute_core.py` module (this folder).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import scanpy as sc  # noqa: E402

import clrmappy as cm  # noqa: E402
from compute_core import (  # noqa: E402
    build_setup_dict,
    combo_key,
    process_combo,
    write_setup,
)


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m"


def write_status_atomic(path: Path, payload: dict):
    """Atomic status write so the Streamlit app never reads a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(
        description="clrmappy headless batch computation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to input .h5ad file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--min-dists", required=True,
                   help="Comma-separated min_dist values (e.g. 0.01,0.1,0.3)")
    p.add_argument("--n-neighbors", required=True,
                   help="Comma-separated n_neighbors values (e.g. 15,30,50)")
    p.add_argument("--metrics", default="euclidean",
                   help="Comma-separated metrics, e.g. euclidean,cosine,correlation")
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--center-around", default="mid", choices=["mean", "mid"])
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip cm.preprocess (input already has X_pca or scaled layer)")
    p.add_argument("--force-recompute", action="store_true",
                   help="Ignore cached .npy files on disk — recompute every "
                        "UMAP from scratch. Default: incremental, only "
                        "missing UMAPs are computed.")
    # Preprocessing knobs (only used when --skip-preprocess is NOT set)
    p.add_argument("--min-genes", type=int, default=20)
    p.add_argument("--max-genes", type=int, default=200)
    p.add_argument("--mt-cutoff", type=int, default=5)
    p.add_argument("--min-cells", type=int, default=100)
    p.add_argument("--n-top-genes", type=int, default=2000)

    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_file = out_dir / "status.json"

    mds = [float(x) for x in args.min_dists.split(",") if x.strip()]
    nns = [int(x) for x in args.n_neighbors.split(",") if x.strip()]
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    combos = list(product(mds, nns, metrics))

    t_start = time.time()
    durations: list[float] = []

    def write(state, completed, current=None, eta=None, error=None, extra=None):
        d = {
            "state": state,                # 'starting' | 'running' | 'done' | 'error'
            "total_combos": len(combos),
            "completed": completed,
            "current": current,
            "elapsed_seconds": time.time() - t_start,
            "eta_seconds": eta,
            "pid": os.getpid(),
            "started_at": t_start,
            "out_dir": str(out_dir.resolve()),
            "args": {
                "min_dists": mds, "n_neighbors": nns,
                "metrics": metrics,
                "n_pcs": args.n_pcs, "center_around": args.center_around,
                "skip_preprocess": args.skip_preprocess,
            },
        }
        if error:
            d["error"] = error
        if extra:
            d.update(extra)
        write_status_atomic(status_file, d)

    write("starting", 0)

    try:
        print(f"[clrmappy batch] Loading {args.input}…", flush=True)
        adata = sc.read_h5ad(args.input)
        print(f"  → {adata.shape[0]:,} cells × {adata.shape[1]:,} genes",
              flush=True)

        if not args.skip_preprocess:
            print("[clrmappy batch] Preprocessing…", flush=True)
            adata = cm.preprocess(
                adata,
                min_genes=args.min_genes, min_cells=args.min_cells,
                max_genes=args.max_genes, mt_cutoff=args.mt_cutoff,
                n_top_genes=args.n_top_genes,
            )
            sc.pp.pca(adata, layer="scaled",
                      svd_solver="arpack", n_comps=args.n_pcs)
        elif "X_pca" not in adata.obsm:
            print("[clrmappy batch] Computing PCA (preprocessing skipped)…",
                  flush=True)
            sc.pp.pca(adata, layer="scaled",
                      svd_solver="arpack", n_comps=args.n_pcs)
        else:
            print("[clrmappy batch] Using existing X_pca.", flush=True)

        # Write setup fingerprint so the Streamlit app can later verify
        # that loaded results match the preprocessing used to produce them.
        setup = build_setup_dict(
            adata,
            input_filename=os.path.basename(args.input),
            skip_preprocess=args.skip_preprocess,
            min_genes=args.min_genes,
            max_genes=args.max_genes,
            mt_cutoff=args.mt_cutoff,
            min_cells=args.min_cells,
            n_top_genes=args.n_top_genes,
            n_pcs=args.n_pcs,
        )
        write_setup(str(out_dir), setup)
        print(f"[clrmappy batch] Wrote setup fingerprint "
              f"(adata={setup['adata_fingerprint']}).", flush=True)

        print(f"[clrmappy batch] Running {len(combos)} UMAP combinations…",
              flush=True)
        write("running", 0)

        for i, (md, nn, metric) in enumerate(combos):
            t0 = time.time()
            ck = combo_key(md, nn, metric)
            print(f"[{i+1}/{len(combos)}] {ck}", flush=True)

            result = process_combo(
                adata, md, nn, metric, args.n_pcs, str(out_dir),
                skip_existing=not args.force_recompute,
            )

            stats = result.get("stats", {})
            tag = "computed" if stats.get("computed_umap") else "cached"
            durations.append(time.time() - t0)
            mean_t = sum(durations) / len(durations)
            eta = mean_t * (len(combos) - i - 1)
            print(
                f"    UMAP {tag} · done in {fmt_duration(durations[-1])} · "
                f"avg {fmt_duration(mean_t)}/combo · "
                f"ETA {fmt_duration(eta)}",
                flush=True,
            )
            write("running", i + 1, current=ck, eta=eta)

        total = time.time() - t_start
        print(f"[clrmappy batch] All done. Total time: {fmt_duration(total)}.",
              flush=True)
        write("done", len(combos))

    except KeyboardInterrupt:
        write("error", len(durations), error="Interrupted (SIGINT)")
        print("[clrmappy batch] Interrupted.", flush=True)
        sys.exit(130)
    except Exception as e:
        traceback.print_exc()
        write("error", len(durations), error=f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
