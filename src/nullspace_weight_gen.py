#!/usr/bin/env python3
"""
nullspace_weight_gen.py

Compute nullspace-ablated weights for behavior directions across layers and
multiple run directories.

Two ablation modes are supported:
  1) pooled (default): build a behavior subspace U_b'(ℓ) by stacking ALL
     available DiffMean vectors for behavior b' across --paths at layer ℓ,
     orthonormalizing (via SVD), then project other behavior weights off this
     subspace:  v_⊥ = v - U (Uᵀ v).
  2) rank1: classic rank-1 ablation using the single vector from the SAME base
     path:     v_⊥ = v - u (uᵀ v)/(uᵀ u).

Inputs (per behavior key):
    SYC: {path}/wDiffMean_raw_syc_L{l}.npy
    GA : {path}/wDiffMean_raw_ga_L{l}.npy
    PR : {path}/wDiffMean_raw_pr_L{l}.npy

Outputs (per base path):
    pooled mode: {outdir}/{basename}/{b}_minus_{bprime}Subspace_L{L}.npy
    rank1  mode: {outdir}/{basename}/{b}_minus_{bprime}_L{L}.npy
and a per-layer JSON summary with metrics & provenance.

Example usage:
    python nullspace_weight_gen.py \
      --layer-min 5 --layer-max 80 \
      --paths runs/qwen3_30/cities_neg runs/qwen3_30/cities_pos \
              runs/qwen3_30/claims runs/qwen3_30/counterfactual \
              runs/qwen3_30/smaller_than runs/qwen3_30/larger_than \
              runs/qwen3_30/sp_en_trans runs/qwen3_30/sp_en_trans_pro \
      --outdir runs/nullspace/ablated --mode pooled

      --outdir runs/nullspace/ablated --mode pooled --compare
"""
from __future__ import annotations

import argparse
import sys
import time
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import csv
from collections import defaultdict

import numpy as np

# Import shared utilities
from .utils import file_io

# -----------------------------
# Helpers
# -----------------------------

def load_vec(path: str) -> Optional[np.ndarray]:
    """Load a vector from file, return None if file doesn't exist."""
    if not file_io.file_exists(path):
        return None
    v = file_io.load_numpy(path)
    return np.asarray(v).reshape(-1)


def ablate_rank1(v: np.ndarray, u: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project v onto the orthogonal complement of u (rank-1).
    Returns v - u * (u·v)/(u·u + eps).
    """
    uu = float(np.dot(u, u))
    if uu <= eps:
        return v.copy()
    uv = float(np.dot(u, v))
    return v - (uv / uu) * u


def build_subspace_svd(cols: List[np.ndarray], tol_scale: float = 1e-8) -> Tuple[np.ndarray, dict]:
    """Given a list of column vectors, build an orthonormal basis U via SVD.
    Returns (U, info). U has shape (d, r). r may be 0 if no columns or all near-zero.
    tol = tol_scale * s_max is used for rank determination.
    info includes raw singular values and the chosen rank.
    """
    if not cols:
        return np.zeros((0, 0), dtype=float), {"rank": 0, "svals": []}
    M = np.column_stack(cols)  # (d, k)
    # Handle degenerate d=0 just in case
    if M.size == 0:
        return np.zeros((0, 0), dtype=float), {"rank": 0, "svals": []}
    U_svd, svals, _ = np.linalg.svd(M, full_matrices=False)
    if svals.size == 0:
        return np.zeros((M.shape[0], 0), dtype=float), {"rank": 0, "svals": []}
    smax = float(svals.max())
    tol = tol_scale * smax if smax > 0 else 0.0
    r = int(np.sum(svals > tol))
    U = U_svd[:, :r] if r > 0 else np.zeros((M.shape[0], 0), dtype=float)
    info = {"rank": r, "svals": svals.tolist(), "k_cols": int(M.shape[1]), "tol": tol}
    return U, info


# -------------------------------------------------------------
# Canonical SVD on unit-normalized columns (across datasets)
# -------------------------------------------------------------
def build_canonical_from_unit_columns(cols: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a list of vectors, normalize each to unit L2, stack into M (d,k),
    and compute thin SVD: M = U S V^T. Returns (U, S, Vt).
    If no valid columns, returns (zeros((0,0)), zeros((0,)), zeros((0,0))).
    """
    vecs = []
    d = None
    for v in cols:
        if v is None:
            continue
        x = np.asarray(v).reshape(-1)
        n = float(np.linalg.norm(x))
        if n <= 1e-12:
            continue
        x = x / n
        if d is None:
            d = x.shape[0]
        if x.shape[0] != d:
            continue
        vecs.append(x)
    if not vecs:
        return np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)
    M = np.column_stack(vecs)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U, s, Vt

# --- Helper: union_orthonormal ---
def union_orthonormal(*mats: np.ndarray, tol_scale: float = 1e-8) -> Tuple[np.ndarray, dict]:
    """Given zero or more (d,r_i) bases or (d,) vectors, stack columns and return an
    orthonormal basis U_union via SVD, with rank chosen by tol_scale * smax.
    Returns (U_union, info) where info contains rank, svals, k_cols.
    """
    cols: List[np.ndarray] = []
    d = None
    for M in mats:
        if M is None:
            continue
        A = np.asarray(M)
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        if A.size == 0:
            continue
        if d is None:
            d = A.shape[0]
        if A.shape[0] != d:
            continue  # skip mismatched dims
        cols.append(A)
    if not cols:
        return np.zeros((0, 0), dtype=float), {"rank": 0, "svals": [], "k_cols": 0}
    M = np.hstack(cols)
    U_svd, svals, _ = np.linalg.svd(M, full_matrices=False)
    smax = float(svals.max()) if svals.size else 0.0
    tol = tol_scale * smax if smax > 0 else 0.0
    r = int(np.sum(svals > tol))
    U = U_svd[:, :r] if r > 0 else np.zeros((M.shape[0], 0), dtype=float)
    return U, {"rank": r, "svals": svals.tolist(), "k_cols": int(M.shape[1]), "tol": tol}


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute nullspace-ablated behavior weights across layers and paths.")
    parser.add_argument("--paths", nargs="*", default=[
        "runs/qwen3_30/cities_neg",
        "runs/qwen3_30/cities_pos",
        "runs/qwen3_30/claims",
        "runs/qwen3_30/counterfactual",
        "runs/qwen3_30/smaller_than",
        "runs/qwen3_30/math",
        "runs/qwen3_30/larger_than",
        "runs/qwen3_30/sp_en_trans",
        "runs/qwen3_30/sp_en_trans_pro",
    ], help="Base directories containing wDiffMean_raw_*_L{l}.npy files.")
    parser.add_argument("--layer-min", type=int, default=1)
    parser.add_argument("--layer-max", type=int, default=80)
    parser.add_argument("--outdir", type=str, default="runs/nullspace/ablated")
    parser.add_argument("--mode", type=str, choices=["pooled", "rank1"], default="pooled",
                        help="'pooled' uses all paths to build a multi-rank subspace per behavior; 'rank1' uses same-path single direction.")
    # Allow behavior template overrides if needed
    parser.add_argument("--tpl-syc", type=str, default="{path}/wDiffMean_raw_syc_L{l}.npy")
    parser.add_argument("--tpl-ga", type=str, default="{path}/wDiffMean_raw_ga_L{l}.npy")
    parser.add_argument("--tpl-pr", type=str, default="{path}/wDiffMean_raw_pr_L{l}.npy")
    parser.add_argument("--svd-tol-scale", type=float, default=1e-8, help="Rank threshold as tol_scale * s_max for pooled subspace.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--compare", action="store_true",
                        help="In pooled mode, also compute rank-1 ablations per base and write similarity metrics vs pooled.")
    # Canonical export options
    parser.add_argument("--export-canon", action="store_true",
                        help="Export canonical behavior directions via SVD of stacked unit-normalized DiffMean columns across datasets.")
    parser.add_argument("--canon-subdir", type=str, default="canonical",
                        help="Subdirectory under outdir to write canonical exports.")
    parser.add_argument("--canon-energy-thresh", type=float, default=0.90,
                        help="Cumulative energy threshold to define r90 for saved basis metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    beh_tpls: Dict[str, str] = {"SYC": args.tpl_syc, "GA": args.tpl_ga, "PR": args.tpl_pr}
    behaviors = ["GA", "PR", "SYC"]  # stable order for summaries
    layers = list(range(args.layer_min, args.layer_max + 1))

    # Aggregate rows for an appendix-ready comparison table
    aggregate_rows: List[Dict[str, object]] = []

    # -------------------------------------------------------------
    # First pass: load all available weights per (layer, behavior, base)
    # -------------------------------------------------------------
    layer_weights: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    for L in layers:
        layer_weights[L] = {b: {} for b in behaviors}
        for base in args.paths:
            for b, tpl in beh_tpls.items():
                path = tpl.format(path=base, l=L)
                vec = load_vec(path)
                if vec is None:
                    if not args.quiet:
                        print(f"[warn] Missing {b} at layer {L} for base {base}")
                    continue
                layer_weights[L][b][base] = vec
        # Optional: sanity check dimensions consistent within a layer
        dims = [v.shape[0] for b in behaviors for v in layer_weights[L][b].values()]
        if len(set(dims)) > 1:
            print(f"[warn] Dimension mismatch at layer {L}: found dims {sorted(set(dims))}. Skipping this layer.")
            layer_weights[L] = {b: {} for b in behaviors}

    # -------------------------------------------------------------
    # For each layer, build pooled subspaces per behavior (if requested)
    # -------------------------------------------------------------
    subspaces: Dict[int, Dict[str, Dict[str, object]]] = {L: {} for L in layers}
    if args.mode == "pooled":
        for L in layers:
            for bprime in behaviors:
                cols = [v for v in layer_weights[L][bprime].values()]
                U, info = build_subspace_svd(cols, tol_scale=args.svd_tol_scale)
                subspaces[L][bprime] = {"U": U, "info": info}
                if not args.quiet:
                    print(f"[L={L:>2}] subspace[{bprime}]: k_cols={info['k_cols'] if 'k_cols' in info else 0}, rank={info['rank']}")

    # -------------------------------------------------------------
    # Produce ablated weights and summaries
    # -------------------------------------------------------------
    for base in args.paths:
        base_name = Path(base).name
        base_out = outdir / base_name
        base_out.mkdir(parents=True, exist_ok=True)
        if not args.quiet:
            print(f"[base] {base} -> {base_out}")

        for L in layers:
            # Skip empty layers
            if all(len(layer_weights[L][b]) == 0 for b in behaviors):
                if not args.quiet:
                    print(f"[skip] No weights at layer {L} (any behavior)")
                continue

            layer_summary = {
                "layer": L,
                "base": base,
                "mode": args.mode,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "argv": sys.argv,
                "python": platform.python_version(),
                "ablations": [],
            }

            for b in behaviors:
                v = layer_weights[L][b].get(base)
                if v is None:
                    continue
                for bprime in behaviors:
                    if b == bprime:
                        continue

                    if args.mode == "rank1":
                        u = layer_weights[L][bprime].get(base)
                        if u is None:
                            continue
                        v_abl = ablate_rank1(v, u)
                        # Rescale ablated vector to match original magnitude for steering
                        nv = float(np.linalg.norm(v)) + 1e-12
                        nv_abl_pre = float(np.linalg.norm(v_abl)) + 1e-12
                        # Log energy removed etc. using nv and nv_abl_pre
                        v_abl = v_abl * (nv / nv_abl_pre)
                        # Assert that the norm matches after rescaling
                        assert np.allclose(np.linalg.norm(v_abl), np.linalg.norm(v), rtol=1e-5), (
                            f"Rescaled vector norm {np.linalg.norm(v_abl)} does not match original vector norm {np.linalg.norm(v)}"
                        )
                        out_name = f"{b}_minus_{bprime}_L{L}.npy"
                        cos_v_u = cosine(v, u)
                        rank_used = 1
                        svals = None
                    else:  # pooled
                        U = subspaces[L].get(bprime, {}).get("U", np.zeros((v.shape[0], 0)))
                        if U.shape[1] == 0:
                            # No subspace available; skip or copy-through
                            # Here we skip saving to avoid confusion.
                            continue
                        # --- pooled ablation ---
                        v_abl = v - U @ (U.T @ v)
                        # Rescale ablated vector to match original magnitude for steering
                        nv = float(np.linalg.norm(v)) + 1e-12
                        nv_abl_pre = float(np.linalg.norm(v_abl)) + 1e-12
                        # Log energy removed etc. using nv and nv_abl_pre
                        v_abl = v_abl * (nv / nv_abl_pre)
                        # Assert that the norm matches after rescaling
                        assert np.allclose(np.linalg.norm(v_abl), np.linalg.norm(v), rtol=1e-5), (
                            f"Rescaled vector norm {np.linalg.norm(v_abl)} does not match original vector norm {np.linalg.norm(v)}"
                        )
                        out_name = f"{b}_minus_{bprime}Subspace_L{L}.npy"
                        cos_v_u = float(np.linalg.norm(U.T @ (v / (np.linalg.norm(v) + 1e-12))))  # == sqrt(energy_in_subspace)
                        rank_used = int(U.shape[1])
                        svals = subspaces[L][bprime]["info"]["svals"] if bprime in subspaces[L] else None

                        # Optionally compute rank-1 comparison (uses same-path u)
                        comp = None
                        if args.compare:
                            u = layer_weights[L][bprime].get(base)
                            if u is not None:
                                # rank-1 ablation
                                v_abl_r1 = ablate_rank1(v, u)
                                # Rescale ablated rank-1 ablation for fair comparison
                                nv_abl_r1_pre = float(np.linalg.norm(v_abl_r1)) + 1e-12
                                v_abl_r1 = v_abl_r1 * (nv / nv_abl_r1_pre)
                                assert np.allclose(np.linalg.norm(v_abl_r1), np.linalg.norm(v), rtol=1e-5), (
                                    f"Rescaled vector norm {np.linalg.norm(v_abl_r1)} does not match original vector norm {np.linalg.norm(v)}"
                                )
                                # Save rank-1 ablated vector for transparency
                                out_name_r1 = f"{b}_minus_{bprime}_Rank1_L{L}.npy"
                                np.save(base_out / out_name_r1, v_abl_r1)

                                # Similarity metrics between pooled and rank-1 ablations
                                nv_abl = float(np.linalg.norm(v_abl)) + 1e-12
                                nv_abl_r1 = float(np.linalg.norm(v_abl_r1)) + 1e-12

                                energy_removed_pooled = 1.0 - (nv_abl * nv_abl) / (nv * nv)
                                energy_removed_r1 = 1.0 - (nv_abl_r1 * nv_abl_r1) / (nv * nv)
                                delta_energy = float(energy_removed_pooled - energy_removed_r1)

                                cos_between_abl = cosine(v_abl, v_abl_r1)
                                rel_l2_diff = float(np.linalg.norm(v_abl - v_abl_r1) / nv)

                                # How much of u lies in the pooled subspace U?
                                u_norm = float(np.linalg.norm(u)) + 1e-12
                                u_in_U = float(np.linalg.norm(U.T @ (u / u_norm)))  # in [0,1]

                                comp = {
                                    "rank1_out": out_name_r1,
                                    "cos_between_abl": float(cos_between_abl),
                                    "rel_l2_diff": float(rel_l2_diff),
                                    "energy_removed_rank1": float(energy_removed_r1),
                                    "energy_removed_pooled": float(energy_removed_pooled),
                                    "delta_energy_removed": float(delta_energy),
                                    "u_fraction_in_pooled_subspace": float(u_in_U),
                                }

                    out_path = base_out / out_name
                    file_io.save_numpy(v_abl, out_path)

                    nv = float(np.linalg.norm(v)) + 1e-12
                    nv_abl = float(np.linalg.norm(v_abl)) + 1e-12
                    energy_removed = 1.0 - (nv_abl * nv_abl) / (nv * nv)

                    # Orthogonality check with the subspace
                    if args.mode == "pooled":
                        proj_norm = float(np.linalg.norm(U.T @ v_abl))
                        orth = proj_norm / (float(np.linalg.norm(v_abl)) + 1e-12)
                    else:
                        orth = abs(float(np.dot(u, v_abl))) / ((float(np.linalg.norm(u)) + 1e-12) * nv_abl) if u is not None else float("nan")

                    entry = {
                        "b": b,
                        "bprime": bprime,
                        "layer": L,
                        "rank_used": rank_used,
                        "k_cols": int(len(layer_weights[L][bprime])),
                        "norm_v": nv,
                        "norm_v_abl": nv_abl,
                        "sqrt_energy_in_subspace": cos_v_u if args.mode == "pooled" else None,
                        "cos_v_u_or_energy_dir": cos_v_u if args.mode == "rank1" else None,
                        "frac_removed_norm": float(1.0 - nv_abl / nv),
                        "energy_removed": float(energy_removed),
                        "orth_check": float(orth),
                    }
                    if svals is not None:
                        entry["svals"] = svals if isinstance(svals, list) else list(svals)
                    if args.mode == "pooled" and 'comp' in locals() and comp is not None:
                        entry["rank1_comparison"] = comp

                    # Append a compact aggregate row (appendix-ready)
                    agg = {
                        "base": base_name,
                        "layer": L,
                        "b": b,
                        "bprime": bprime,
                        "rank_used": rank_used,
                        "k_cols": int(len(layer_weights[L][bprime])),
                        "energy_removed_pooled": float(energy_removed) if args.mode == "pooled" else None,
                    }
                    if args.mode == "pooled" and 'comp' in locals() and comp is not None:
                        agg.update({
                            "energy_removed_rank1": float(comp["energy_removed_rank1"]),
                            "delta_energy_removed": float(comp["delta_energy_removed"]),
                            "cos_between_abl": float(comp["cos_between_abl"]),
                            "rel_l2_diff": float(comp["rel_l2_diff"]),
                        })
                    else:
                        agg.update({
                            "energy_removed_rank1": float(energy_removed) if args.mode == "rank1" else None,
                            "delta_energy_removed": None,
                            "cos_between_abl": None,
                            "rel_l2_diff": None,
                        })
                    aggregate_rows.append(agg)

                    layer_summary["ablations"].append(entry)
                    if not args.quiet:
                        msg = f"[L={L:>2}] {base_name}: {b} - {bprime} ({args.mode}, r={rank_used}) -> {out_name} | energy_removed={energy_removed:.3f} orth≈{orth:.2e}"
                        if args.mode == "pooled" and 'comp' in locals() and comp is not None:
                            msg += f" | r1 cos={comp['cos_between_abl']:.3f} dE={comp['delta_energy_removed']:+.3f} u∈U={comp['u_fraction_in_pooled_subspace']:.2f}"
                        print(msg)
                    continue

                    # rank1 branch output and summary (outside pooled block)
                    out_path = base_out / out_name
                    np.save(out_path, v_abl)

                    nv = float(np.linalg.norm(v)) + 1e-12
                    nv_abl = float(np.linalg.norm(v_abl)) + 1e-12
                    nu = float(np.linalg.norm(layer_weights[L][bprime].get(base))) if args.mode == "rank1" and layer_weights[L][bprime].get(base) is not None else float("nan")
                    energy_removed = 1.0 - (nv_abl * nv_abl) / (nv * nv)
                    # Orthogonality check (rank1: with u; pooled: with the subspace)
                    if args.mode == "rank1":
                        orth_num = abs(float(np.dot(layer_weights[L][bprime][base], v_abl))) if layer_weights[L][bprime].get(base) is not None else float("nan")
                        orth_den = (float(np.linalg.norm(layer_weights[L][bprime][base])) + 1e-12) * nv_abl if layer_weights[L][bprime].get(base) is not None else float("nan")
                        orth = orth_num / orth_den if orth_den == orth_den else float("nan")
                    else:
                        U = subspaces[L][bprime]["U"] if bprime in subspaces[L] else np.zeros((v.shape[0], 0))
                        proj_norm = float(np.linalg.norm(U.T @ v_abl))
                        orth = proj_norm / (float(np.linalg.norm(v_abl)) + 1e-12) if U.shape[1] > 0 else float("nan")

                    entry = {
                        "b": b,
                        "bprime": bprime,
                        "layer": L,
                        "rank_used": rank_used,
                        "k_cols": int(len(layer_weights[L][bprime])) if args.mode == "pooled" else 1,
                        "norm_v": nv,
                        "norm_v_abl": nv_abl,
                        "cos_v_u_or_energy_dir": cos_v_u,
                        "frac_removed_norm": float(1.0 - nv_abl / nv),
                        "energy_removed": float(energy_removed),
                        "orth_check": float(orth),
                    }
                    if svals is not None:
                        entry["svals"] = svals if isinstance(svals, list) else list(svals)

                    # Aggregate row for rank1-only mode (pooled metrics are None)
                    agg = {
                        "base": base_name,
                        "layer": L,
                        "b": b,
                        "bprime": bprime,
                        "rank_used": rank_used,
                        "k_cols": 1,
                        "energy_removed_pooled": None,
                        "energy_removed_rank1": float(energy_removed),
                        "delta_energy_removed": None,
                        "cos_between_abl": None,
                        "rel_l2_diff": None,
                    }
                    aggregate_rows.append(agg)

                    layer_summary["ablations"].append(entry)
                    if not args.quiet:
                        print(f"[L={L:>2}] {base_name}: {b} - {bprime} ({args.mode}, r={rank_used}) -> {out_name} | energy_removed={energy_removed:.3f} orth≈{orth:.2e}")

            # ---------------- Union ablations: b minus the union of the other two behaviors ----------------
            for b in behaviors:
                v = layer_weights[L][b].get(base)
                if v is None:
                    continue
                others = [x for x in behaviors if x != b]
                tag = f"{others[0]}+{others[1]}"

                if args.mode == "rank1":
                    # Build union from available same-base rank1 vectors for the two other behaviors
                    u1 = layer_weights[L][others[0]].get(base)
                    u2 = layer_weights[L][others[1]].get(base)
                    Uu, info_u = union_orthonormal(u1, u2, tol_scale=args.svd_tol_scale)
                    if Uu.shape[1] == 0:
                        continue
                    v_abl = v - Uu @ (Uu.T @ v)
                    # Rescale ablated vector to match original magnitude for steering
                    nv = float(np.linalg.norm(v)) + 1e-12
                    nv_abl_pre = float(np.linalg.norm(v_abl)) + 1e-12
                    v_abl = v_abl * (nv / nv_abl_pre)
                    assert np.allclose(np.linalg.norm(v_abl), np.linalg.norm(v), rtol=1e-5), (
                        f"Rescaled vector norm {np.linalg.norm(v_abl)} does not match original vector norm {np.linalg.norm(v)}"
                    )
                    out_name = f"{b}_minus_{others[0]}{others[1]}_L{L}.npy"
                    out_path = base_out / out_name
                    np.save(out_path, v_abl)

                    nv_abl = float(np.linalg.norm(v_abl)) + 1e-12
                    energy_removed = 1.0 - (nv_abl * nv_abl) / (nv * nv)
                    proj_norm = float(np.linalg.norm(Uu.T @ v_abl))
                    orth = proj_norm / (float(np.linalg.norm(v_abl)) + 1e-12)
                    entry = {
                        "b": b,
                        "bprime": tag,
                        "layer": L,
                        "rank_used": int(Uu.shape[1]),
                        "k_cols": 2,
                        "norm_v": nv,
                        "norm_v_abl": nv_abl,
                        "frac_removed_norm": float(1.0 - nv_abl / nv),
                        "energy_removed": float(energy_removed),
                        "orth_check": float(orth),
                        "union": True,
                    }
                    layer_summary["ablations"].append(entry)

                    # Aggregate row for appendix
                    agg = {
                        "base": base_name,
                        "layer": L,
                        "b": b,
                        "bprime": tag,
                        "rank_used": int(Uu.shape[1]),
                        "k_cols": 2,
                        "energy_removed_pooled": None,
                        "energy_removed_rank1": float(energy_removed),
                        "delta_energy_removed": None,
                        "cos_between_abl": None,
                        "rel_l2_diff": None,
                    }
                    aggregate_rows.append(agg)

                else:  # pooled mode
                    # Build union subspace from pooled U of the two other behaviors
                    U1 = subspaces[L].get(others[0], {}).get("U", None)
                    U2 = subspaces[L].get(others[1], {}).get("U", None)
                    Uu, info_u = union_orthonormal(U1, U2, tol_scale=args.svd_tol_scale)
                    if Uu.shape[1] == 0:
                        continue
                    v_abl = v - Uu @ (Uu.T @ v)
                    # Rescale ablated vector to match original magnitude for steering
                    nv = float(np.linalg.norm(v)) + 1e-12
                    nv_abl_pre = float(np.linalg.norm(v_abl)) + 1e-12
                    v_abl = v_abl * (nv / nv_abl_pre)
                    assert np.allclose(np.linalg.norm(v_abl), np.linalg.norm(v), rtol=1e-5), (
                        f"Rescaled vector norm {np.linalg.norm(v_abl)} does not match original vector norm {np.linalg.norm(v)}"
                    )
                    out_name = f"{b}_minus_{others[0]}{others[1]}Subspace_L{L}.npy"
                    out_path = base_out / out_name
                    np.save(out_path, v_abl)

                    nv_abl = float(np.linalg.norm(v_abl)) + 1e-12
                    energy_removed = 1.0 - (nv_abl * nv_abl) / (nv * nv)
                    proj_norm = float(np.linalg.norm(Uu.T @ v_abl))
                    orth = proj_norm / (float(np.linalg.norm(v_abl)) + 1e-12)

                    entry = {
                        "b": b,
                        "bprime": tag,
                        "layer": L,
                        "rank_used": int(Uu.shape[1]),
                        "k_cols": int((subspaces[L].get(others[0], {}).get("info", {}).get("k_cols", 0)) + (subspaces[L].get(others[1], {}).get("info", {}).get("k_cols", 0))),
                        "norm_v": nv,
                        "norm_v_abl": nv_abl,
                        "frac_removed_norm": float(1.0 - nv_abl / nv),
                        "energy_removed": float(energy_removed),
                        "orth_check": float(orth),
                        "union": True,
                    }

                    # Optional: if --compare, compute *rank-1 union* and attach comparison
                    if args.compare:
                        u1 = layer_weights[L][others[0]].get(base)
                        u2 = layer_weights[L][others[1]].get(base)
                        Uu_r1, _ = union_orthonormal(u1, u2, tol_scale=args.svd_tol_scale)
                        if Uu_r1.shape[1] > 0:
                            v_abl_r1 = v - Uu_r1 @ (Uu_r1.T @ v)
                            # Rescale
                            nv_abl_r1_pre = float(np.linalg.norm(v_abl_r1)) + 1e-12
                            v_abl_r1 = v_abl_r1 * (nv / nv_abl_r1_pre)
                            assert np.allclose(np.linalg.norm(v_abl_r1), np.linalg.norm(v), rtol=1e-5), (
                                f"Rescaled vector norm {np.linalg.norm(v_abl_r1)} does not match original vector norm {np.linalg.norm(v)}"
                            )
                            np.save(base_out / f"{b}_minus_{others[0]}{others[1]}_Rank1Union_L{L}.npy", v_abl_r1)
                            nv_abl_r1 = float(np.linalg.norm(v_abl_r1)) + 1e-12
                            energy_removed_r1 = 1.0 - (nv_abl_r1 * nv_abl_r1) / (nv * nv)
                            cos_between_abl = cosine(v_abl, v_abl_r1)
                            rel_l2_diff = float(np.linalg.norm(v_abl - v_abl_r1) / nv)
                            entry["rank1_union_comparison"] = {
                                "cos_between_abl": float(cos_between_abl),
                                "rel_l2_diff": float(rel_l2_diff),
                                "energy_removed_rank1_union": float(energy_removed_r1),
                                "energy_removed_pooled_union": float(energy_removed),
                                "delta_energy_removed": float(energy_removed - energy_removed_r1),
                            }

                    layer_summary["ablations"].append(entry)

                    # Aggregate row for appendix
                    agg = {
                        "base": base_name,
                        "layer": L,
                        "b": b,
                        "bprime": tag,
                        "rank_used": int(Uu.shape[1]),
                        "k_cols": entry["k_cols"],
                        "energy_removed_pooled": float(energy_removed),
                        "energy_removed_rank1": entry.get("rank1_union_comparison", {}).get("energy_removed_rank1_union") if args.compare else None,
                        "delta_energy_removed": entry.get("rank1_union_comparison", {}).get("delta_energy_removed") if args.compare else None,
                        "cos_between_abl": entry.get("rank1_union_comparison", {}).get("cos_between_abl") if args.compare else None,
                        "rel_l2_diff": entry.get("rank1_union_comparison", {}).get("rel_l2_diff") if args.compare else None,
                    }
                    aggregate_rows.append(agg)

            # Write per-layer summary for this base
            if layer_summary["ablations"]:
                file_io.save_json(layer_summary, base_out / f"summary_L{L}.json")

    # -----------------------------
    # Export canonical behavior directions across datasets (SVD on unit columns)
    # -----------------------------
    if args.export_canon:
        canon_root = outdir / args.canon_subdir
        for L in layers:
            for b in behaviors:
                cols = [v for v in layer_weights[L][b].values()]
                U, s, Vt = build_canonical_from_unit_columns(cols)
                b_dir = canon_root / b
                b_dir.mkdir(parents=True, exist_ok=True)
                meta = {
                    "layer": L,
                    "behavior": b,
                    "k_cols": int(len(cols)),
                    "svals": s.tolist() if s.size else [],
                    "energy_total": float(np.sum(s*s)) if s.size else 0.0,
                    "energy_cum": (np.cumsum((s*s)/ (np.sum(s*s) + 1e-12)).tolist() if s.size else []),
                    "r90": None,
                    "sigma1": float(s[0]) if s.size else 0.0,
                }
                if s.size:
                    energy_cum = np.cumsum(s*s)
                    r90 = int(np.searchsorted(energy_cum, args.canon_energy_thresh * energy_cum[-1]) + 1)
                    meta["r90"] = r90
                    # Save u1 (unit vector) and a σ₁-scaled version for steering
                    u1 = U[:, 0]
                    sigma1 = float(s[0])
                    u1_scaled = u1 * sigma1
                    np.save(b_dir / f"u1_L{L}.npy", u1)               # unit-norm canonical
                    np.save(b_dir / f"u1_scaled_L{L}.npy", u1_scaled) # σ₁-scaled canonical (recommended for steering)

                    # Save basis up to r90 and full U as metadata NPZ
                    U_r = U[:, :r90]
                    np.savez(b_dir / f"basis_L{L}.npz", U=U, U_r=U_r, svals=s, r90=r90, sigma1=sigma1)
                else:
                    # No data; save empty placeholders for consistency
                    np.save(b_dir / f"u1_L{L}.npy", np.zeros((0,), dtype=float))
                    np.save(b_dir / f"u1_scaled_L{L}.npy", np.zeros((0,), dtype=float))
                    np.savez(b_dir / f"basis_L{L}.npz", U=np.zeros((0,0), dtype=float), U_r=np.zeros((0,0), dtype=float), svals=np.zeros((0,), dtype=float), r90=0, sigma1=0.0)
                # Write JSON sidecar with metadata
                file_io.save_json(meta, b_dir / f"meta_L{L}.json")
        if not args.quiet:
            print(f"[canon] Exported canonical directions to {canon_root}")

    # -----------------------------
    # Write appendix-ready aggregates
    # -----------------------------
    if aggregate_rows:
        csv_path = outdir / "overall_comparison.csv"
        fieldnames = [
            "base", "layer", "b", "bprime", "rank_used", "k_cols",
            "energy_removed_pooled", "energy_removed_rank1",
            "delta_energy_removed", "cos_between_abl", "rel_l2_diff",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in aggregate_rows:
                writer.writerow(row)

        # Grouped means by (b, bprime)
        groups = defaultdict(list)
        for row in aggregate_rows:
            key = (row["b"], row["bprime"])
            groups[key].append(row)
        summary_rows = []
        for (b, bprime), rows in groups.items():
            def _mean(key):
                vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
                return float(sum(vals) / len(vals)) if vals else None
            summary_rows.append({
                "b": b,
                "bprime": bprime,
                "mean_energy_removed_pooled": _mean("energy_removed_pooled"),
                "mean_energy_removed_rank1": _mean("energy_removed_rank1"),
                "mean_delta_energy_removed": _mean("delta_energy_removed"),
                "mean_cos_between_abl": _mean("cos_between_abl"),
                "mean_rel_l2_diff": _mean("rel_l2_diff"),
                "count": len(rows),
            })
        summary_path = outdir / "overall_pairwise_means.csv"
        summary_fields = [
            "b", "bprime", "mean_energy_removed_pooled", "mean_energy_removed_rank1",
            "mean_delta_energy_removed", "mean_cos_between_abl", "mean_rel_l2_diff", "count"
        ]
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

        if not args.quiet:
            print(f"[appendix] Wrote {csv_path}")
            print(f"[appendix] Wrote {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()